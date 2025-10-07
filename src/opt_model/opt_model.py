import gurobipy as gp
from gurobipy import GRB

class OptModel:
    """
    Gurobi optimization model for flexible consumer scheduling:
    - Single flexible load, curtailable PV, daily energy target, tariffs, prices.
    """

    def __init__(self, model_data, question_name: str):
        """
        Initialize and set up the optimization model.
        Args:
            model_data (ModelData): Output from DataProcessor with scenario data.
        """
        self.T = len(model_data.hours)

        # Store model data for access
        self.data = model_data

        # Start Gurobi model
        self.m = gp.Model("flexible_consumer")

        # --- Variables ---

        # d_t = scheduled flexible load consumption each hour [kWh]
        self.d = self.m.addVars(self.T, lb=0, ub=model_data.d_max_t, name="d")

        # x_t = electricity *imported* from grid each hour [kWh]
        self.x = self.m.addVars(self.T, lb=0, ub=model_data.x_max_t, name="x")

        # y_t = electricity *exported* to grid each hour [kWh]
        self.y = self.m.addVars(self.T, lb=0, ub=model_data.y_max_t, name="y")

        # s_pv_t = PV energy *actually used* in each hour [kWh]
        self.s_pv = self.m.addVars(self.T, lb=0, ub=model_data.s_t, name="s_pv")  # ≤ available PV each hour

        # --- Constraints ---

        # Power balance: load met by *used* PV + net grid import
        for t in range(self.T):
            self.m.addConstr(self.d[t] == self.s_pv[t] + self.x[t] - self.y[t], name=f"balance_{t}")
        
        if question_name in ['1b', '1c']:
            for t in range(self.T):
                # Hourly minimum energy consumption
                self.m.addConstr(self.d[t] >= model_data.d_min_t[t], f"min_load_{t}")
        elif question_name == '1a':
            # Daily minimum energy consumption
            self.m.addConstr(gp.quicksum(self.d[t] for t in range(self.T)) >= model_data.d_min_total, "min_total_load")

        # --- Objective: minimize total daily net cost ---

        cost_expr = gp.quicksum(
            model_data.p_buy_t[t] * self.x[t]
            + model_data.tau_import_t[t] * self.x[t]
            - model_data.p_sell_t[t] * self.y[t]
            + model_data.tau_export_t[t] * self.y[t]
            for t in range(self.T)
        )
        self.m.setObjective(cost_expr, GRB.MINIMIZE)

        # Optional: set output flag as needed
        self.m.setParam("OutputFlag", 1)

    def solve(self):
        """Runs optimization and stores solution."""
        self.m.optimize()
        self.solution = None
        if self.m.status == GRB.OPTIMAL:
            self.solution = {
                "d": [self.d[t].X for t in range(self.T)],
                "s_pv": [self.s_pv[t].X for t in range(self.T)],
                "x": [self.x[t].X for t in range(self.T)],
                "y": [self.y[t].X for t in range(self.T)],
                "obj": self.m.objVal,
            }
        return self.solution
