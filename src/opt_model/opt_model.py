import gurobipy as gp
from gurobipy import GRB

class OptModel:
    """
    Gurobi optimization model for flexible consumer scheduling:
    - Single flexible load, curtailable PV, daily energy target, tariffs, prices.
    """

    def __init__(self, model_data):
        """Initialize and set up the enhanced optimization model."""
        self.T = len(model_data.hours)
        self.data = model_data
        self.m = gp.Model("flexible_consumer_enhanced")

        # --- Variables ---
        
        # Core variables
        self.d = self.m.addMVar(self.T, lb=0, ub=model_data.d_max_t, name="d")
        self.x = self.m.addMVar(self.T, lb=0, name="x")  # No upper bound, penalties handle excess
        self.y = self.m.addMVar(self.T, lb=0, name="y")  # No upper bound, penalties handle excess
        self.s_pv = self.m.addMVar(self.T, lb=0, ub=model_data.s_t, name="s_pv")
        
        # Excess variables for penalty calculation
        self.x_excess = self.m.addMVar(self.T, lb=0, name="x_excess")  # Import above limit
        self.y_excess = self.m.addMVar(self.T, lb=0, name="y_excess")  # Export above limit

        # --- Constraints ---
        
        # Power balance: load met by used PV + net grid import
        self.m.addConstr(self.d == self.s_pv + self.x - self.y, name="power_balance")

        # Daily minimum energy consumption
        self.m.addConstr(gp.quicksum(self.d[t] for t in range(self.T)) >= model_data.d_min_total, 
                        "min_total_load")

        # Excess import/export constraints
        self.m.addConstr(self.x_excess >= self.x - model_data.x_max_t, name="excess_import")
        self.m.addConstr(self.y_excess >= self.y - model_data.y_max_t, name="excess_export")

        # Load ramping constraints (if ramp rates < 1.0)
        if model_data.ramp_up_max_ratio < 1.0 or model_data.ramp_down_max_ratio < 1.0:
            max_ramp_up = model_data.ramp_up_max_ratio * model_data.d_max_t[0]
            max_ramp_down = model_data.ramp_down_max_ratio * model_data.d_max_t[0]
            
            for t in range(1, self.T):
                self.m.addConstr(self.d[t] - self.d[t-1] <= max_ramp_up, 
                               f"ramp_up_{t}")
                self.m.addConstr(self.d[t-1] - self.d[t] <= max_ramp_down, 
                               f"ramp_down_{t}")

        # Minimum load ratio constraint (if load must operate above minimum when on)
        if model_data.d_min_ratio > 0:
            # This would typically require binary variables for on/off state
            # For now, simplified as minimum when non-zero
            for t in range(self.T):
                # If d[t] > 0, then d[t] >= d_min_ratio * d_max[t]
                # This is a logical constraint that would need binary variables for exact implementation
                pass  # Placeholder for more complex on/off logic

        # --- Objective: minimize total daily net cost including penalties ---
        
        regular_cost = gp.quicksum(
            model_data.p_buy_t[t] * self.x[t]
            + model_data.tau_import_t[t] * self.x[t]
            - model_data.p_sell_t[t] * self.y[t]
            + model_data.tau_export_t[t] * self.y[t]
            for t in range(self.T)
        )
        
        penalty_cost = gp.quicksum(
            model_data.penalty_excess_import * self.x_excess[t]
            + model_data.penalty_excess_export * self.y_excess[t]
            for t in range(self.T)
        )
        
        total_cost = regular_cost + penalty_cost
        self.m.setObjective(total_cost, GRB.MINIMIZE)

        # Set solver parameters
        self.m.setParam("OutputFlag", 1)

    def solve(self):
        """Runs optimization and stores enhanced solution."""
        self.m.optimize()
        self.solution = None
        
        if self.m.status == GRB.OPTIMAL:
            self.solution = {
                "d": [self.d[t].X for t in range(self.T)],
                "s_pv": [self.s_pv[t].X for t in range(self.T)],
                "x": [self.x[t].X for t in range(self.T)],
                "y": [self.y[t].X for t in range(self.T)],
                "x_excess": [self.x_excess[t].X for t in range(self.T)],
                "y_excess": [self.y_excess[t].X for t in range(self.T)],
                "obj": self.m.objVal,
                "regular_cost": sum(
                    self.data.p_buy_t[t] * self.x[t].X
                    + self.data.tau_import_t[t] * self.x[t].X
                    - self.data.p_sell_t[t] * self.y[t].X
                    + self.data.tau_export_t[t] * self.y[t].X
                    for t in range(self.T)
                ),
                "penalty_cost": sum(
                    self.data.penalty_excess_import * self.x_excess[t].X
                    + self.data.penalty_excess_export * self.y_excess[t].X
                    for t in range(self.T)
                ),
            }
        
        return self.solution
    
    



class OptModelb1:
    
    def __init__(self, model_data):
        """Initialize and set up the enhanced optimization model."""
        self.T = len(model_data.hours)
        self.data = model_data
        self.m = gp.Model("flexible_consumer_enhanced")

        # --- Variables ---
        
        # Core variables
        self.d = self.m.addMVar(self.T, lb=0, ub=model_data.d_max_t, name="d")
        self.x = self.m.addMVar(self.T, lb=0, name="x")  # No upper bound, penalties handle excess
        self.y = self.m.addMVar(self.T, lb=0, name="y")  # No upper bound, penalties handle excess
        self.s_pv = self.m.addMVar(self.T, lb=0, ub=model_data.s_t, name="s_pv")
        self.z = self.m.addMVar(self.T, lb=0, name="z")
        
        # Excess variables for penalty calculation
        self.x_excess = self.m.addMVar(self.T, lb=0, name="x_excess")  # Import above limit
        self.y_excess = self.m.addMVar(self.T, lb=0, name="y_excess")  # Export above limit

        # --- Constraints ---

        # Daily minimum energy consumption
        self.m.addConstr(gp.quicksum(self.d[t] for t in range(self.T)) >= model_data.d_min_total, 
                        "min_total_load")
        
        # Power balance: load met by used PV + net grid import
        self.m.addConstr(self.d == self.s_pv + self.x - self.y, name="power_balance")

    
        # Excess import/export constraints
        self.m.addConstr(self.x_excess >= self.x - model_data.x_max_t, name="excess_import")
        self.m.addConstr(self.y_excess >= self.y - model_data.y_max_t, name="excess_export")

        # Load ramping constraints (if ramp rates < 1.0)
        if model_data.ramp_up_max_ratio < 1.0 or model_data.ramp_down_max_ratio < 1.0:
            max_ramp_up = model_data.ramp_up_max_ratio * model_data.d_max_t[0]
            max_ramp_down = model_data.ramp_down_max_ratio * model_data.d_max_t[0]
            
            for t in range(1, self.T):
                self.m.addConstr(self.d[t] - self.d[t-1] <= max_ramp_up, 
                               f"ramp_up_{t}")
                self.m.addConstr(self.d[t-1] - self.d[t] <= max_ramp_down, 
                               f"ramp_down_{t}")
                
        # Minimum load ratio constraint (if load must operate above minimum when on)
        if model_data.d_min_ratio > 0:
            # This would typically require binary variables for on/off state
            # For now, simplified as minimum when non-zero
            for t in range(self.T):
                # If d[t] > 0, then d[t] >= d_min_ratio * d_max[t]
                # This is a logical constraint that would need binary variables for exact implementation
                pass  # Placeholder for more complex on/off logic

        
        #Constraints for z to rapresent the absolute value of the load shift
        for t in range(self.T):
            self.m.addConstr(self.z[t] - self.d[t] + model_data.d_given_t[t] >= 0, name=f"abs_pos_{t}")
            self.m.addConstr(self.z[t] + self.d[t] - model_data.d_given_t[t] >= 0, name=f"abs_neg_{t}")



        # --- Objective: minimize total daily net cost including penalties ---
        
        regular_cost = gp.quicksum(
            model_data.p_buy_t[t] * self.x[t]
            + model_data.tau_import_t[t] * self.x[t]
            - model_data.p_sell_t[t] * self.y[t]
            + model_data.tau_export_t[t] * self.y[t]
            for t in range(self.T)
        )
        
        penalty_cost = gp.quicksum(
            model_data.penalty_excess_import * self.x_excess[t]
            + model_data.penalty_excess_export * self.y_excess[t]
            + model_data.p_pen * self.z[t]
            for t in range(self.T)
        )
        
        total_cost = regular_cost + penalty_cost
        self.m.setObjective(total_cost, GRB.MINIMIZE)

        # Set solver parameters
        self.m.setParam("OutputFlag", 1)

    def solve(self):
        """Runs optimization and stores enhanced solution."""
        self.m.optimize()
        self.solution = None
        
        if self.m.status == GRB.OPTIMAL:
            self.solution = {
                "d": [self.d[t].X for t in range(self.T)],
                "s_pv": [self.s_pv[t].X for t in range(self.T)],
                "x": [self.x[t].X for t in range(self.T)],
                "y": [self.y[t].X for t in range(self.T)],
                "x_excess": [self.x_excess[t].X for t in range(self.T)],
                "y_excess": [self.y_excess[t].X for t in range(self.T)],
                "obj": self.m.objVal,
                "regular_cost": sum(
                    self.data.p_buy_t[t] * self.x[t].X
                    + self.data.tau_import_t[t] * self.x[t].X
                    - self.data.p_sell_t[t] * self.y[t].X
                    + self.data.tau_export_t[t] * self.y[t].X
                    for t in range(self.T)
                ),
                "penalty_cost": sum(
                    self.data.penalty_excess_import * self.x_excess[t].X
                    + self.data.penalty_excess_export * self.y_excess[t].X
                    + self.data.p_pen * self.z[t].X

                    for t in range(self.T)
                ),
                "z": [self.z[t].X for t in range(self.T)],
            }
        
        return self.solution
    
    def save_LP_duals(self):
        """
        Extract and save primal and dual values from the solved optimization model.
        Returns:
            tuple: (optimal_objective, optimal_variables, optimal_duals)
        """
        if self.m.status == GRB.OPTIMAL:
            
            
            # Save dual values (shadow prices of constraints)
            optimal_duals = {c.ConstrName: c.Pi for c in self.m.getConstrs()}
            
            
            
            self.optimal_duals = optimal_duals
            
        else:
            print("Optimization was not successful. Cannot extract LP results.")
            
            optimal_duals = None
            

        return  optimal_duals
    
    def print_LP_results(self):
        """
        Nicely print the LP results: objective, decision variable values, and duals.
        """
        if self.m.status == GRB.OPTIMAL:
            print("\n-------------------   RESULTS  -------------------")
            print(f"Optimal objective value: {self.m.objVal:.4f}\n")
            
            if hasattr(self, "solution") and self.solution is not None and "penalty_cost" in self.solution:
                print(f"Total Penalty: {round(self.solution['penalty_cost'], 2)}\n")
            else:
                print("Total Penalty: (not available)\n")
            
            print("Decision variables (primal values):")
            for v in self.m.getVars():
                print(f"  {v.VarName:20s} = {v.X:10.4f}")
            
            print("\nDual variables (shadow prices):")
            for c in self.m.getConstrs():
                print(f"  {c.ConstrName:20s} = {c.Pi:10.4f}")
            
            print("--------------------------------------------------\n")
        
        else:
            print("Optimization was not successful. No results to print.")

