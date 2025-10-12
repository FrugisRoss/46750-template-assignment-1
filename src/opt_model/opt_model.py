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
        self.m.addConstr(self.d -self.s_pv - self.x + self.y <= 0, name="power_balance_le")
        self.m.addConstr(self.d -self.s_pv - self.x + self.y >= 0, name="power_balance_ue")

    
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

class OptModelc1:
    
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

        # Excess variables for battery operation
        self.b_d = self.m.addMVar(self.T, lb=0,ub=model_data.power_discharge_max_kW, name="b_dis") # Discharging power
        self.b_c = self.m.addMVar(self.T, lb=0,ub=model_data.power_charge_max_kW, name="b_cha") # Charging power
        self.b_soc = self.m.addMVar(self.T, lb=0, ub=model_data.storage_capacity_kWh, name="b_soc") # State of charge

        # --- Constraints ---

        # Daily minimum energy consumption
        self.m.addConstr(gp.quicksum(self.d[t] for t in range(self.T)) >= model_data.d_min_total, 
                        "min_total_load")
        
        # Power balance: load met by used PV + net grid import
        self.m.addConstr(self.d == self.s_pv + self.x - self.y - self.b_c + self.b_d, name="power_balance")
    
        # Excess import/export constraints
        self.m.addConstr(self.x_excess >= self.x - model_data.x_max_t, name="excess_import")
        self.m.addConstr(self.y_excess >= self.y - model_data.y_max_t, name="excess_export")

        # Initial SOC constraint (only once, outside loops)
        self.m.addConstr(self.b_soc[0] == model_data.initial_soc_ratio * model_data.storage_capacity_kWh, name="initial_day_soc")
 
        # Battery discharge limit linked to SOC
        for t in range(self.T):
            if t > 0: # From second time step onwards, use previous SOC
                self.m.addConstr(self.b_d[t] <= self.b_soc[t-1], name=f"discharge_limit_{t}")
            elif t == self.T - 1: # Last time step, ensure final SOC is met
                    self.m.addConstr(self.b_d[0] <= model_data.final_soc_ratio * model_data.storage_capacity_kWh - self.b_soc[t-1], name="discharge_limit_end")
            else: # First time step, use initial SOC
                self.m.addConstr(self.b_d[0] <= model_data.initial_soc_ratio * model_data.storage_capacity_kWh, name="discharge_limit_0")
 
        # Charging cannot exceed remaining SOC capacity
        for t in range(self.T):
            self.m.addConstr(self.b_c[t] <= model_data.storage_capacity_kWh - self.b_soc[t], name=f"charge_limit_{t}")


        
        # SOC evolution for all hours from 1 to T-1
        for t in range(1, self.T):
            self.m.addConstr(self.b_soc[t] == self.b_soc[t-1] +
                            model_data.charging_efficiency * self.b_c[t-1] -
                            (1/model_data.discharging_efficiency) * self.b_d[t-1], name=f"soc_update_{t}")
 
        
        # End of day SOC constraint
        self.m.addConstr(self.b_soc[self.T-1] == model_data.final_soc_ratio * model_data.storage_capacity_kWh, name="end_day_soc")
        


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
                "b_d": [self.b_d[t].X for t in range(self.T)],
                "b_c": [self.b_c[t].X for t in range(self.T)],
                "b_soc": [self.b_soc[t].X for t in range(self.T)],
                "Charging Operation": [self.b_c[t].X - self.b_d[t].X for t in range(self.T)],
                "Price Import": self.data.p_buy_t,
                "Price Export": self.data.p_sell_t,
                "Import Tariff": self.data.tau_import_t,
                "Export Tariff": self.data.tau_export_t,
                "Export_Import": [self.y[t].X - self.x[t].X for t in range(self.T)],
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

class OptModel2b:
    
    def __init__(self, model_data):
        """Initialize and set up the enhanced optimization model."""
        self.T = len(model_data.hours)
        self.data = model_data
        self.m = gp.Model("flexible_consumer_with_investment")
    
        # --- Variables ---
        # --- Investment variable ---
        # s is the scaling factor of the reference battery (continuous, >=0).
        self.s = self.m.addVar(lb=0.0, name="s")
        
        # Core variables operational
        self.d = self.m.addMVar(self.T, lb=0, ub=model_data.d_max_t, name="d")
        self.x = self.m.addMVar(self.T, lb=0, name="x")  # No upper bound, penalties handle excess
        self.y = self.m.addMVar(self.T, lb=0, name="y")  # No upper bound, penalties handle excess
        self.s_pv = self.m.addMVar(self.T, lb=0, ub=model_data.s_t, name="s_pv")
        self.z = self.m.addMVar(self.T, lb=0, name="z")

        # Excess variables for penalty calculation
        self.x_excess = self.m.addMVar(self.T, lb=0, name="x_excess")  # Import above limit
        self.y_excess = self.m.addMVar(self.T, lb=0, name="y_excess")  # Export above limit

        # --- Battery (investment-scaled) variables and constraints ---

        # reference capacities/power
        soc_cap_ref = model_data.storage_capacity_kWh           # SOE^{max}_base
        p_ch_ref = model_data.power_charge_max_kW               # base charge power (kW)
        p_dis_ref = model_data.power_discharge_max_kW           # base discharge power (kW)

        # --- Constraints ---

        # make power variables with no tight ub; actual limits enforced by constraints with `s`
        self.b_c = self.m.addMVar(self.T, lb=0.0, ub=GRB.INFINITY, name="b_charge")    # charging power [kW]
        self.b_d = self.m.addMVar(self.T, lb=0.0, ub=GRB.INFINITY, name="b_discharge") # discharging power [kW]
        # SOC at T+1 time points (0..T)
        self.b_soc = self.m.addMVar(self.T + 1, lb=0.0, ub=GRB.INFINITY, name="b_soc")


        # Daily minimum energy consumption
        self.m.addConstr(gp.quicksum(self.d[t] for t in range(self.T)) >= model_data.d_min_total, 
                        "min_total_load")
        
        # Power balance: load met by used PV + net grid import + battery discharge - battery charge
        self.m.addConstr(self.d == self.s_pv + self.x - self.y - self.b_c + self.b_d, name="power_balance")
    
        # Excess import/export constraints
        self.m.addConstr(self.x_excess >= self.x - model_data.x_max_t, name="excess_import")
        self.m.addConstr(self.y_excess >= self.y - model_data.y_max_t, name="excess_export")


        # Battery technical limits scaled linearly
        for t in range(self.T):
            self.m.addConstr(self.b_c[t] <= self.s * p_ch_ref, name=f"charge_power_limit_{t}")
            self.m.addConstr(self.b_d[t] <= self.s * p_dis_ref, name=f"discharge_power_limit_{t}")

            self.m.addConstr(self.b_soc[t] <= self.s * soc_cap_ref, name=f"soc_limit_{t}")

        # Initial SOC constraint 
        self.m.addConstr(self.b_soc[0] == model_data.initial_soc_ratio * self.s * soc_cap_ref, name="initial_day_soc")
 
        # Battery discharge limit linked to SOC
        for t in range(self.T):
            if t > 0:
                self.m.addConstr(self.b_d[t] <= self.b_soc[t-1], name=f"discharge_limit_{t}")
            else:
                self.m.addConstr(self.b_d[0] <= model_data.initial_soc_ratio * self.s * soc_cap_ref, name="discharge_limit_0")
 
        # Charging cannot exceed remaining SOC capacity
        for t in range(self.T):
            self.m.addConstr(self.b_c[t] <= soc_cap_ref - self.b_soc[t], name=f"charge_limit_{t}")
 
        # SOC evolution for all hours from 1 to T-1
        for t in range(1, self.T):
            self.m.addConstr(self.b_soc[t] == self.b_soc[t-1] +
                            model_data.charging_efficiency * self.b_c[t-1] -
                            (1/model_data.discharging_efficiency) * self.b_d[t-1], name=f"soc_update_{t}")
 
        # End of day SOC constraint
        self.m.addConstr(self.b_soc[self.T-1] == model_data.final_soc_ratio * self.s * model_data.storage_capacity_kWh, name="end_day_soc")


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
        # Investment cost (one-time, scales with battery size)
        investment_cost = model_data.capital_cost_per_kWh * self.s * model_data.storage_capacity_kWh

        total_cost = model_data.N_replications * (regular_cost + penalty_cost) + investment_cost
        
        self.m.setObjective(total_cost, GRB.MINIMIZE)

        # Set solver parameters
        self.m.setParam("OutputFlag", 1)

    def solve(self):
        """Runs optimization and stores enhanced solution."""
        self.m.optimize()
        self.solution = None
        
        if self.m.status == GRB.OPTIMAL:
            s_star = self.s.X
            self.solution = {
                "optimal_scale": s_star,
                "installed_capacity_kWh": s_star * self.data.storage_capacity_kWh,
                "installed_charge_power_kW": s_star * self.data.power_charge_max_kW,
                "installed_discharge_power_kW": s_star * self.data.power_discharge_max_kW,
                "objective_value": self.m.objVal,
                "hourly_dispatch": {
                    "d": [self.d[t].X for t in range(self.T)],
                    "s_pv": [self.s_pv[t].X for t in range(self.T)],
                    "x": [self.x[t].X for t in range(self.T)],
                    "y": [self.y[t].X for t in range(self.T)],
                    "x_excess": [self.x_excess[t].X for t in range(self.T)],
                    "y_excess": [self.y_excess[t].X for t in range(self.T)],
                    "z": [self.z[t].X for t in range(self.T)],
                    "b_d": [self.b_d[t].X for t in range(self.T)],
                    "b_c": [self.b_c[t].X for t in range(self.T)],
                    # Use SOC from 0 to T-1 hours to align with plots
                    "b_soc": [self.b_soc[t].X for t in range(self.T)],
                },
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