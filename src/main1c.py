#%%
from data_ops import DataLoader
from data_ops.data_processor import DataProcessor1c
from data_ops.data_processor import update_penalty_load_shifting
from opt_model.opt_model import OptModelc1
import numpy as np
from data_ops.data_visualizer import plot_column_vs_hours, plot_columns_vs_hours, plot_sensitivity_vs_hours
from pathlib import Path
from matplotlib import pyplot as plt
from data_ops import DataLoader, DataProcessor
import os

# %%
############## Question 1b ##############
# Get the base directory "grandparent"  of the current file
grandparent_dir = Path(__file__).resolve().parents[1]

# %%
############## Question 1a: Single simulation ##############
# 1. Load and process data

#Set the penalty for load shifting
load_shifting_penalty = 3.0  # Penalty cost per kWh of load shifting
# Update the usage_preferences.json file with the new penalty
update_penalty_load_shifting(grandparent_dir / "data" / "question_1c"/"usage_preferences.json", load_shifting_penalty)
# 1. Load and process data
loader = DataLoader(input_path=grandparent_dir / "data" / "question_1c")
raw = loader.get_data()
#print(raw)
processor = DataProcessor1c(raw)
model_data = processor.build_model_data()
#print(model_data)
#print(model_data.d_given_t)
print(model_data.p_pen)


#%%
# 2. Build and solve optimization model
optm = OptModelc1(model_data)
solution = optm.solve()

# Save LP results
duals = optm.save_LP_duals()

# Print LP results
optm.print_LP_results()


# %%
if solution:
    print("Scheduled load each hour:", np.array(solution['d']).round(2).tolist())
    print("PV usage each hour:", np.array(solution['s_pv']).round(2).round(2).tolist())
    print("Grid imports:", np.array(solution['x']).round(2).tolist())
    print("Grid exports:", np.array(solution['y']).round(2).tolist())
    print("Total Penalty:", round(solution['penalty_cost'], 2))
    print("z:", np.array(solution['z']).round(2).tolist())
else:
    print("No feasible solution found.")

# %%
# plot_column_vs_hours(solution, column='d', y_label="Served Load [kWh]", figsize=(10, 4), hour_start=0, ax=None, title="Served Load vs Hour", show=False)
# plot_column_vs_hours(solution, column='z', y_label="Absolute Load Shift [kWh]", figsize=(10, 4), hour_start=0, ax=None, title="Absolute Load Shift vs Hour", show=False)
# plot_column_vs_hours(solution, column='b_d', y_label="Battery Discharge [kWh]", figsize=(10, 4), hour_start=0, ax=None, title="Discharging vs Hour", show=False)
# plot_column_vs_hours(solution, column='b_c', y_label="Battery Charge [kWh]", figsize=(10, 4), hour_start=0, ax=None, title="Charging vs Hour", show=False)
# plot_column_vs_hours(solution, column='b_soc', y_label="Battery SOE [kWh]", figsize=(10, 4), hour_start=0, ax=None, title="SOE vs Hour", show=False)

columns = ['d', 'z', 'b_d', 'b_c', 'b_soc']
labels = { 'd': 'Served Load', 'z': 'Load Shift', 'b_d': 'Battery Discharging Power', 'b_c': 'Battery Charging Power', 'b_soc': 'Battery SOE' }
 
fig = plot_columns_vs_hours(solution,
                            columns=columns,
                            labels=labels,
                            y_label="Electricity [kWh]",
                            title="Do your thing: " + str("Battery Operation"),
                            hour_start=0,
                            figsize=(10, 4),
                            show=False)

columns = ['d', 's_pv', 'x', 'y']
labels = {'s_pv': 'PV production [kWh]',
          'd': 'Electricity consumed [kWh]',
          'x': 'Electricity imported [kWh]',
          'y': 'Electricity exported [kWh]'}
 
fig2 = plot_columns_vs_hours(solution,
                            columns=columns,
                            labels=labels,
                            y_label="Electricity [kWh]",
                            title="Do your thing: " + str("Electricity Balance"),
                            hour_start=0,
                            figsize=(10, 4),
                            show=False)



# %%
plt.show()

# %%
