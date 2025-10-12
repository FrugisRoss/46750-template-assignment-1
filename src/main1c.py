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
load_shifting_penalty = 1.8  # Penalty cost per kWh of load shifting
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

out_dir = grandparent_dir / "Assignments"
os.makedirs(out_dir, exist_ok=True)

columns = ['d', 'Export_Import', 'Charging Operation', 'b_soc', 'Price Import']
labels = { 'd': 'Served Load', 'Export_Import': 'Export/Import(with sign)', 'Charging Operation': 'Battery Operation(with sign)', 'b_soc': 'Battery SOE', 'Price Import': 'Electricity Price'}
 
fig = plot_columns_vs_hours(solution,
                            columns=columns,
                            labels=labels,
                            y_label="Electricity [kWh]",
                            title=str("Battery Operation and price"),
                            hour_start=0,
                            figsize=(10, 4),
                            show=False)

fig.savefig(os.path.join(out_dir, 'battery_operation.pdf'), format='pdf', bbox_inches='tight')

fig1, ax = plt.subplots(figsize=(10, 4))
plot_column_vs_hours(solution, column='d', y_label="Served Load [kWh]", figsize=(10, 4), hour_start=0, ax=ax, title="Served Load vs Hour", show=False)
fig1.savefig(os.path.join(out_dir, 'battery_served_load_vs_hour_base.pdf'), format='pdf', bbox_inches='tight')
#plt.show(fig)

# Absolute load shift plot -> save as vector (SVG and PDF)
fig2, ax2 = plt.subplots(figsize=(10, 4))
plot_column_vs_hours(solution, column='z', y_label="Absolute Load Shift [kWh]", figsize=(10, 4), hour_start=0, ax=ax2, title="Absolute Load Shift vs Hour", show=False)
fig2.savefig(os.path.join(out_dir, 'battery_absolute_load_shift_vs_hour_base.pdf'), format='pdf', bbox_inches='tight')
#plt.show(fig2)


# %%


def run_penalty_sensitivity(penalty_values, json_path, data_path, model_class):
    """
    Runs the optimization for each value in penalty_values.
    
    Args:
        penalty_values (list): List of penalty values to test.
        json_path (str): Path to usage_preferences.json file.
        data_path (str): Path to the folder with data.
        model_class (class): Optimization model class (e.g., OptModelb1).

    Returns:
        dict: {penalty_value: solution_dict}
    """
    all_solutions = {}

    for penalty in penalty_values:
        print(f"\n--------- Running model for load_shifting_penalty = {penalty} --------")

        # Update penalty in JSON
        update_penalty_load_shifting(json_path, penalty)

        # Load and preprocess data
        loader = DataLoader(input_path=data_path)
        raw = loader.get_data()
        processor = DataProcessor1c(raw)
        model_data = processor.build_model_data()

        # Build and solve model
        optm = model_class(model_data)
        solution = optm.solve()
        # Save LP results
        duals = optm.save_LP_duals()

        # Print LP results
        optm.print_LP_results()
        all_solutions[penalty] = solution
        

    return all_solutions
# Define penalty values to test
penalty_values = [ 1.8]

# Paths
json_path = grandparent_dir / "data" / "question_1c"/"usage_preferences.json"
data_path = grandparent_dir / "data" / "question_1c"

# Run sensitivity analysis
solutions = run_penalty_sensitivity(
    penalty_values=penalty_values,
    json_path=json_path,
    data_path=data_path,
    model_class=OptModelc1
)


# %%

# Plot served load vs hours and save
fig3, ax3 = plt.subplots(figsize=(10, 4))
fig3, ax3 = plot_sensitivity_vs_hours(
    solutions,
    column="d",
    y_label="Served Load [kWh]",
    title="Served Load vs Hour for Different Penalties",
    legend_title="Load shifting penalty",
    ax=ax3,
    show=False,
)
fig3.savefig(os.path.join(out_dir, 'battery_served_load_vs_hour_sensitivity.pdf'), format='pdf', bbox_inches='tight')
#plt.show(fig3)

fig4, ax4 = plt.subplots(figsize=(10, 4))
fig4, ax4 = plot_sensitivity_vs_hours(
    solutions,
    column="z",
    y_label="Absolute Load Shift [kWh]",
    title="Load Shift vs Hour for Different Penalties",
    legend_title="Load shifting penalty",
    ax=ax4,
    show=False,
)
fig4.savefig(os.path.join(out_dir, 'battery_absolute_load_shift_vs_hour_sensitivity.pdf'), format='pdf', bbox_inches='tight')
#plt.show(fig4)
# %%
plt.show()
