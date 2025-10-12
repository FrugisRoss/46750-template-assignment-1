#%%
from data_ops import DataLoader, DataProcessor
from data_ops.data_processor import DataProcessor1b
from data_ops.data_processor import update_penalty_load_shifting
from opt_model import OptModel
from opt_model.opt_model import OptModelb1
import numpy as np
from data_ops.data_visualizer import plot_column_vs_hours, plot_sensitivity_vs_hours
import os
import matplotlib.pyplot as plt
from pathlib import Path
# %%

grandparent_dir = Path(__file__).resolve().parents[1]
############## Question 1b ##############

#Set the penalty for load shifting
load_shifting_penalty = 1.8
update_penalty_load_shifting(grandparent_dir / "data" / "question_1b"/"usage_preferences.json", load_shifting_penalty)
# 1. Load and process data
loader = DataLoader(input_path=grandparent_dir / "data" / "question_1b")
raw = loader.get_data()
#print(raw)
processor = DataProcessor1b(raw)
model_data = processor.build_model_data()
#print(model_data)
print(model_data.d_given_t)
print(model_data.p_pen)


#%%
# 2. Build and solve optimization model
optm = OptModelb1(model_data)
solution = optm.solve()

# Save LP results
duals = optm.save_LP_duals()

# Print LP results
optm.print_LP_results()

# %%

out_dir = grandparent_dir / "results" / "question_1b"
os.makedirs(out_dir, exist_ok=True)


fig, ax = plt.subplots(figsize=(10, 4))
plot_column_vs_hours(solution, column='d', y_label="Served Load [kWh]", figsize=(10, 4), hour_start=0, ax=ax, title="Served Load vs Hour", show=False)
fig.savefig(os.path.join(out_dir, 'served_load_vs_hour_base.pdf'), format='pdf', bbox_inches='tight')
#plt.show(fig)

# Absolute load shift plot -> save as vector (SVG and PDF)
fig2, ax2 = plt.subplots(figsize=(10, 4))
plot_column_vs_hours(solution, column='z', y_label="Absolute Load Shift [kWh]", figsize=(10, 4), hour_start=0, ax=ax2, title="Absolute Load Shift vs Hour", show=False)
fig2.savefig(os.path.join(out_dir, 'absolute_load_shift_vs_hour_base.pdf'), format='pdf', bbox_inches='tight')
#plt.show(fig2)


# %%


############## Sensitivity Analysis ##############

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
        processor = DataProcessor1b(raw)
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

penalty_values = [ 0.2, 3.4 ]

json_path = '../46750-template-assignment-1/data/question_1b/usage_preferences.json'
data_path = '../46750-template-assignment-1/data/question_1b'

solutions = run_penalty_sensitivity(
    penalty_values=penalty_values,
    json_path=json_path,
    data_path=data_path,
    model_class=OptModelb1
)

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
fig3.savefig(os.path.join(out_dir, 'served_load_vs_hour_sensitivity.pdf'), format='pdf', bbox_inches='tight')
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
fig4.savefig(os.path.join(out_dir, 'absolute_load_shift_vs_hour_sensitivity.pdf'), format='pdf', bbox_inches='tight')
#plt.show(fig4)
# %%
plt.show()