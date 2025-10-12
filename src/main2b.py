#%%
from data_ops import DataLoader
from data_ops.data_processor import DataProcessor1c, ScenarioModifier, update_penalty_load_shifting
from opt_model.opt_model import OptModel2b
import numpy as np
from data_ops.data_visualizer import plot_columns_vs_hours, plot_sensitivity_vs_hours
from pathlib import Path
from matplotlib import pyplot as plt
from data_ops import DataLoader
import os

# %%
############## Question 1b ##############
# Get the base directory "grandparent"  of the current file
grandparent_dir = Path(__file__).resolve().parents[1]

# %%
############## Question 1a: Single simulation ##############
# 1. Load and process data

#Set the penalty for load shifting
load_shifting_penalty = 1.8 # Penalty cost per kWh of load shifting
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

# Add capital cost to model data
battery_capital_cost_per_kWh = 2500  # Example capital cost per kWh
model_data.capital_cost_per_kWh = battery_capital_cost_per_kWh

# Add N_replications to model data (365 days for 10 years)
model_data.N_replications = 365 * 10


#%%
# 2. Build and solve optimization model
optm = OptModel2b(model_data)
solution = optm.solve()

# Save LP results
duals = optm.save_LP_duals()

# Print LP results
optm.print_LP_results()


# %%
if solution:
    print("Total Cost:", round(solution['objective_value'], 2))
    print("Installed Battery Capacity (kWh):", round(solution['installed_capacity_kWh'], 2))
    print("Scaling Factor for Battery Capacity:", round(solution['optimal_scale'], 4))
else:
    print("No feasible solution found.")

# %%
# Assuming your solution dict from solve() is called 'solution'
hourly_data = solution.get("hourly_dispatch", {})


columns = ['d', 'z', 'b_d', 'b_c', 'b_soc']
labels = { 'd': 'Served Load', 'z': 'Load Shift', 'b_d': 'Battery Discharging Power', 'b_c': 'Battery Charging Power', 'b_soc': 'Battery SOE' }
 
fig = plot_columns_vs_hours(hourly_data,
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
 
fig2 = plot_columns_vs_hours(hourly_data,
                            columns=columns,
                            labels=labels,
                            y_label="Electricity [kWh]",
                            title="Do your thing: " + str("Electricity Balance"),
                            hour_start=0,
                            figsize=(10, 4),
                            show=False)


out_dir = grandparent_dir / "results" / "question_2b"
os.makedirs(out_dir, exist_ok=True)
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
        model_data.capital_cost_per_kWh = battery_capital_cost_per_kWh
        # Add N_replications to model data (365 days for 10 years)
        model_data.N_replications = 365 * 10


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
penalty_values = [ 0.2, 1.0, 1.8,2.6, 3.4 ]

# Paths
json_path = grandparent_dir / "data" / "question_1c"/"usage_preferences.json"
data_path = grandparent_dir / "data" / "question_1c"

# Run sensitivity analysis
solutions = run_penalty_sensitivity(
    penalty_values=penalty_values,
    json_path=json_path,
    data_path=data_path,
    model_class=OptModel2b
)

# Define empy dict to store hourly data for each solution
hourly_data_dict = {}
# Extract hourly data for each solution
for sol in solutions:
    hourly_data_dict[sol] = solutions[sol].get("hourly_dispatch", {})




# %%

# Plot served load vs hours and save
fig3, ax3 = plt.subplots(figsize=(10, 4))
fig3, ax3 = plot_sensitivity_vs_hours(
    hourly_data_dict,
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
    hourly_data_dict,
    column="z",
    y_label="Absolute Load Shift [kWh]",
    title="Load Shift vs Hour for Different Penalties",
    legend_title="Load shifting penalty",
    ax=ax4,
    show=False,
)
fig4.savefig(os.path.join(out_dir, 'battery_absolute_load_shift_vs_hour_sensitivity.pdf'), format='pdf', bbox_inches='tight')
# %%
# Plot battery charging and discharging power vs hours and save
fig5, ax5 = plt.subplots(figsize=(10, 4))
fig5, ax5 = plot_sensitivity_vs_hours(
    hourly_data_dict,
    column="b_c",
    y_label="Battery Charging Power [kWh]",
    title="Battery Charging Power vs Hour for Different Penalties",
    legend_title="Load shifting penalty",
    ax=ax5,
    show=False,
)
fig5.savefig(os.path.join(out_dir, 'battery_charging_power_vs_hour_sensitivity.pdf'), format='pdf', bbox_inches='tight')
#%%
fig6, ax6 = plt.subplots(figsize=(10, 4))
fig6, ax6 = plot_sensitivity_vs_hours(
    hourly_data_dict,
    column="b_d",
    y_label="Battery Discharging Power [kWh]",
    title="Battery Discharging Power vs Hour for Different Penalties",
    legend_title="Load shifting penalty",
    ax=ax6,
    show=False,
)
fig6.savefig(os.path.join(out_dir, 'battery_discharging_power_vs_hour_sensitivity.pdf'), format='pdf', bbox_inches='tight')
#%% 

columns = ['d', 's_pv', 'x', 'y']
labels = {'s_pv': 'PV production [kWh]',
          'd': 'Electricity consumed [kWh]',
          'x': 'Electricity imported [kWh]',
          'y': 'Electricity exported [kWh]'}

model_data

modifier = ScenarioModifier(model_data)
# Create sensitivity plots for different scenarios
scenarios = {
    "base": model_data,
    "multiplied_pricing": modifier.multiply_price(10.0),
    "new_export_tariff": modifier.new_export_tariff(2.0),
}

for name, data in scenarios.items():
    print(f"Running scenario: {name}")
    optm = OptModel2b(data)
    sol = optm.solve()
    hourly_data = sol.get("hourly_dispatch", {})
    out_dir = grandparent_dir / 'results' / 'question_2b' / name
    os.makedirs(out_dir, exist_ok=True)

    fig = plot_columns_vs_hours(hourly_data,
                            columns=columns,
                            labels=labels,
                            y_label="Electricity [kWh]",
                            title="Electricity usage vs Hour for Scenario: " + name,
                            hour_start=0,
                            figsize=(10, 4),
                            show=False)
    fig.savefig(os.path.join(out_dir, f'combined_energy_vs_hour_{name}_2b.pdf'), format='pdf', bbox_inches='tight')

# %%
columns = ['d', 'b_c', 'b_d', 'b_soc']
labels = { 'd': 'Served Load', 'b_c': 'Charging', 'b_d': 'Discharging', 'b_soc': 'Battery SOE', }
for name, data in scenarios.items():
    print(f"Running scenario: {name}")
    optm = OptModel2b(data)
    sol = optm.solve()
    hourly_data = sol.get("hourly_dispatch", {})
    out_dir = grandparent_dir / 'results' / 'question_2b' / name
    os.makedirs(out_dir, exist_ok=True)
    fig = plot_columns_vs_hours(hourly_data,
                                columns=columns,
                                labels=labels,
                                y_label="Electricity [kWh]",
                                title=str("Battery Operation vs Hour for Scenario: " + name),
                                hour_start=0,
                                figsize=(10, 4),
                                show=False)
    fig.savefig(os.path.join(out_dir, f'battery_operation_vs_hour_{name}_2b.pdf'), format='pdf', bbox_inches='tight')

# %%
for name, data in scenarios.items():
    print(f"Running scenario: {name}")
    optm = OptModel2b(data)
    sol = optm.solve()
    hourly_data = sol.get("hourly_dispatch", {})
    optm.print_LP_results()
    # Print main important results for each scenario
    print(f"Total generated PV for scenario {name}: {sum(hourly_data['s_pv']):.2f} kWh")
    print(f"Total imported energy for scenario {name}: {sum(hourly_data['x']):.2f} kWh")
    print(f"Total exported energy for scenario {name}: {sum(hourly_data['y']):.2f} kWh")
    print(f"Total self-consumed PV for scenario {name}: {sum(hourly_data['d']):.2f} kWh")
    print("Total Cost:", round(sol['objective_value'], 2))
    print("Installed Battery Capacity (kWh):", round(sol['installed_capacity_kWh'], 2))
    print("Scaling Factor for Battery Capacity:", round(sol['optimal_scale'], 4))


# %%
