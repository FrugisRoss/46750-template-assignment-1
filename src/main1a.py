"""
We decided to not use a runner script and instead have a main.py script that can be run in an interactive environment 
for each question separately. This enables easy tweaking and visualization of results.
"""
#%%
from data_ops import DataLoader, DataProcessor, ScenarioModifier
from opt_model import OptModel
from pathlib import Path
from data_ops.data_visualizer import plot_columns_vs_hours
from matplotlib import pyplot as plt
import os

# Get the base directory "grandparent"  of the current file
grandparent_dir = Path(__file__).resolve().parents[1]


# %%
############## Question 1a: Single simulation ##############
# 1. Load and process data
loader = DataLoader(input_path=grandparent_dir / "data" / "question_1a")
raw = loader.get_data()
print(raw)
processor = DataProcessor(raw)
model_data = processor.build_model_data()
print(model_data)

# 2. Build and solve optimization model
optm = OptModel(model_data)
solution = optm.solve()
#%%

# Print LP results
optm.print_LP_results()

#
# %%

out_dir = grandparent_dir / 'results' / 'question_1a'

# %%
columns = ['d', 's_pv', 'x', 'y']
labels = {'s_pv': 'PV production [kWh]',
          'd': 'Electricity consumed [kWh]',
          'x': 'Electricity imported [kWh]',
          'y': 'Electricity exported [kWh]'}


modifier = ScenarioModifier(model_data)
# Create sensitivity plots for different scenarios
scenarios = {
    "base": model_data,
    "multiplied_pricing": modifier.multiply_price(10.0),
    "new_export_tariff": modifier.new_export_tariff(2.0),
    "high_min_consumption": modifier.new_min_consumption(20.0),
}

for name, data in scenarios.items():
    print(f"Running scenario: {name}")
    optm = OptModel(data)
    sol = optm.solve()
    out_dir = grandparent_dir / 'results' / 'question_1a' / name
    os.makedirs(out_dir, exist_ok=True)

    fig = plot_columns_vs_hours(sol,
                            columns=columns,
                            labels=labels,
                            y_label="Electricity [kWh]",
                            title="Electricity usage vs Hour for Scenario: " + name,
                            hour_start=0,
                            figsize=(10, 4),
                            show=False)
    fig.savefig(os.path.join(out_dir, f'combined_energy_vs_hour_{name}.pdf'), format='pdf', bbox_inches='tight')

# %%
for name, data in scenarios.items():
    print(f"Running scenario: {name}")
    optm = OptModel(data)
    sol = optm.solve()
    optm.print_LP_results()
    # Print main important results for each scenario
    print(f"Total generated PV for scenario {name}: {sum(sol['s_pv']):.2f} kWh")
    print(f"Total imported energy for scenario {name}: {sum(sol['x']):.2f} kWh")
    print(f"Total exported energy for scenario {name}: {sum(sol['y']):.2f} kWh")
    print(f"Total self-consumed PV for scenario {name}: {sum(sol['d']):.2f} kWh")
# %%