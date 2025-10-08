"""
Placeholder for main function to execute the model runner. This function creates a single/multiple instance of the Runner class, prepares input data,
and runs a single/multiple simulation.

Suggested structure:
- Import necessary modules and functions.
- Define a main function to encapsulate the workflow (e.g. Create an instance of your the Runner class, Run a single simulation or multiple simulations, Save results and generate plots if necessary.)
- Prepare input data for a single simulation or multiple simulations.
- Execute main function when the script is run directly.
"""
#%%
from data_ops import DataLoader, DataProcessor
from opt_model import OptModel


# %%
############## Question 1a: Single simulation ##############
# 1. Load and process data
loader = DataLoader(input_path='/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/46750 Optimization in modern power systems/46750-template-assignment-1/data/question_1a')
raw = loader.get_data()
print(raw)
processor = DataProcessor(raw)
model_data = processor.build_model_data()
print(model_data)

# 2. Build and solve optimization model
optm = OptModel(model_data)
solution = optm.solve()

# %%
if solution:
    print(f"Optimal cost: {solution['obj']}")
    print(f"Scheduled load each hour: {solution['d']}")
    print(f"PV usage each hour: {solution['s_pv']}")
    print(f"Grid imports: {solution['x']}")
    print(f"Grid exports: {solution['y']}")
else:
    print("No feasible solution found.")
#