#%%
from data_ops import DataLoader, DataProcessor
from data_ops.data_processor import DataProcessor1b
from opt_model import OptModel
from opt_model.opt_model import OptModelb1
import numpy as np

# %%
############## Question 1b ##############

# 1. Load and process data
loader = DataLoader(input_path='/Users/rofrug/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/First Year PhD/46750 Optimization in modern power systems/46750-template-assignment-1/data/question_1b')
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
