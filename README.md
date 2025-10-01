# Assignment 1, group 28

## Overview

This repository serves as a template for **Group Assignment 1** in the course **46750 - Optimization in Modern Power Systems**. It provides a structured starting point for your project, including:

- Example datasets for all assignment questions
- Starter Python code to help you begin your analysis
- Licensing information
- Dependency files (`requirements.txt` and `environment.yaml`)
- A `.gitignore` file
- This `README.md` with setup and usage instructions

**Note:** This structure is not definitive, and can be adapted to meet each groups' needs as the project advances.

## Installation

Follow the installation instructions below to start working on your group assignment.

### 1. **Clone the repository**

To begin, create a copy of this repository for your group. 

### 2. **Create a virtual environment**

Follow these steps to set up a clean Python environment and install the required packages, so that all required packages are installed when running your code, and your project packages won’t affect the system or other projects.

#### **Option A: Using pip**

To create an isolated Python environment in a folder called venv (its own Python interpreter + its own site-packages and pip):
```bash
python -m venv venv 
```
Then activate that environment (on macOS/Linux):
```bash
source venv/bin/activate 
```
Or, using Windows cmd: `\venv\Scripts\Activate.ps1` or Windows PowerShell: `venv\Scripts\Activate.ps1`. Your shell’s PATH is changed so python/pip now point to the ones inside venv.

Install all packages listed in requirements.txt into the active virtual environment (modify the requirements file as needed).
```bash
pip install -r requirements.txt
```

#### **Option B: Using conda**

Create and activate a virtual environment, using the file environment.yaml (modify the environment file as needed):
```bash
conda env create -f environment.yaml
conda activate gurobi-opt
``` 

### Getting Started

1. **Install dependencies** as described above.
2. **Explore the starter code** in `main.py` and the `src/` folder to understand the workflow.
3. **Add your code**:
    - Implement new functions or classes in the appropriate `src/` subfolder.
    - Update `main.py` to call your new code for data processing, model setup, or result analysis.
4. **Run simulations** by executing:
    ```bash
    python main.py
    ```
    or, if using a Jupyter notebook, run the provided cells.

5. **Visualize results**: Output files, plots, or logs will be generated as specified in your code. Adjust the code to save results in your preferred format.

**Note:** As you extend the codebase, document any new scripts or modules, and changes in structure, in this README for clarity and reproducibility.

### Starter Code Structure

 The starter code is organized as follows:

- `main.py`: Entry point for running simulations and analyses. This script parses arguments, loads data, initializes models, and coordinates the workflow.
- `src/`: Contains all source code modules.
    - `src/data_ops/`: Classes and functions for loading, validating, and preprocessing input datasets (e.g., reading JSON files, checking data integrity, and preparing data structures for modeling).
    - `src/opt_model/`: Modular optimization models and algorithms for each assignment question. Each submodule can represent a different modeling approach or scenario, making it easy to extend or modify optimization logic.
    - `src/runner/`: Scripts or classes that orchestrate the end-to-end execution of simulations, including setting up experiments, running optimization routines, and collecting results.
    - `src/utils/`: Utility functions and helpers, such as plotting routines, configuration file parsers, logging setup, and other reusable code snippets.

## Input Data Structure

The repositories include base datasets under the `data/question_name` directories, organized as follows:

- **Consumers Data (`consumers.json`)**  
    Contains a list of consumers, each with:
    - `consumer_id`: Unique identifier for the consumer
    - `connection_bus`: Bus ID where the consumer is connected
    - `list_appliances`: List of appliance IDs owned by the consumer

- **Appliances Data (`appliance_params.json`)**  
    Contains a list of all appliances and their technical characteristics. Each appliance entry includes:

    - **For DERs (Distributed Energy Resources):**
        - `DER_id`: Unique identifier for the DER appliance
        - `DER_type`: DER technology type (e.g., "PV" for solar photovoltaic, "wind" for wind turbine)
        - `max_power_kW`: Maximum power output (kW)
        - `min_power_ratio`: Minimum operating power as a fraction of max power (unitless, 0–1)
        - `max_ramp_rate_up_ratio`: Maximum allowed increase in power per time step, as a fraction of max power (unitless, 0–1)
        - `max_ramp_rate_down_ratio`: Maximum allowed decrease in power per time step, as a fraction of max power (unitless, 0–1)

    - **For Loads:**
        - `load_id`: Unique identifier for the load appliance
        - `load_type`: Type of load (e.g., "EV", "heater")
        - `max_load_kWh_per_hour`: Maximum energy consumption per hour (kWh/h)
        - `max_ramp_rate_up_ratio`: Maximum allowed increase in load per time step, as a fraction of max load (unitless, 0–1)
        - `max_ramp_rate_down_ratio`: Maximum allowed decrease in load per time step, as a fraction of max load (unitless, 0–1)
        - `min_on_time_h`: Minimum consecutive hours the load must stay ON (h)
        - `min_off_time_h`: Minimum consecutive hours the load must stay OFF (h)

    - **For Storages:**
        - `storage_id`: Unique identifier for the storage appliance
        - `storage_capacity_kWh`: Total energy storage capacity (kWh)
        - `max_charging_power_ratio`: Maximum charging power as a fraction of storage capacity per hour (unitless, 0–1)
        - `max_discharging_power_ratio`: Maximum discharging power as a fraction of storage capacity per hour (unitless, 0–1)
        - `charging_efficiency`: Fraction of energy retained during charging (unitless, 0–1)
        - `discharging_efficiency`: Fraction of energy retained during discharging (unitless, 0–1)

**Note:** All ratios are relative to the respective appliance's maximum capacity or power. Units are indicated in parentheses.
    

- **Usage Preferences (`usage_preference.json`)**  
    Specifies user-defined preferences and constraints for energy usage and appliance operation. Example structure:
    - `consumer_id`: Unique identifier for the consumer
    - `_preferences`: containing
        - **Grid preferences**: Preferences for grid interaction (e.g., "prefer self-consumption", "allow export up to X kWh")
        - **DER preferences**: Preferences for usage of distributed energy resources (e.g., "curtailment cost", "limit wind export", "green consumption ratio")
        - **Load preferences**: Preferences for consumption (daily/hourly) and flexibility:
            - `load_id`: Unique load identifier
            - `min_total_energy_per_day_hour_equivalent`: Minimum daily energy usage (kWh or equivalent hours)
            - `max_total_energy_per_day_hour_equivalent`: Maximum daily energy usage (kWh or equivalent hours)
            - `hourly_profile_ratio`: Desired hourly usage pattern (array of ratios)
        - **Storages**: Preferences for usage of energy storage:
            - `storage_id`: Unique storage identifier
            - `initial_soc_ratio`: Initial state of charge (0–1)
            - `final_soc_ratio`: Desired final state of charge (0–1)
        - **Heat pumps**: Preferences for heat pump operation (e.g., "min runtime", "preferred hours")

- **DER Production (`DER_production.json`)**  
    Contains time series data for DER output profiles at each consumer location.
    - `consumer_id`: Location where the DER is evaluated
    - `DER_type`: Type of DER (e.g., "PV", "wind")
    - `hourly_profile_ratio`: Array of normalized hourly production values (0–1)

- **Bus Data (`bus_params.json`)**  
    Defines technical and economic parameters for each network bus.
    - `bus_id`: Unique bus identifier
    - `import_tariff`: Tariff for net energy import (DKK/kWh)
    - `export_tariff`: Tariff for net energy export (DKK/kWh)
    - `max_import_kw`: Maximum allowed import power (kW)
    - `max_export_kw`: Maximum allowed export power (kW)
    - `price_DKK_per_kWh`: Additional price information if applicable

**Note:**  
These files allow customization of user behavior, DER production, and network constraints for simulation and optimization. Students can extend or replace these datasets as needed to conduct adequate simulations and sensitivity analysis. We recommend that any new or modified files follow the same structure for compatibility with the starter code, and easy grading. Please document all new datasets in this README.md file.

## Starter Code Structure

