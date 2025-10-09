"""
data_loader.py

Simple data loader for the flexible consumer scheduling problem.

This module reads JSON configuration files and returns structured Python objects
(dictionaries and dataclasses) that downstream modules can use.

Expected files in a data folder (example: question_1a/):
- appliance_params.json
- bus_params.json
- consumer_params.json
- DER_production.json
- usage_preference.json

The DataLoader class provides methods to load and access this data.

"""
#%%
# -----------------------------
# Load Data
# -----------------------------
import json
import csv
import pandas as pd
from pathlib import Path

from pathlib import Path
from dataclasses import dataclass
from logging import Logger
import pandas as pd
from dataclasses import dataclass
import os
from typing import Dict, Any

class DataLoader:
    """
    Loads energy system input data for a given configuration/question from structured CSV and json files
    and an auxiliary configuration metadata file.

    >>> data = DataLoader(input_path='data/question_1a')
    
    """
    question: str
    input_path: Path

    def __init__(self, input_path):
        """
        Initializes the DataLoader with the base directory containing the JSON input files.
        :param base_path: Folder path where question 1a JSON data files are located.
        """

        self.input_path = Path(input_path).resolve()
        self.appliance_params: Dict[str, Any] = {}
        self.bus_params: Dict[str, Any] = {}
        self.consumer_params: Dict[str, Any] = {}
        self.der_production: Dict[str, Any] = {}
        self.usage_preference: Dict[str, Any] = {}

        # load immediately
        self.load_all()
        self._populate_data()

    def _load_json(self, filename: str) -> Any:
        """Helper method to load a JSON file from the base path."""
        path = os.path.join(self.input_path, filename)
        with open(path, 'r') as f:
            return json.load(f)


    def load_all(self):
        """Loads all required JSON files into respective attributes."""
        self.appliance_params = self._load_json('appliance_params.json')
        self.bus_params = self._load_json('bus_params.json')
        self.consumer_params = self._load_json('consumer_params.json')
        self.der_production = self._load_json('DER_production.json')
        self.usage_preference = self._load_json('usage_preferences.json')


    def _populate_data(self) -> Dict[str, Any]:
        """
        Returns a dictionary of all loaded data.
        :return: Dictionary with keys 'der', 'bus', 'consumer', 'der_production', 'usage_preference'.
        """
        self.data = {
            "appliance": self.appliance_params,
            "bus": self.bus_params,
            "consumer": self.consumer_params,
            "der_production": self.der_production,
            "usage_preference": self.usage_preference,
        }
        
    def get_data(self) -> Dict[str, Any]:
        """Return the combined dictionary (already populated during __init__)."""
        return self.data

    
#%%
# if __name__ == '__main__':
#     data = DataLoader(input_path='/Users/lalka/Projects/46750-template-assignment-1/data/question_1a')
    # dl.load_all()
    # data = dl.get_data()
    # print(data)  
# %%