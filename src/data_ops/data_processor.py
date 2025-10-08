from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelData:
    """
    Canonical, model-ready inputs for the primal LP.

    Time is hourly with T = len(hours). Units assumed:
      - Energy kWh per hour step, power kW where relevant.
      - Prices in DKK/kWh, tariffs in DKK/kWh.

    s_t: PV available energy (kWh) each hour that can be curtailed.
    p_buy_t: grid purchase price (DKK/kWh).
    p_sell_t: grid sale price (DKK/kWh).
    tau_t: grid tariff fee per kWh exchanged (DKK/kWh), applied to imports and exports.
    d_min_total: minimum total daily consumption (kWh).
    d_max_t: per-hour maximum flexible consumption bound (kWh).
    x_max_t: per-hour max import (kWh).
    y_max_t: per-hour max export (kWh).
    hours: index [0..T-1] or a pandas Index of 24 hours if provided upstream.
    notes: free-form notes about assumptions made.
    """
    # Core energy flows
    s_t: np.ndarray                    # PV available energy (kWh)
    p_buy_t: np.ndarray               # grid purchase price (DKK/kWh)
    p_sell_t: np.ndarray              # grid sale price (DKK/kWh)
    tau_import_t: np.ndarray          # grid import tariff (DKK/kWh)
    tau_export_t: np.ndarray          # grid export tariff (DKK/kWh)
    
    # Penalties for excess import/export beyond limits
    penalty_excess_import: float       # DKK/kWh for importing above x_max_t
    penalty_excess_export: float       # DKK/kWh for exporting above y_max_t
    
    # Load requirements and bounds
    d_min_total: float                # minimum total daily consumption (kWh)
    d_max_t: np.ndarray              # per-hour maximum flexible consumption (kWh)
    d_min_ratio: float               # minimum power ratio for flexible load
    
    # Grid limits (regular operation, penalties apply beyond these)
    x_max_t: np.ndarray              # per-hour max import without penalty (kWh)
    y_max_t: np.ndarray              # per-hour max export without penalty (kWh)
    
    # Ramp rate constraints for flexible load
    ramp_up_max_ratio: float         # max ramp up per hour as ratio of max_load
    ramp_down_max_ratio: float       # max ramp down per hour as ratio of max_load
    
    # On/off time constraints (for loads that have minimum runtime)
    min_on_time_h: int               # minimum consecutive hours load must stay on
    min_off_time_h: int              # minimum consecutive hours load must stay off
    
    # Time indexing
    hours: pd.Index
    
    # Storage and heat pump (for future extension)
    storage_params: Optional[Dict[str, Any]]    # battery parameters if available
    heat_pump_params: Optional[Dict[str, Any]]  # heat pump parameters if available
    
    # Metadata
    notes: Dict[str, Any]

@dataclass
class ModelData1b(ModelData):
    
    p_pen: np.ndarray
    d_given_t: np.ndarray


class DataProcessor:
    """
    Transforms raw nested dictionaries from DataLoader into model-ready arrays and scalars.

    Assumptions:
      - Single consumer, single bus.
      - One PV DER with nameplate max_power_kW and hourly_profile_ratio in [0,1].
      - One fully flexible load with max_load_kWh_per_hour (treated as hourly max).
      - Grid tariffs apply symmetrically to imports and exports.
      - energy_price_DKK_per_kWh is the buy price; sale price can be provided
        or derived (e.g., equal to price or via a discount).
      - Tariffs are provided separately as import/export tariff

    Raises ValueError with informative messages if inputs are inconsistent.
    """

    def __init__(self, raw: Dict[str, Any]) -> None:
        self.raw = raw
        self._validate_top_keys()

    def _validate_top_keys(self) -> None:
        required = ["appliance", "bus", "consumer", "der_production", "usage_preference"]
        missing = [k for k in required if k not in self.raw]
        if missing:
            raise ValueError(f"Missing top-level keys in data: {missing}")

    def _extract_single_bus(self) -> Dict[str, Any]:
        bus_list = self.raw.get("bus", [])
        if not isinstance(bus_list, list) or len(bus_list) != 1:
            raise ValueError("Expected exactly one bus entry.")
        return bus_list[0]

    def _extract_single_consumer(self) -> Dict[str, Any]:
        cons_list = self.raw.get("consumer", [])
        if not isinstance(cons_list, list) or len(cons_list) != 1:
            raise ValueError("Expected exactly one consumer entry.")
        return cons_list[0]

    def _extract_usage_pref(self) -> Dict[str, Any]:
        prefs = self.raw.get("usage_preference", [])
        if not isinstance(prefs, list) or len(prefs) != 1:
            raise ValueError("Expected exactly one usage_preference entry.")
        return prefs[0]

    def _extract_pv_specs(self) -> Tuple[float, str]:
        """Return (pv_nameplate_kW, pv_id)."""
        app = self.raw.get("appliance", {})
        ders = app.get("DER", []) or []
        if not ders:
            raise ValueError("No DER entries found; expected one PV.")
        pv = None
        for d in ders:
            if d.get("DER_type", "").upper() == "PV":
                pv = d
                break
        if pv is None:
            raise ValueError("No PV DER found in appliance.DER.")
        nameplate = float(pv.get("max_power_kW", 0.0))
        if nameplate <= 0:
            raise ValueError("PV nameplate max_power_kW must be > 0.")
        return nameplate, pv.get("DER_id", "PV")

    def _extract_flexible_load_specs(self) -> Tuple[float, str, Dict[str, Any]]:
        """Return (max_load_kWh_per_hour, load_id, load_constraints)."""
        app = self.raw.get("appliance", {})
        loads = app.get("load", []) or []
        if not loads:
            raise ValueError("No load entries found; expected one fully flexible load.")
        ffl = None
        for L in loads:
            lt = L.get("load_type", "").lower()
            if "fully flexible" in lt:
                ffl = L
                break
        if ffl is None:
            raise ValueError("No 'fully flexible load' found in appliance.load.")
        
        max_per_h = float(ffl.get("max_load_kWh_per_hour", 0.0))
        if max_per_h <= 0:
            raise ValueError("max_load_kWh_per_hour must be > 0 for flexible load.")
        
        # Extract additional load constraints
        constraints = {
            "min_power_ratio": float(ffl.get("min_load_ratio", 0.0)),
            "ramp_up_ratio": float(ffl.get("max_ramp_rate_up_ratio", 1.0)),
            "ramp_down_ratio": float(ffl.get("max_ramp_rate_down_ratio", 1.0)),
            "min_on_time_h": int(ffl.get("min_on_time_h", 0)),
            "min_off_time_h": int(ffl.get("min_off_time_h", 0)),
        }
        
        return max_per_h, ffl.get("load_id", "FFL"), constraints

    def _extract_pv_profile(self) -> List[float]:
        prod = self.raw.get("der_production", [])
        if not isinstance(prod, list) or len(prod) < 1:
            raise ValueError("Missing der_production list.")
        # Pick first solar profile for the single consumer
        pv_entry = None
        for e in prod:
            if str(e.get("DER_type", "")).lower() in ("solar", "pv"):
                pv_entry = e
                break
        if pv_entry is None:
            raise ValueError("No solar/PV der_production entry found.")
        profile = pv_entry.get("hourly_profile_ratio", None)
        if profile is None or len(profile) == 0:
            raise ValueError("hourly_profile_ratio missing/empty in der_production.")
        return [float(v) for v in profile]

    def _extract_min_daily_energy(self, load_id: str) -> float:
        prefs = self._extract_usage_pref()
        lp = prefs.get("load_preferences", []) or []
        entry = None
        for p in lp:
            if p.get("load_id") == load_id:
                entry = p
                break
        if entry is None:
            raise ValueError(f"No usage preference found for load_id={load_id}.")
        v = entry.get("min_total_energy_per_day_hour_equivalent")
        if v is None:
            raise ValueError("min_total_energy_per_day_hour_equivalent must be provided.")
        return float(v)

    def _extract_prices_and_tariffs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
        """Returns (p_buy_t, p_sell_t, tau_import_t, tau_export_t, hours_index)"""
        bus = self._extract_single_bus()

        prices = bus.get("energy_price_DKK_per_kWh")
        p_buy = np.asarray([float(x) for x in prices], dtype=float)
        p_sell = p_buy.copy()  # or adapt for real case

        tau_import = float(bus.get("import_tariff_DKK/kWh", 0.0))
        tau_export = float(bus.get("export_tariff_DKK/kWh", 0.0))

        tau_import_t = np.full_like(p_buy, tau_import, dtype=float)
        tau_export_t = np.full_like(p_buy, tau_export, dtype=float)
        hours = pd.RangeIndex(start=0, stop=len(p_buy), step=1, name="hour")
        return p_buy, p_sell, tau_import_t, tau_export_t, hours

    def _extract_penalty_costs(self) -> Tuple[float, float]:
        """Extract penalty costs for excess import/export."""
        bus = self._extract_single_bus()
        penalty_import = float(bus.get("penalty_excess_import_DKK/kWh", 0.0))
        penalty_export = float(bus.get("penalty_excess_export_DKK/kWh", 0.0))
        return penalty_import, penalty_export

    def _bounds_from_bus(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        bus = self._extract_single_bus()
        x_max = float(bus.get("max_import_kW", 0.0))
        y_max = float(bus.get("max_export_kW", 0.0))
        if x_max <= 0 or y_max < 0:
            raise ValueError("max_import_kW must be >0 and max_export_kW >=0.")
        # 1-hour resolution: kW -> kWh per step = same numeric value.
        return np.full(T, x_max, dtype=float), np.full(T, y_max, dtype=float)

    def _extract_storage_heat_pump(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Extract storage and heat pump parameters if available."""
        app = self.raw.get("appliance", {})
        storage = app.get("storage")
        heat_pump = app.get("heat_pump")
        return storage, heat_pump

    def build_model_data(self) -> ModelData:
        """Main entry point: returns enhanced ModelData with penalties and constraints."""
        # time and prices
        p_buy_t, p_sell_t, tau_import_t, tau_export_t, hours = self._extract_prices_and_tariffs()
        T = len(hours)

        # PV availability s_t
        pv_nameplate_kW, pv_id = self._extract_pv_specs()
        profile = self._extract_pv_profile()
        if len(profile) != T:
            raise ValueError(f"PV profile length {len(profile)} != T={T}.")
        s_t = pv_nameplate_kW * np.asarray(profile, dtype=float)

        # Flexible load bounds and constraints
        d_max_per_h, load_id, load_constraints = self._extract_flexible_load_specs()
        d_max_t = np.full(T, d_max_per_h, dtype=float)

        # Grid bounds and penalty costs
        x_max_t, y_max_t = self._bounds_from_bus(T)
        penalty_import, penalty_export = self._extract_penalty_costs()

        # Daily minimum energy
        d_min_total = self._extract_min_daily_energy(load_id)

        # Storage and heat pump (for future use)
        storage_params, heat_pump_params = self._extract_storage_heat_pump()

        # Validation
        if np.any(p_buy_t < 0) or np.any(p_sell_t < 0) or np.any(tau_import_t < 0) or np.any(tau_export_t < 0):
            raise ValueError("Negative prices/tariffs are not supported.")
        if d_min_total < 0:
            raise ValueError("Minimum daily energy must be >= 0.")

        notes = {
            "pv_id": pv_id,
            "load_id": load_id,
            "assumptions": {
                "symmetric_tariff": False,
                "sell_price_equals_buy_price": True,
                "hour_resolution_h": 1.0,
                "penalty_excess_enabled": True,
            },
        }

        return ModelData(
            s_t=s_t,
            p_buy_t=p_buy_t,
            p_sell_t=p_sell_t,
            tau_import_t=tau_import_t,
            tau_export_t=tau_export_t,
            penalty_excess_import=penalty_import,
            penalty_excess_export=penalty_export,
            d_min_total=float(d_min_total),
            d_max_t=d_max_t,
            d_min_ratio=load_constraints["min_power_ratio"],
            x_max_t=x_max_t,
            y_max_t=y_max_t,
            ramp_up_max_ratio=load_constraints["ramp_up_ratio"],
            ramp_down_max_ratio=load_constraints["ramp_down_ratio"],
            min_on_time_h=load_constraints["min_on_time_h"],
            min_off_time_h=load_constraints["min_off_time_h"],
            hours=hours,
            storage_params=storage_params,
            heat_pump_params=heat_pump_params,
            notes=notes,
        )

class DataProcessor1b(DataProcessor):
    """
    Extension of DataProcessor for 1b, extracting d_given_t and p_pen for ModelData1b.
    """


    def _extract_given_load_profile(self, load_id: str, T: int, d_max_per_h: float) -> np.ndarray:
        """
        Extract the given hourly load profile (d_given_t) for the specified load_id.
        Returns a numpy array of length T (typically 24) with absolute hourly values
        (kWh) obtained by scaling the hourly_profile_ratio by d_max_per_h.
        If not found or malformed, raises ValueError.
        """
        prefs = self._extract_usage_pref()
        lp = prefs.get("load_preferences", []) or []
        entry = None
        for p in lp:
            if p.get("load_id") == load_id:
                entry = p
                break
        if entry is None:
            raise ValueError(f"No usage preference found for load_id={load_id}.")

        # Extract ratio profile and validate
        d_given_ratio = entry.get("hourly_profile_ratio", None)
        if d_given_ratio is None:
            raise ValueError(f"hourly_profile_ratio missing for load_id={load_id}.")
        if len(d_given_ratio) != T:
            raise ValueError(f"hourly_profile_ratio length {len(d_given_ratio)} != T={T} for load_id={load_id}.")

        # Convert to absolute hourly values (kWh) by scaling with d_max_per_h
        d_given_t = np.asarray([float(v) for v in d_given_ratio], dtype=float) #* float(d_max_per_h)
        return d_given_t


    def _extract_penalty_profile(self, load_id: str) -> float:

        prefs = self._extract_usage_pref()
        lp = prefs.get("load_preferences", []) or []
        entry = None
        for p in lp:
            if p.get("load_id") == load_id:
                entry = p
                break
        if entry is not None and "penalty_load_shifting" in entry and entry["penalty_load_shifting"] is not None:
            p_pen = float(entry["penalty_load_shifting"])
            return float(p_pen)
        else:
            raise ValueError(f"Penalty profile not found for load_id={load_id}.")
        
       

    def build_model_data(self) -> ModelData1b:
        """
        Returns ModelData1b with d_given_t and p_pen extracted.
        """
        # time and prices
        p_buy_t, p_sell_t, tau_import_t, tau_export_t, hours = self._extract_prices_and_tariffs()
        T = len(hours)


        # PV availability s_t
        pv_nameplate_kW, pv_id = self._extract_pv_specs()
        profile = self._extract_pv_profile()
        if len(profile) != T:
            raise ValueError(f"PV profile length {len(profile)} != T={T}.")
        s_t = pv_nameplate_kW * np.asarray(profile, dtype=float)

        # Flexible load bounds and constraints
        d_max_per_h, load_id, load_constraints = self._extract_flexible_load_specs()
        d_max_t = np.full(T, d_max_per_h, dtype=float)

        # Grid bounds and penalty costs
        x_max_t, y_max_t = self._bounds_from_bus(T)
        penalty_import, penalty_export = self._extract_penalty_costs()

        # Storage and heat pump (for future use)
        storage_params, heat_pump_params = self._extract_storage_heat_pump()

        # d_given_t and p_pen
        d_given_t = self._extract_given_load_profile(load_id, T, d_max_per_h)
        p_pen = self._extract_penalty_profile(load_id)

        # Daily minimum energy
        d_min_total = self._extract_min_daily_energy(load_id)

        # Validation
        if np.any(p_buy_t < 0) or np.any(p_sell_t < 0) or np.any(tau_import_t < 0) or np.any(tau_export_t < 0):
            raise ValueError("Negative prices/tariffs are not supported.")
        if d_min_total < 0:
            raise ValueError("Minimum daily energy must be >= 0.")

        notes = {
            "pv_id": pv_id,
            "load_id": load_id,
            "assumptions": {
                "symmetric_tariff": False,
                "sell_price_equals_buy_price": True,
                "hour_resolution_h": 1.0,
                "penalty_excess_enabled": True,
            },
        }

        return ModelData1b(
            s_t=s_t,
            p_buy_t=p_buy_t,
            p_sell_t=p_sell_t,
            tau_import_t=tau_import_t,
            tau_export_t=tau_export_t,
            penalty_excess_import=penalty_import,
            penalty_excess_export=penalty_export,
            d_min_total=float(d_min_total),
            d_max_t=d_max_t,
            d_min_ratio=load_constraints["min_power_ratio"],
            x_max_t=x_max_t,
            y_max_t=y_max_t,
            ramp_up_max_ratio=load_constraints["ramp_up_ratio"],
            ramp_down_max_ratio=load_constraints["ramp_down_ratio"],
            min_on_time_h=load_constraints["min_on_time_h"],
            min_off_time_h=load_constraints["min_off_time_h"],
            hours=hours,
            storage_params=storage_params,
            heat_pump_params=heat_pump_params,
            notes=notes,
            p_pen=p_pen,
            d_given_t=d_given_t,
        )
