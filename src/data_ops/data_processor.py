from __future__ import annotations

from dataclasses import dataclass
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
    s_t: np.ndarray
    p_buy_t: np.ndarray
    p_sell_t: np.ndarray
    tau_import_t: np.ndarray
    tau_export_t: np.ndarray
    d_min_total: float
    d_max_t: np.ndarray
    x_max_t: np.ndarray
    y_max_t: np.ndarray
    hours: pd.Index
    notes: Dict[str, Any]


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

    def _extract_flexible_load_specs(self) -> Tuple[float, str]:
        """Return (max_load_kWh_per_hour, load_id)."""
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
        return max_per_h, ffl.get("load_id", "FFL")

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
        """
        Returns (p_buy_t, p_sell_t, tau_import_t, tau_export_t, hours_index)

        - p_buy_t from bus['energy_price_DKK_per_kWh'] (buy).
        - p_sell_t: if not provided, can set equal or with discount.
        - tau_import_t and tau_export_t: distinct tariff arrays.
        """
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

    def _bounds_from_bus(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        bus = self._extract_single_bus()
        x_max = float(bus.get("max_import_kW", 0.0))
        y_max = float(bus.get("max_export_kW", 0.0))
        if x_max <= 0 or y_max < 0:
            raise ValueError("max_import_kW must be >0 and max_export_kW >=0.")
        # 1-hour resolution: kW -> kWh per step = same numeric value.
        return np.full(T, x_max, dtype=float), np.full(T, y_max, dtype=float)

    def build_model_data(self) -> ModelData:
        """
        Main entry point: returns ModelData with arrays/scalars for the LP.
        """
        # time and prices
        p_buy_t, p_sell_t, tau_import_t, tau_export_t, hours = self._extract_prices_and_tariffs()
        T = len(hours)

        # PV availability s_t
        pv_nameplate_kW, pv_id = self._extract_pv_specs()
        profile = self._extract_pv_profile()
        if len(profile) != T:
            raise ValueError(f"PV profile length {len(profile)} != T={T}.")
        # Convert to kWh per hour step: nameplate (kW) * ratio (0..1) * 1h
        s_t = pv_nameplate_kW * np.asarray(profile, dtype=float)

        # Flexible load bounds
        d_max_per_h, load_id = self._extract_flexible_load_specs()
        d_max_t = np.full(T, d_max_per_h, dtype=float)

        # Grid bounds
        x_max_t, y_max_t = self._bounds_from_bus(T)

        # Daily minimum energy
        d_min_total = self._extract_min_daily_energy(load_id)

        # Optional simple checks
        if np.any(p_buy_t < 0) or np.any(p_sell_t < 0) or np.any(tau_import_t < 0) or np.any(tau_export_t < 0):
            raise ValueError("Negative prices/tariffs are not supported by this simple processor.")
        if d_min_total < 0:
            raise ValueError("Minimum daily energy must be >= 0.")

        notes = {
            "pv_id": pv_id,
            "load_id": load_id,
            "assumptions": {
                "symmetric_tariff": False,
                "sell_price_equals_buy_price": True,
                "hour_resolution_h": 1.0,
            },
        }

        return ModelData(
            s_t=s_t,
            p_buy_t=p_buy_t,
            p_sell_t=p_sell_t,
            tau_import_t=tau_import_t,
            tau_export_t=tau_export_t,
            d_min_total=float(d_min_total),
            d_max_t=d_max_t,
            x_max_t=x_max_t,
            y_max_t=y_max_t,
            hours=hours,
            notes=notes,
        )

    # Convenience: return as dict for solvers that like dict inputs
    def as_dict(self) -> Dict[str, Any]:
        md = self.build_model_data()
        return {
            "s_t": md.s_t,
            "p_buy_t": md.p_buy_t,
            "p_sell_t": md.p_sell_t,
            "tau__import_t": md.tau__import_t,
            "tau__export_t": md.tau__export_t,
            "d_min_total": md.d_min_total,
            "d_max_t": md.d_max_t,
            "x_max_t": md.x_max_t,
            "y_max_t": md.y_max_t,
            "hours": md.hours,
            "notes": md.notes,
        }
