# Retrofit Debugging Analysis

This document outlines the investigation into why conventional generators are not dispatching correctly in baseline and retrofit scenarios.

## Analysis of `add_electricity.py`

Several areas in `workflow/scripts/add_electricity.py` could contribute to the issue:

*   **Carrier Sanitization (`sanitize_carriers`, `add_missing_carriers`)**: New carriers like `CCGT-retrofit` might not be fully defined in `config.yaml`, which could indicate incomplete integration.
    - Update: 
*   **Capital Cost Updates (`update_capital_costs`)**: Incorrect regional multipliers or missing state data could lead to wrong `capital_cost` values.
    - Update
*   **Dynamic Pricing (`apply_dynamic_pricing`)**: Errors in time-varying fuel cost data or geographical mapping could result in incorrect marginal costs.
    - Update: 
*   **Conventional Generator Attachment (`attach_conventional_generators`)**: This is a critical area with several potential failure points:
    *   **Data Filtering**: Plants might be incorrectly filtered out based on investment period or interconnection.
    *   **Cost Data Overwriting**: Logic in the script could be unintentionally overriding retrofitted values for `heat_rate`.
            - This would be happening in:
    *   **Parameter Assignment**: Errors in `efficiency` and `marginal_cost` from upstream scripts are carried over here.
            - Update: After fixing the heat rate -> efficiency function, the result is: 
    *   **Default Values**: Unrealistic unit commitment defaults (e.g., high `p_min_pu`) could prevent dispatch.
*   **Seasonal Derates (`apply_seasonal_capacity_derates`)**: Overly aggressive derates could limit generator output.
*   **PUDL Fuel Costs (`apply_pudl_fuel_costs`)**: This function appears incomplete and could be a source of conflicting cost data if active.

## Analysis of `build_cost_data.py`

The review of `workflow/scripts/build_cost_data.py`, which creates `costs.csv`, revealed the following:

*   **Default Emissions**: `CCGT-retrofit` is correctly assigned the same base `co2_emissions` as a standard `CCGT` (0.18058 tCO2/MWh_thermal).
*   **Lifetime Data**: `CCGT-retrofit` is correctly assigned a `lifetime` of 55 years, same as a standard `CCGT`.
*   **Cost Calculation**: The script correctly processes data from NREL ATB, EIA AEO, and EFS to calculate annualized capital costs.
*   **`CCGT-retrofit` Cost Definition**: The script does **not** explicitly define separate capital (`capex`) or fixed O&M (`fom`) costs for the `CCGT-retrofit` carrier. This means retrofitted plants likely inherit costs from standard `CCGT` plants, which could lead to inaccurate economic assessments.

## Synthesis and Final Diagnosis

A thorough review of the data pipeline (`build_powerplants.py` -> `apply_retrofit.py` -> `add_electricity.py`, and `build_cost_data.py`) points to the following likely causes for the dispatch failure:

1.  **Incorrect Heat Rate Imputation**: This is the most probable cause. The `impute_missing_plant_data` function in `build_powerplants.py` can assign unrealistically high `heat_rate` values to generators with missing data, making their `marginal_cost` non-competitive.
2.  **Excessive Heat Rate Penalty**: The `heat_rate_penalty` in `apply_retrofit.py` might be set too high in `config.yaml`, making retrofitted generators uneconomical.
3.  **Incorrect Fuel Costs**: Similar to the heat rate, imputed `fuel_cost` values in `build_powerplants.py` could be incorrectly high.
4.  **Source Data Errors**: The issue could stem from incomplete or incorrect entries in the source PUDL, CEMS, or ADS data.
5.  **Unit Commitment Constraints**: Restrictive constraints (e.g., high `p_min_pu`, long `min_up_time`) could prevent dispatch.
6.  **Incomplete `CCGT-retrofit` Handling**: The `CCGT-retrofit` carrier might not be fully handled, causing some parameters to fall back to incorrect default values.

## Recommendations

To resolve this issue, the following steps are recommended:

1.  **Inspect Output Files**: Manually inspect `powerplants.csv` and `costs.csv`. Look for generators with unusually high `heat_rate` or `marginal_cost` values. Check the `heat_rate_source` and `fuel_cost_source` columns to see if values are being imputed.
2.  **Review `config.yaml`**: Double-check the `heat_rate_penalty` value for the retrofit scenario.
3.  **Add Granular Logging**: Add logging to `build_powerplants.py` and `apply_retrofit.py` to trace `heat_rate`, `fuel_cost`, and `marginal_cost` for specific generators before and after imputation and retrofitting.
4.  **Improve Imputation Strategy**: If imputation is the root cause, develop a more sophisticated imputation method.
5.  **Trace a Single Generator**: Follow a single problematic generator through the entire data pipeline to see exactly how its parameters are being modified.


MISMATCH between weather year and investment period years: 2024 is a leap year but 2019 (the weather year) is not 

## Leap-Year Snapshot Mismatch (July 2025)

During a Baytown baseline run the workflow crashed inside
`broadcast_investment_horizons_index` with an `AssertionError`.
Diagnosis showed that the renewable-profile NetCDFs contain **8760** hourly
values (weather year 2019) while the network snapshots for some planning
horizons (e.g. 2024) expect **8784** hours.  The helper attempted to merge the
two without accounting for leap-day hours, so the final index lengths
diverged.

### Fix implemented
*  **Script modified:** `workflow/scripts/add_electricity.py`
*  **Function affected:** `broadcast_investment_horizons_index`
*  **Logic:**
   1.  Replicate the single-year profile for every investment period.
   2.  Re-index to the exact snapshot hours for that period, **forward-filling**
       any missing timestamps (covers 29-Feb).
   3.  Concatenate all periods and confirm the length matches `n.snapshots`.

This makes the code robust to mixing leap and non-leap source years without
regenerating the weather cut-outs. THIS IS BAD PRACTICE AND WE SHOULD SOLVE IT BY GETTING THE RIGHT WEATHER YEARS!!!!!

### July 2025 follow-up – conventional generators not dispatching

Root causes uncovered:
1. A "C" prefix added inside seasonal derate & must-run helpers broke
   alignment with generator IDs in the network, so p_max_pu and p_min_pu
   were multiplied by `NaN → 0`.
2. Generic ATB cost mapping overwrote retrofit-specific heat-rate and
   marginal-cost values, driving costs to ∞.

Fixes implemented (see `add_electricity.py`):
* Removed the ID prefix lines, leaving generator IDs unchanged.
* Replaced *overwrite* cost mapping with a `fillna` strategy and a final
  `combine_first` restore of plant-level overrides.

After these changes conventional & retrofit units commit and the model
solves.
