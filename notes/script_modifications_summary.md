# Summary of PyPSA-USA Workflow Modifications

This document outlines the key logical changes made to the PyPSA-USA workflow scripts to support a comparative analysis between a baseline and a "retrofit" scenario.

## 1. `workflow/rules/build_electricity.smk`

The initial problem was that the workflow was not selecting the correct power plant data file for the retrofit scenario.

-   **Dynamic Input File Selection**: The `add_electricity` rule was modified to use a dedicated function, `get_powerplants_input`. This function checks the snakemake `config` and provides `powerplants.retrofitted.csv` as the input if `config["retrofits"]["enable"]` is true, otherwise it defaults to `powerplants.csv`.
-   **Input Name Correction**: The rule's network input was renamed from `base_network` to `network` to match the name expected by the `add_electricity.py` script, resolving an `AttributeError`.

## 2. `workflow/scripts/apply_retrofit.py`

This script is responsible for creating the `powerplants.retrofitted.csv` file. It was updated to ensure the data for retrofitted plants was correct and consistent.

-   **Carrier Renaming**: Added logic to explicitly rename the `carrier` of affected plants (e.g., from `CCGT` to `CCGT-retrofit`). This is essential for distinguishing them in downstream analysis.
-   **Efficiency Calculation**: Added a line to recalculate the `efficiency` based on the newly penalized `heat_rate` (`efficiency = 1 / heat_rate`). This ensures the data remains physically consistent.

## 3. `workflow/scripts/add_electricity.py`

This script saw the most significant changes. The primary goal was to prevent generic cost data from overwriting the specific modifications made in the retrofit scenario.

-   **Preservation of Retrofit Data**: The `attach_conventional_generators` function was refactored. It now:
    1.  Creates a temporary copy of the incoming power plant data (`plants_original`).
    2.  Applies the generic, carrier-wide costs from the main `costs.csv` file.
    3.  Uses `plants_regional.update(plants_original)` to restore the specific values from the input file (e.g., the higher `heat_rate` and lower `efficiency` for `CCGT-retrofit` plants), preventing them from being overwritten.

-   **Robust Handling of Optional Cost Overrides**:
    The script was crashing with a `KeyError` if certain optional cost structures (like `vom_cost`) were not defined in the snakemake configuration. This was fixed by wrapping the logic for each optional cost override in a check, like `if "vom_cost" in conventional_inputs:`. This makes the script more resilient and prevents crashes when these optional configurations are omitted.

-   **Bug Fixes**:
    -   A reference to a non-existent `efficiency_r` column was removed to fix an `AttributeError`. This was a leftover from a previous version of the code. 

    ## 4. `build_cost_data.py`
     - Updated to include cost data for the CCGT retrofit carrier. This lack of cost data might have been what was preventing the retrofit logic from applying downstream. Since CCGT-retrofit can essentially be treated as its own carrier 

## 5. `workflow/scripts/add_electricity.py` (July 2025)

**Issue:** Runs that include a leap-year planning horizon (e.g. 2024) crashed
with an `AssertionError` in `broadcast_investment_horizons_index` because the
renewable-profile DataFrames (8760 hours from weather year 2019) were being
matched against network snapshots that include 29-Feb (8784 hours).

**Change:** The helper `broadcast_investment_horizons_index` was rewritten to

1. clone the 8760-hour profile for each investment year,
2. align it to the exact set of hourly snapshots for that year,
3. forward-fill any missing hours (thus supplying 29-Feb in leap years), and
4. perform a final consistency check.

This makes the workflow calendar-agnostic and eliminates the crash without
regenerating weather data. 

### 6. Conventional-generator robustness (July 2025)

* **attach_conventional_generators** now maps generic cost columns with
  `fillna`, thereby *preserving* any plant-level overrides such as those
  coming from the retrofit script.
* Introduced a final `combine_first` with the original plant table so
  retrofit-specific heat-rate, efficiency and marginal-cost survive later
  carrier-level overrides.
* Removed the temporary "C" prefix for generator IDs in
  `apply_seasonal_capacity_derates`, `apply_must_run_ratings`, and PuDL
  fuel-cost alignment, fixing the silent p-max-pu = 0 issue. 