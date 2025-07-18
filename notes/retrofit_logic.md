Retrofit Configuration:
- The system uses a configuration-driven approach where retrofit targets are specified in the config file under config["retrofits"]["targets"]

Each retrofit target is a dictionary containing:
- plant_id_eia: The EIA plant ID to retrofit
- generator_ids: List of specific generator IDs at that plant to retrofit
- heat_rate_penalty: Optional multiplier for heat rate (defaults to 1.0)
- co2_capture_rate: Optional CO2 capture rate to apply

Retrofit Application Process (from apply_retrofit.py):
- The apply_retrofit_penalties function processes each retrofit target and:
 - Validates that the plant and generator IDs exist
 - Applies the heat rate penalty by multiplying the existing heat rate
 - Updates the efficiency (calculated as 1/heat rate)
 - Changes the carrier from "CCGT" to "CCGT-retrofit"
 - Marks the plant as retrofit_penalized = True
 - Sets land_region to prevent clustering
 - If specified, applies the CO2 capture rate by reducing emissions
 - Cost and Emissions Data:
 - From build_cost_data.py, we can see that CCGT-retrofit:
 - Uses the same base CO2 emissions as regular CCGT (0.18058 tCO2/MWh_thermal)
 - Has the same lifetime as regular CCGT (55 years)
 - Uses natural gas as fuel with the same fuel cost as regular CCGT


Integration with Network:
- The attach_conventional_generators function in add_electricity.py has special handling to preserve retrofit-specific values:
- Makes a copy of original plant data before mapping generic costs
- Updates with original values to preserve retrofit-specific data
- Applies any custom VOM costs or fuel costs if specified

Key Effects of Retrofit:
- Increased heat rate (reduced efficiency) due to the heat rate penalty
- Reduced CO2 emissions if a capture rate is specified
- Preserved from clustering to maintain individual plant characteristics
- Maintains separate tracking from regular CCGTs for analysis (THIS CURRENTLY HAS A BUG)

## July 2025 – Robust cost & ID handling

Two model–blocking issues were fixed:

1. **Generator-ID consistency**  
   Seasonal capacity-derate and must-run functions used an artificial
   "C" prefix that broke column alignment with the network generator
   table.  The prefix was removed everywhere; the original IDs are now
   used consistently from `build_powerplants` → `apply_retrofit` →
   `add_electricity` → solver.

2. **Cost-mapping order**  
   Generic ATB/AEO cost columns are now applied with
   `fillna()`, i.e. *only* where plant-level data are missing.  A final
   `combine_first` with the pre-mapped DataFrame guarantees the
   retrofit-specific heat-rate, efficiency, marginal-cost and capital-cost
   values win over generic mappings and over any subsequent
   carrier-level overrides.

This clean layering eliminates the inadvertent overwriting that caused
retrofit units to end up with prohibitive marginal costs or missing cost
parameters altogether, which in turn had been preventing them from
dispatching and leading to infeasible models.
