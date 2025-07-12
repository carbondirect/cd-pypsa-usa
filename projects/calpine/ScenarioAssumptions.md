The initial transmission configuration will be using the TAMU ERCOT network, which should be similar to the network used by Sienna.

**Baytown Retrofit Workflow (v1)**
The Baytown retrofit is a scenario that simulates the addition of Carbon Capture and Storage (CCS) technology to the natural gas generators at the Baytown Energy Center (EIA Plant ID: 55327).

The workflow is controlled by your config.yaml file and implemented through a series of Snakemake rules and Python scripts.


Enabling the Retrofit setting: The entire workflow is triggered by a setting in the configuration file. When the user sets enable: retrofits: true in your config.yaml, it activates the apply_ccs_retrofit rule in the workflow/rules/build_electricity.smk file.

Executing the Retrofit Script: The apply_ccs_retrofit rule runs the workflow/scripts/apply_retrofit.py script.


Input: The script takes the baseline powerplants.csv file and the relevant technology cost file (e.g., costs_2030.csv) as inputs. This is also where we could include custom capital costs if we so desire. We use a 2019 weather year for renewables - it’s worth exploring if we can/should do a more recent weather year. I think these come from Copernicus so in principle more is available. The initial spatial config using the TAMU geospatial regions can handle up to 1338 clusters, but for expediency, I am initially using 200 clusters. 

Logic: The retrofit script identifies the two Baytown generators (CGT-1, CGT-2), applies a heat rate penalty (10% for now) and a CO2 capture rate (90% for now), and recalculates their marginal cost. Fuel costs can also be made to be dynamic.
Output: It saves the modified list of power plants to a new file: resources/powerplants_ccs.csv.
Integration into the Main Workflow: The add_electricity rule, which is a core part of building the PyPSA network, conditionally selects the power plants file to use.
If enable: retrofits: is true, it uses the modified powerplants_ccs.csv file.
If false or not present, it uses the original powerplants.csv.
This ensures that the rest of the PyPSA-USA workflow (network simplification, clustering, solving) will automatically use the characteristics of the retrofitted Baytown plant for that specific model run.
Key Assumptions
This workflow relies on a set of clear assumptions that are important for interpreting the results:
Targeted Retrofit: Only the specified generators at the Baytown plant are retrofitted. All other plants in the model remain unchanged.
Operational Penalties:
Efficiency Loss: Retrofitting imposes a 10% heat rate penalty, making the plant less efficient. 
CO2 Capture: The CCS technology is assumed to capture 90% of the CO2 emissions.
Cost Adjustments: The operational marginal cost is increased to reflect the higher fuel consumption due to the heat rate penalty.
No Capital Costs in Script: The apply_retrofit.py script itself does not add the significant capital cost of installing the CCS equipment. This cost would need to be included in the cost data files that are used by the main PyPSA model to ensure it is properly accounted for during investment optimization.
Static Assumptions: The heat rate penalty and capture rate are assumed to be constant and do not change based on the plant's operating conditions.


## Things to be aware of 
- Coal plant retirements might not be captured in the current PUDL query, or are a little underspecified. This shouldn't matter too much for production cost modeling since we are only modeling a single year. 
- We should consider dynamic fuel costs and how these might impact wholesale power prices (this is a configurable option)
- We are ignoring capital costs in this model and solely focusing on operational changes resulting from the heat rate penalty.
- Handling export into SPP can probably be done by simply adding a node (or nodes) to the network with some set level of demand. 
- Questions of net capacity reduction vs heat rate penalty need to be resolved. For instance, if I take a 10% heat rate hit, do I also need to model the parasitic load's impact on net capacity? Something to think about. 
