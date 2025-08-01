"""Combines all time independent cost data sources into a standard format."""

import logging

import constants as const
import duckdb
import pandas as pd
from _helpers import calculate_annuity
from build_sector_costs import (
    EfsIceTransportationData,
    EfsTechnologyData,
    EiaBuildingData,
)

logger = logging.getLogger(__name__)

# Impute emissions factor data
# https://www.eia.gov/environment/emissions/co2_vol_mass.php
# Units: [tCO2/MWh_thermal]
EMISSIONS_DATA = [
    {"pypsa-name": "coal", "parameter": "co2_emissions", "value": 0.3453},
    {"pypsa-name": "oil", "parameter": "co2_emissions", "value": 0.34851},
    {"pypsa-name": "geothermal", "parameter": "co2_emissions", "value": 0.04029},
    {"pypsa-name": "waste", "parameter": "co2_emissions", "value": 0.1016},
    {"pypsa-name": "gas", "parameter": "co2_emissions", "value": 0.18058},
    {"pypsa-name": "CCGT", "parameter": "co2_emissions", "value": 0.18058},
    {"pypsa-name": "OCGT", "parameter": "co2_emissions", "value": 0.18058},
    {"pypsa-name": "CCGT-retrofit", "parameter": "co2_emissions", "value": 0.18058},
    {
        "pypsa-name": "geothermal",
        "parameter": "heat_rate_mmbtu_per_mwh",
        "value": 8.881,
    },  # AEO 2023
]

LIFETIME_DATA = [
    {"pypsa-name": "coal", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "oil", "parameter": "lifetime", "value": 55},  # using gas CT
    # Confirm with Jabs / NREL. 30 is way too small
    {"pypsa-name": "geothermal", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "waste", "parameter": "lifetime", "value": 55},  # using gas CT
    {"pypsa-name": "CCGT", "parameter": "lifetime", "value": 55},
    {"pypsa-name": "CCGT-retrofit", "parameter": "lifetime", "value": 55},
    {"pypsa-name": "OCGT", "parameter": "lifetime", "value": 55},
    {"pypsa-name": "CCGT-95CCS", "parameter": "lifetime", "value": 55},
    {"pypsa-name": "CCGT-97CCS", "parameter": "lifetime", "value": 55},
    {"pypsa-name": "coal-95CCS", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "coal-99CCS", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "SMR", "parameter": "lifetime", "value": 40},
    {"pypsa-name": "nuclear", "parameter": "lifetime", "value": 60},
    {"pypsa-name": "biomass", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "offwind_floating", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "offwind", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "onwind", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "solar", "parameter": "lifetime", "value": 30},
    {
        "pypsa-name": "2hr_battery_storage",
        "parameter": "lifetime",
        "value": 20,
    },  # inquired with NREL on why they have CRP of 20 but lifetime of 15
    {"pypsa-name": "4hr_battery_storage", "parameter": "lifetime", "value": 20},
    {"pypsa-name": "6hr_battery_storage", "parameter": "lifetime", "value": 20},
    {"pypsa-name": "8hr_battery_storage", "parameter": "lifetime", "value": 20},
    {"pypsa-name": "10hr_battery_storage", "parameter": "lifetime", "value": 20},
]  # https://github.com/NREL/ReEDS-2.0/blob/e65ed5ed4ffff973071839481309f77d12d802cd/inputs/plant_characteristics/maxage.csv#L4


def create_duckdb_instance():
    """Set up DuckDB to read parquet files directly."""
    duckdb.connect(database=":memory:", read_only=False)
    # Install httpfs extension to access remote files if needed
    duckdb.query("INSTALL httpfs;")


def load_pudl_atb_data(parquet_path: str):
    """Loads ATB data directly from parquet files."""
    create_duckdb_instance()

    query = f"""
    WITH finance_cte AS (
        SELECT
            wacc_real,
            technology_description,
            model_case_nrelatb,
            scenario_atb,
            projection_year,
            cost_recovery_period_years,
            report_year
        FROM read_parquet('{parquet_path}/core_nrelatb__yearly_projected_financial_cases_by_scenario.parquet')
    )
    SELECT *
    FROM read_parquet('{parquet_path}/core_nrelatb__yearly_projected_cost_performance.parquet') atb
    LEFT JOIN finance_cte AS finance
        ON atb.technology_description = finance.technology_description
            AND atb.model_case_nrelatb = finance.model_case_nrelatb
            AND atb.scenario_atb = finance.scenario_atb
            AND atb.projection_year = finance.projection_year
            AND atb.cost_recovery_period_years = finance.cost_recovery_period_years
            AND atb.report_year = finance.report_year
    WHERE atb.report_year = 2024
    """
    return duckdb.query(query).to_df()


def load_pudl_aeo_data(parquet_path: str):
    """Loads AEO data directly from parquet files."""
    query = f"""
    SELECT *
    FROM read_parquet('{parquet_path}/core_eiaaeo__yearly_projected_fuel_cost_in_electric_sector_by_type.parquet') aeo
    WHERE aeo.report_year = 2023
    """
    return duckdb.query(query).to_df()


def match_technology(row, tech_dict):
    for key, value in tech_dict.items():
        # Match technology and techdetail
        if row["technology_description"] == value.get("technology") and row[
            "technology_description_detail_1"
        ] == value.get("techdetail"):
            return key
        # Match technology and techdetail2
        elif row["technology_description"] == value.get("technology") and row[
            "technology_description_detail_2"
        ] == value.get("techdetail2"):
            return key

    return None


def get_sector_costs(
    efs_tech_costs: str,
    efs_icev_costs: str,
    eia_tech_costs,
    year: int,
    additional_costs_csv: str | None = None,
) -> pd.DataFrame:
    """Gets end-use tech costs for sector coupling studies."""

    def correct_units(df: pd.DataFrame) -> pd.DataFrame:
        # USD/gal -> USD/MWh (water storage)
        # assume cp = 4.186 kJ/kg/C
        # (USD / gal) * (1 gal / 3.75 liter) * (1L / 1 kg H2O) = 0.267 USD / kg water
        # (0.267 USD / kg) * (1 / 4.186 kJ/kg/C) * (1 / 1C) = 0.0637 USD / kJ
        # (0.0637 USD / kJ) * (1000 kJ / 1 MJ) * (3600sec / 1hr) = 229335 USD / MWh
        df.loc[df.unit.str.contains("USD/gal"), "value"] *= 229335
        df.unit = df.unit.str.replace("USD/gal", "USD/MWh")

        df.unit = df.unit.str.replace("$/", "USD/")

        df.loc[df.unit.str.contains("/kW"), "value"] *= 1e3
        df.unit = df.unit.str.replace("/kW", "/MW")

        # 1 euro = 1.11 USD
        # taked on Sept. 2, 2024

        df.loc[df.unit.str.contains("EUR/"), "value"] *= 1.11
        df.unit = df.unit.str.replace("EUR/", "USD/")

        return df

    def get_investment_year_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
        return df[df.year == year].drop(columns="year")

    def calculate_capex(df: pd.DataFrame, discount_rate: float) -> pd.DataFrame:
        """Calcualtes capex based on annuity payments."""
        capex = df.copy().set_index(["technology", "parameter"])
        capex = capex.value.unstack().fillna(0)

        # n years should be
        # n.snapshot_weightings.loc[n.investment_periods[x]].objective.sum() / 8760.0

        capex["capital_cost"] = (
            (
                calculate_annuity(
                    capex["lifetime"],
                    discount_rate,
                )
                + capex["FOM"] / 100
            )
            * capex["investment"]
            * 1
        )

        capex = capex["capital_cost"].dropna().to_frame(name="value")

        investment = df[df.parameter == "investment"].set_index("technology")

        assert len(capex) == len(investment)

        final = capex.reindex_like(investment)
        final["parameter"] = "capital_cost"
        final["unit"] = investment.unit
        final["source"] = investment.source
        final["further_description"] = investment.further_description

        return final

    bev = EfsTechnologyData(efs_tech_costs).get_data("Transportation")
    ice = EfsIceTransportationData(efs_icev_costs).get_data()
    builidng = EiaBuildingData(eia_tech_costs).get_data()

    df = pd.concat([bev, ice, builidng])
    df = get_investment_year_data(df, year)
    df = df.rename(columns={"further description": "further_description"})

    if additional_costs_csv:
        additional = pd.read_csv(additional_costs_csv)
        assert all(x in df.columns for x in additional.columns)
        df = pd.concat([df, additional]).reset_index(drop=True)

    df = correct_units(df)

    discount_rate = 0.05
    capex = calculate_capex(df, discount_rate)
    capex = capex.reset_index()[df.columns]

    final = pd.concat([df, capex])
    final["value"] = final.value.round(4)

    final = final.rename(columns={"technology": "pypsa-name"})

    return final


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_cost_data", year=2030)
        rootpath = ".."
    else:
        rootpath = "."

    costs = snakemake.params.costs
    atb_params = costs.get("atb")
    aeo_params = costs.get("aeo")

    tech_year = snakemake.wildcards.year
    if int(tech_year) < 2024:
        logger.warning(
            "Minimum cost year supported is 2024, using 2024 expansion costs.",
        )
    years = range(2024, 2051)
    tech_year = min(years, key=lambda x: abs(x - int(tech_year)))

    emissions_data = EMISSIONS_DATA

    # Path to parquet files
    parquet_path = snakemake.params.pudl_path

    # Import PUDLs ATB data
    pudl_atb = load_pudl_atb_data(parquet_path)
    pudl_atb["pypsa-name"] = pudl_atb.apply(
        match_technology,
        axis=1,
        tech_dict=const.ATB_TECH_MAPPER,
    )
    pudl_atb = pudl_atb[pudl_atb["pypsa-name"].notnull()]

    # Group by pypsa-name and filter for correct cost recovery period
    pudl_atb = (
        pudl_atb.groupby("pypsa-name")[pudl_atb.columns]
        .apply(
            lambda x: x[x["cost_recovery_period_years"] == const.ATB_TECH_MAPPER[x.name].get("crp", 30)],
        )
        .reset_index(drop=True)
    )

    # Filter for the correct year, scenario, and model case
    pudl_atb_filt = pudl_atb[pudl_atb.projection_year == tech_year]
    if tech_year < 2030:
        logger.warning(
            "Using 2030 ATB data for offwind_floating; earlier data not available.",
        )
        pudl_atb_offwind_floating = pudl_atb[
            (pudl_atb["pypsa-name"] == "offwind_floating") & (pudl_atb.projection_year == 2030)
        ]
        pudl_atb = pd.concat(
            [pudl_atb_filt, pudl_atb_offwind_floating],
            ignore_index=True,
        )
    else:
        pudl_atb = pudl_atb_filt

    pudl_atb = pudl_atb[pudl_atb.scenario_atb == atb_params.get("scenario", "Moderate")]
    pudl_atb = pudl_atb[pudl_atb.model_case_nrelatb == atb_params.get("model_case", "Market")]

    pudl_premelt = pudl_atb.copy()
    # Pivot Data
    cols = [
        "cost_recovery_period_years",
        "capacity_factor",
        "capex_per_kw",
        "capex_overnight_per_kw",
        "capex_overnight_additional_per_kw",
        "capex_grid_connection_per_kw",
        "capex_construction_finance_factor",
        "fuel_cost_per_mwh",
        "heat_rate_mmbtu_per_mwh",
        "heat_rate_penalty",
        "levelized_cost_of_energy_per_mwh",
        "net_output_penalty",
        "opex_fixed_per_kw",
        "opex_variable_per_mwh",
        "wacc_real",
    ]
    # pivot such that cols all get moved to one column
    pudl_atb = pudl_atb.melt(
        id_vars="pypsa-name",
        value_vars=cols,
        var_name="parameter",
        value_name="value",
    )

    emissions_data = EMISSIONS_DATA

    # Impute Transmission Data
    # TEPCC 2023
    # WACC & Lifetime: https://emp.lbl.gov/publications/improving-estimates-transmission
    # Subsea costs: Purvins et al. (2018): https://doi.org/10.1016/j.jclepro.2018.03.095
    transmission_data = [
        {
            "pypsa-name": "HVAC overhead",
            "parameter": "capex_per_mw_km",
            "value": 2481.43,
        },
        {
            "pypsa-name": "HVAC overhead",
            "parameter": "cost_recovery_period_years",
            "value": 60,
        },
        {"pypsa-name": "HVAC overhead", "parameter": "wacc_real", "value": 0.044},
        {
            "pypsa-name": "HVDC overhead",
            "parameter": "capex_per_mw_km",
            "value": 1026.53,
        },
        {
            "pypsa-name": "HVDC overhead",
            "parameter": "cost_recovery_period_years",
            "value": 60,
        },
        {"pypsa-name": "HVDC overhead", "parameter": "wacc_real", "value": 0.044},
        {
            "pypsa-name": "HVDC submarine",
            "parameter": "capex_per_mw_km",
            "value": 504.141,
        },
        {
            "pypsa-name": "HVDC submarine",
            "parameter": "cost_recovery_period_years",
            "value": 60,
        },
        {"pypsa-name": "HVDC submarine", "parameter": "wacc_real", "value": 0.044},
        {
            "pypsa-name": "HVDC inverter pair",
            "parameter": "capex_per_kw",
            "value": 173.730,
        },
        {
            "pypsa-name": "HVDC inverter pair",
            "parameter": "cost_recovery_period_years",
            "value": 60,
        },
        {"pypsa-name": "HVDC inverter pair", "parameter": "wacc_real", "value": 0.044},
    ]
    pudl_atb = pd.concat(
        [
            pudl_atb,
            pd.DataFrame(emissions_data),
            pd.DataFrame(transmission_data),
            pd.DataFrame(LIFETIME_DATA),
        ],
        ignore_index=True,
    )
    pudl_atb = pudl_atb.drop_duplicates(
        subset=["pypsa-name", "parameter"],
        keep="last",
    )

    # Load AEO Fuel Cost Data
    aeo = load_pudl_aeo_data(parquet_path)
    aeo = aeo[aeo.projection_year == tech_year]
    aeo = aeo[aeo.model_case_eiaaeo == aeo_params.get("scenario", "Reference")]
    cols = ["fuel_type_eiaaeo", "fuel_cost_real_per_mmbtu_eiaaeo"]
    aeo = aeo[cols]
    aeo = aeo.groupby("fuel_type_eiaaeo").mean()
    aeo["fuel_cost_real_per_mwhth"] = aeo["fuel_cost_real_per_mmbtu_eiaaeo"] * 3.412
    aeo = pd.melt(
        aeo.reset_index(),
        id_vars="fuel_type_eiaaeo",
        value_vars=["fuel_cost_real_per_mwhth"],
        var_name="parameter",
        value_name="value",
    )
    aeo = aeo.rename(columns={"fuel_type_eiaaeo": "pypsa-name"})

    addnl_fuels = pd.DataFrame(
        [
            {
                "pypsa-name": "nuclear",
                "parameter": "fuel_cost_real_per_mwhth",
                "value": 2.782,
            },
            {
                "pypsa-name": "biomass",
                "parameter": "fuel_cost_real_per_mwhth",
                "value": 7.49,
            },
        ],
    )
    aeo = pd.concat([aeo, addnl_fuels], ignore_index=True)

    tech_fuel_map = {
        "CCGT": "natural_gas",
        "OCGT": "natural_gas",
        "CCGT-retrofit": "natural_gas",
        "CCGT-95CCS": "natural_gas",
        "CCGT-97CCS": "natural_gas",
        "coal-95CCS": "coal",
        "coal-99CCS": "coal",
        "SMR": "nuclear",
    }
    tech_fuels = pd.DataFrame(
        [
            {
                "pypsa-name": new_name,
                "parameter": "fuel_cost_real_per_mwhth",
                "value": aeo.loc[aeo["pypsa-name"] == source_name, "value"].values[0],
            }
            for new_name, source_name in tech_fuel_map.items()
        ],
    )
    aeo = pd.concat([aeo, tech_fuels], ignore_index=True)
    pudl_atb = pd.concat([pudl_atb, aeo], ignore_index=True)

    # Calculate Annualized Costs and Marinal Costs
    # Apply: marginal_cost = opex_variable_per_mwh + fuel_cost_real_per_mwhth / efficiency
    pivot_atb = pudl_atb.pivot(
        index="pypsa-name",
        columns="parameter",
        values="value",
    ).reset_index()

    # Create Hydrogen Combustion Turbine from OCGT using assumptions per ReEDS
    # https://nrel.github.io/ReEDS-2.0/model_documentation.html#hydrogen
    hydrogen_ct = pivot_atb[pivot_atb["pypsa-name"] == "OCGT"].copy()
    hydrogen_ct["pypsa-name"] = "hydrogen_ct"
    hydrogen_ct["capex_overnight_per_kw"] *= 1.2
    hydrogen_ct["capex_per_kw"] = (
        (hydrogen_ct["capex_overnight_per_kw"] + hydrogen_ct["capex_grid_connection_per_kw"])
        * hydrogen_ct["capex_construction_finance_factor"]
        / 100
    )
    hydrogen_ct["fuel_cost_real_per_mwhth"] = 20 * 3.412  # 20 USD/MMBtu * 3.412 MMBtu/MWh_th
    hydrogen_ct["co2_emissions"] = 0
    pivot_atb = pd.concat([pivot_atb, hydrogen_ct], ignore_index=True)

    pivot_atb["efficiency"] = 3.412 / pivot_atb["heat_rate_mmbtu_per_mwh"]
    pivot_atb["fuel_cost"] = pivot_atb["fuel_cost_real_per_mwhth"] / pivot_atb["efficiency"]
    pivot_atb["marginal_cost"] = pivot_atb["opex_variable_per_mwh"] + pivot_atb["fuel_cost"]

    # Impute storage WACC from Utility Scale Solar. TODO: Revisit this assumption
    for x in [2, 4, 6, 8, 10]:
        pivot_atb.loc[
            pivot_atb["pypsa-name"] == f"{x}hr_battery_storage",
            "wacc_real",
        ] = pivot_atb.loc[
            pivot_atb["pypsa-name"] == "solar",
            "wacc_real",
        ].values[0]
        pivot_atb.loc[
            pivot_atb["pypsa-name"] == f"{x}hr_battery_storage",
            "efficiency",
        ] = 0.85  # 2023 ATB

    pivot_atb["annualized_capex_per_mw"] = (
        calculate_annuity(
            pivot_atb["cost_recovery_period_years"],
            pivot_atb["wacc_real"],
        )
        * pivot_atb["capex_per_kw"]
        * 1
        # change to nyears
    ) * 1e3

    pivot_atb["annualized_capex_per_mw_km"] = (
        calculate_annuity(
            pivot_atb["cost_recovery_period_years"],
            pivot_atb["wacc_real"],
        )
        * pivot_atb["capex_per_mw_km"]
        * 1
        # change to nyears
    )

    # Calculate grid interrconnection costs per MW-KM
    # All land-based resources assume 1 mile of spur line
    # All offshore resources assume 30 km of subsea cable
    pivot_atb["capex_grid_connection_per_kw_km"] = pivot_atb["capex_grid_connection_per_kw"] / 1.609
    pivot_atb.loc[
        pivot_atb["pypsa-name"].str.contains("offshore"),
        "capex_grid_connection_per_kw_km",
    ] = pivot_atb["capex_grid_connection_per_kw"] / 30

    pivot_atb["annualized_connection_capex_per_mw_km"] = (
        calculate_annuity(
            pivot_atb["cost_recovery_period_years"],
            pivot_atb["wacc_real"],
        )
        * pivot_atb["capex_grid_connection_per_kw_km"]
        * 1
        # change to nyears
    )

    pivot_atb["annualized_capex_fom"] = pivot_atb["annualized_capex_per_mw"] + (pivot_atb["opex_fixed_per_kw"] * 1e3)
    pudl_atb = pivot_atb.melt(
        id_vars=["pypsa-name"],
        value_vars=pivot_atb.columns.difference(["pypsa-name"]),
        var_name="parameter",
        value_name="value",
    )
    pudl_atb = pudl_atb.reset_index(drop=True)
    pudl_atb["value"] = pudl_atb["value"].round(3)

    egs_costs = pd.read_csv(snakemake.input.egs_costs)
    egs_costs = egs_costs.query("investment_horizon == @tech_year").drop(
        columns="investment_horizon",
    )
    pudl_atb = pd.concat([pudl_atb, egs_costs], ignore_index=True)

    pudl_atb.to_csv(snakemake.output.tech_costs, index=False)

    # sector costs
    sector_costs = get_sector_costs(
        snakemake.input.efs_tech_costs,
        snakemake.input.efs_icev_costs,
        snakemake.input.eia_tech_costs,
        tech_year,
        snakemake.input.additional_costs,
    )
    sector_costs.to_csv(snakemake.output.sector_costs, index=False)
