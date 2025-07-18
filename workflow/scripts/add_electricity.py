# PyPSA USA Authors
"""
Adds existing conventional generators, renewable generators, and storage devices to the network.

This script will add all generator unit availabilities (capacity-factors) to the network, for all investment horizons.
"""

import logging
import os

import constants as const
import dill as pickle
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import (
    calculate_annuity,
    configure_logging,
    export_network_for_gis_mapping,
    update_p_nom_max,
    weighted_avg,
)
from sklearn.neighbors import BallTree

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def sanitize_carriers(n, config):
    """
    Sanitize the carrier information in a PyPSA Network object.

    The function ensures that all unique carrier names are present in the network's
    carriers attribute, and adds nice names and colors for each carrier according
    to the provided configuration dictionary.

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA Network object that represents an electrical power system.
    config : dict
        A dictionary containing configuration information, specifically the
        "plotting" key with "nice_names" and "tech_colors" keys for carriers.

    Returns
    -------
    None
        The function modifies the 'n' PyPSA Network object in-place, updating the
        carriers attribute with nice names and colors.

    Warnings
    --------
    Raises a warning if any carrier's "tech_colors" are not defined in the config dictionary.
    """
    for c in n.iterate_components():
        if "carrier" in c.df:
            add_missing_carriers(n, c.df.carrier)

    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"]).reindex(carrier_i).fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = n.carriers.nice_name.where(
        n.carriers.nice_name != "",
        nice_names,
    )
    colors = pd.Series(config["plotting"]["tech_colors"]).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = n.carriers.color.where(n.carriers.color != "", colors)


def add_missing_carriers(n, carriers):
    """Function to add missing carriers to the network without raising errors."""
    missing_carriers = set(carriers) - set(n.carriers.index)
    if len(missing_carriers) > 0:
        n.madd("Carrier", missing_carriers)


def clean_locational_multiplier(df: pd.DataFrame):
    """Updates format of locational multiplier data."""
    df = df.fillna(1)
    df = df[["State", "Location Variation"]]
    return df.groupby("State").mean()


def update_capital_costs(
    n: pypsa.Network,
    carrier: str,
    costs: pd.DataFrame,
    multiplier: pd.DataFrame,
):
    """Applies regional multipliers to capital cost data."""
    # map generators to states
    bus_state_mapper = n.buses.to_dict()["state"]
    gen = n.generators[n.generators.carrier == carrier].copy()
    gen["state"] = gen.bus.map(bus_state_mapper)
    gen = gen[gen["state"].isin(multiplier.index)]  # drops any regions that do not have cost multipliers

    # log any states that do not have multipliers attached
    missed = gen[~gen["state"].isin(multiplier.index)]
    if not missed.empty:
        logger.warning(f"CAPEX cost multiplier not applied to {missed.state.unique()}")

    # apply multiplier to annualized capital investment cost
    gen["annualized_capex_per_mw"] = gen.apply(
        lambda x: costs.at[carrier, "annualized_capex_per_mw"] * multiplier.at[x["state"], "Location Variation"],
        axis=1,
    )

    # get fixed costs based on overnight capital costs with multiplier applied
    gen["fom"] = costs.at[carrier, "opex_fixed_per_kw"] * 1e3

    # find final annualized capital cost
    gen["capital_cost"] = gen["annualized_capex_per_mw"] + gen["fom"]

    # overwrite network generator dataframe with updated values
    n.generators.loc[gen.index] = gen


def apply_dynamic_pricing(
    n: pypsa.Network,
    carrier: str,
    geography: str,
    df: pd.DataFrame,
    vom: float = 0,
):
    """
    Applies user-supplied dynamic pricing.

    Arguments
    ---------
    n: pypsa.Network,
    carrier: str,
        carrier to apply fuel cost data to (ie. Gas)
    geography: str,
        column of geography to search over (ie. balancing_area, state, reeds_zone, ...)
    df: pd.DataFrame,
        Fuel costs data
    vom: float = 0
        Additional flat $/MWh cost to add onto the fuel costs
    """
    assert geography in n.buses.columns

    gens = n.generators.copy()
    gens[geography] = gens.bus.map(n.buses[geography])
    gens = gens[(gens.carrier == carrier) & (gens[geography].isin(df.columns))]

    if gens.empty:
        return

    eff = n.get_switchable_as_dense("Generator", "efficiency").T
    eff = eff[eff.index.isin(gens.index)].T
    eff.columns.name = ""

    fuel_cost_per_gen = {gen: df[gens.at[gen, geography]] for gen in gens.index}
    fuel_costs = pd.DataFrame.from_dict(fuel_cost_per_gen)
    fuel_costs.index = pd.to_datetime(fuel_costs.index)
    fuel_costs = broadcast_investment_horizons_index(n, fuel_costs)

    marginal_costs = fuel_costs.div(eff, axis=1)
    marginal_costs = marginal_costs + vom

    # drop any data that has been assigned at a coarser resolution
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"][
        [x for x in n.generators_t["marginal_cost"] if x not in marginal_costs]
    ]

    # assign new marginal costs
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(
        marginal_costs,
        how="inner",
    )


def update_transmission_costs(n, costs, length_factor=1.0):
    # TODO: line length factor of lines is applied to lines and links.
    # Separate the function to distinguish
    n.lines["capital_cost"] = (
        n.lines["length"] * length_factor * costs.at["HVAC overhead", "annualized_capex_per_mw_km"]
    )

    if n.links.empty:
        return

    dc_b = n.links.carrier == "DC"

    # If there are no dc links, then the 'underwater_fraction' column
    # may be missing. Therefore we have to return here.
    if n.links.loc[dc_b].empty:
        return

    costs = (
        n.links.loc[dc_b, "length"]
        * length_factor
        * (
            (1.0 - n.links.loc[dc_b, "underwater_fraction"]) * costs.at["HVDC overhead", "annualized_capex_per_mw_km"]
            + n.links.loc[dc_b, "underwater_fraction"] * costs.at["HVDC submarine", "annualized_capex_per_mw_km"]
        )
        + costs.at["HVDC inverter pair", "annualized_capex_per_mw"]
    )
    n.links.loc[dc_b, "capital_cost"] = costs


def load_powerplants(
    plants_fn,
    investment_periods: list[int],
    interconnect: str | None = None,
) -> pd.DataFrame:
    plants = pd.read_csv(
        plants_fn,
    )
    # Filter out non-conus plants and plants that are not built by first investment period.
    plants = plants.set_index("generator_name")
    plants = plants[plants.build_year <= investment_periods[0]]
    plants = plants[plants.nerc_region != "non-conus"]
    if (interconnect is not None) & (interconnect != "usa"):
        plants["interconnection"] = plants["nerc_region"].map(const.NERC_REGION_MAPPER)
        plants = plants[plants.interconnection == interconnect]
    return plants


def match_nearest_bus(plants_subset, buses_subset):
    """Assign the nearest bus to each plant in the given subsets."""
    if plants_subset.empty or buses_subset.empty:
        return plants_subset

    # Create a BallTree for the given subset of buses
    tree = BallTree(buses_subset[["x", "y"]].values, leaf_size=2)

    # Find nearest bus for each plant in the subset
    distances, indices = tree.query(
        plants_subset[["longitude", "latitude"]].values,
        k=1,
    )

    # Map the nearest bus information back to the plants subset
    plants_subset["bus_assignment"] = buses_subset.reset_index().iloc[indices.flatten()]["Bus"].values
    plants_subset["distance_nearest"] = distances.flatten()

    return plants_subset


def match_plant_to_bus(n, plants):
    """
    Matches each plant to it's corresponding bus in the network enfocing a
    match to the correct State.

    # Efficient matching taken from #
    https://stackoverflow.com/questions/58893719/find-nearest-point-in-other-dataframe-with-a-lot-of-data
    """
    plants_matched = plants.copy()
    plants_matched["bus_assignment"] = None
    plants_matched["distance_nearest"] = None

    # Get a copy of buses and create a geometry column with GPS coordinates
    buses = n.buses.copy()
    buses["geometry"] = gpd.points_from_xy(buses["x"], buses["y"])

    # First pass: Assign each plant to the nearest bus in the same state
    for state in buses["state"].unique():
        buses_in_state = buses[buses["state"] == state]
        plants_in_state = plants_matched[
            (plants_matched["state"] == state) & (plants_matched["bus_assignment"].isnull())
        ]

        # Update plants_matched with the nearest bus within the same state
        plants_matched.update(match_nearest_bus(plants_in_state, buses_in_state))

    # Second pass: Assign any remaining unmatched plants to the nearest bus regardless of state
    unmatched_plants = plants_matched[plants_matched["bus_assignment"].isnull()]
    if not unmatched_plants.empty:
        plants_matched.update(match_nearest_bus(unmatched_plants, buses))

    return plants_matched


def filter_plants_by_region(
    plants: pd.DataFrame,
    regions_onshore: gpd.GeoDataFrame,
    regions_offshore: gpd.GeoDataFrame,
    reeds_shapes: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Filters the plants dataframe to remove plants not within the onshore and
    offshore geometries.
    """
    plants = plants.copy()
    plants["geometry"] = gpd.points_from_xy(
        plants.longitude,
        plants.latitude,
        crs="EPSG:4326",
    )
    gdf_plants = gpd.GeoDataFrame(plants, geometry="geometry")
    plants_onshore = gpd.sjoin(gdf_plants, regions_onshore, how="inner")
    plants_offshore = gpd.sjoin(gdf_plants, regions_offshore, how="inner")
    if not plants_offshore.empty:
        logger.warning(f"Offshore plants: {plants_offshore}")
    plants_filt = pd.concat([plants_onshore, plants_offshore])

    # Some plants like Diablo Canyon near oceans don't have region due to
    # imprecise ReEDS Shapes. We filter plants that have no reeds regions,
    # then search these points again.
    plants_in_regions = gpd.sjoin(
        gdf_plants,
        reeds_shapes,
        how="inner",
        predicate="intersects",
    )
    plants_no_region = gdf_plants[~gdf_plants.index.isin(plants_in_regions.index)]
    if not plants_no_region.empty:
        plants_no_region = plants_no_region.to_crs(epsg=3857)
        plants_nearshore = gpd.sjoin_nearest(
            plants_no_region,
            regions_onshore.to_crs(epsg=3857),
            how="inner",
            max_distance=2000,
            distance_col="distance",
        )
        plants_nearshore = plants_nearshore.to_crs(epsg=4326)
        plants_filt = pd.concat([plants_filt, plants_nearshore])

    plants_filt = plants_filt.drop(columns=["geometry"])
    plants_filt = plants_filt[~plants_filt.index.duplicated()]

    plants_filt[plants_filt.index.str.contains("Diablo")]
    gdf_plants[gdf_plants.index.str.contains("Diablo")]
    return pd.DataFrame(plants_filt)


def attach_renewable_capacities_to_atlite(
    n: pypsa.Network,
    plants_df: pd.DataFrame,
    renewable_carriers: list,
):
    plants = plants_df.query(
        "bus_assignment in @n.buses.index",
    )
    for tech in renewable_carriers:
        plants_filt = plants.query("carrier == @tech").copy()
        if plants_filt.empty:
            continue

        generators_tech = n.generators[n.generators.carrier == tech].copy()
        generators_tech["sub_assignment"] = generators_tech.bus.map(n.buses.sub_id)
        plants_filt["sub_assignment"] = plants_filt.bus_assignment.map(n.buses.sub_id)

        build_year_avg = plants_filt.groupby(["sub_assignment"])[plants_filt.columns].apply(
            lambda x: pd.Series(
                {field: weighted_avg(x, field, "p_nom") for field in ["build_year"]},
            ),
        )

        caps_per_bus = (
            plants_filt[["sub_assignment", "p_nom"]].groupby("sub_assignment").sum().p_nom
        )  # namplate capacity per sub_id

        if caps_per_bus[~caps_per_bus.index.isin(generators_tech.sub_assignment)].sum() > 0:
            # p_all = plants_filt[["sub_assignment", "p_nom", "latitude", "longitude"]]
            # missing_plants = p_all[~p_all.sub_assignment.isin(generators_tech.sub_assignment)]
            missing_capacity = caps_per_bus[~caps_per_bus.index.isin(generators_tech.sub_assignment)].sum()
            # missing_plants.to_csv(f"missing_{tech}_plants.csv",)

            logger.info(
                f"There are {np.round(missing_capacity / 1000, 4)} GW of {tech} plants that are not in the network. See git issue #16.",
            )

        logger.info(
            f"{np.round(caps_per_bus.sum() / 1000, 2)} GW of {tech} capacity added.",
        )
        mapped_values = generators_tech.sub_assignment.map(caps_per_bus).dropna()
        n.generators.loc[mapped_values.index, "p_nom"] = mapped_values
        n.generators.loc[mapped_values.index, "p_nom_min"] = mapped_values
        mapped_values = generators_tech.sub_assignment.map(
            build_year_avg.build_year,
        ).dropna()
        n.generators.loc[mapped_values.index, "build_year"] = mapped_values.astype(int)


def attach_conventional_generators(
    n: pypsa.Network,
    costs: pd.DataFrame,
    plants: pd.DataFrame,
    conventional_carriers: list,
    extendable_carriers: list,
    conventional_params,
    renewable_carriers: list,
    conventional_inputs,
    unit_commitment=None,
    fuel_price=None,
):
    """
    Attaches conventional generators to the network.

    This function processes a DataFrame of power plants, filters for
    conventional carriers, and adds them to the PyPSA network object. It
    handles retrofitting logic, cost updates, and other generator-specific
    attributes.
    """
    # get retrofit plants from config
    retro_cfg = conventional_inputs.get("retrofits", {})
    retro_targets = retro_cfg.get("targets", [])
    retro_pids = [d["plant_id_eia"] for d in retro_targets]

    # Store a copy of the original plant data to preserve retrofit-specific values
    plants_og = plants.copy()

    # Filter out plants that are targeted for retrofitting, these will be handled later
    if retro_cfg.get("enable", False):
        retrofit_target_plants = plants[plants.plant_id_eia.isin(retro_pids)]
        plants = plants[~plants.plant_id_eia.isin(retro_pids)]

    # # Filter out plants that are targeted for retrofitting so they can be handled seperately
    # retrofit_target_plants = plants[plants.plant_id_eia.isin(retro_pids)]
    # plants = plants[~plants.plant_id_eia.isin(retro_pids)]

    plants = plants[plants.carrier.isin(conventional_carriers)].copy()
    plants = plants[~plants.index.duplicated(keep="first")]

    for tech in conventional_carriers:
        plants_regional = plants.copy()
        logger.info(f"Region plant attributes (initial) {plants_regional.columns}")
        # Store a copy of the original plant data to preserve retrofit-specific values
        plants_original = plants_regional.copy()

        # -------------------------------------------------------------
        # Map generic cost parameters ONLY where the plant-level value
        # is missing.  This keeps retrofit-specific overrides intact.
        # -------------------------------------------------------------
        for param in costs.columns:
            mapped = plants_regional.carrier.map(costs[param])
            if param in plants_regional.columns:
                plants_regional[param] = plants_regional[param].fillna(mapped)
            else:
                plants_regional[param] = mapped

        # Finally, restore any retrofit-specific values that still differ
        # from the generic mapping.  combine_first keeps non-NaNs from the
        # original and fills remaining NaNs from the generic columns.
        plants_regional = plants_original.combine_first(plants_regional)

        # if a VOM is provided for a carrier, overwrite the cost data
        if "vom_cost" in conventional_inputs:
            for carrier, vom in conventional_inputs["vom_cost"].items():
                if carrier in conventional_carriers:
                    plants_regional.loc[plants_regional.carrier == carrier, "vom_cost"] = vom

        # if a fuel_cost is provided for a carrier, overwrite the cost data
        if "fuel_cost" in conventional_inputs:
            for carrier, fuel_cost in conventional_inputs["fuel_cost"].items():
                if carrier in conventional_carriers:
                    plants_regional.loc[
                        plants_regional.carrier == carrier, "fuel_cost"
                    ] = fuel_cost

        # if a heat_rate is provided for a carrier, overwrite the cost data
        if "heat_rate" in conventional_inputs:
            for carrier, heat_rate in conventional_inputs["heat_rate"].items():
                if carrier in conventional_carriers:
                    plants_regional.loc[
                        plants_regional.carrier == carrier, "heat_rate"
                    ] = heat_rate

        # if a balancing_authority_code is provided for a carrier, overwrite the cost data
        if "ba_eia" in conventional_inputs:
            for carrier, ba_eia in conventional_inputs["ba_eia"].items():
                if carrier in conventional_carriers:
                    plants_regional.loc[plants_regional.carrier == carrier, "ba_eia"] = ba_eia

        # if a ads_balancing_area is provided for a carrier, overwrite the cost data
        if "ba_ads" in conventional_inputs:
            for carrier, ba_ads in conventional_inputs["ba_ads"].items():
                if carrier in conventional_carriers:
                    plants_regional.loc[
                        plants_regional.carrier == carrier, "ba_ads"
                    ] = ba_ads

        plants_regional["efficiency"] = plants_regional["efficiency"].astype(float)
        logger.info(f"Regional plant attributes (after effiency and cost mapping) {plants_regional.columns}")
        plants_regional.loc[:, "p_min_pu"] = (
            plants_regional.minimum_load_mw / plants_regional.p_nom
        )
        plants_regional.loc[:, "p_min_pu"] = (
            plants_regional.p_min_pu.clip(
                upper=np.minimum(plants_regional.summer_derate, plants_regional.winter_derate),
                lower=0,
            )
            .astype(float)
            .fillna(0)
        )
        committable_fields = ["start_up_cost", "min_down_time", "min_up_time", "p_min_pu"]
        for attr in committable_fields:
            default = pypsa.components.component_attrs["Generator"].default[attr]
            if unit_commitment:
                plants_regional[attr] = plants_regional[attr].astype(float).fillna(default)
            else:
                plants_regional[attr] = default
        committable_attrs = {attr: plants_regional[attr] for attr in committable_fields}

        # Define generators using modified ppl DataFrame
        caps = plants_regional.groupby("carrier").p_nom.sum().div(1e3).round(2)
        logger.info(f"Adding {len(plants_regional)} generators with capacities [GW] \n{caps}")
        n.madd(
            "Generator",
            plants_regional.index,
            carrier=plants_regional.carrier,
            bus=plants_regional.bus_assignment.astype(str),
            p_nom_min=plants_regional.p_nom.where(
                plants_regional.carrier.isin(conventional_carriers),
                0,
            ),  # enforces that plants cannot be retired/sold-off at their capital cost
            p_nom=plants_regional.p_nom.where(plants_regional.carrier.isin(conventional_carriers), 0),
            p_nom_extendable=plants_regional.carrier.isin(extendable_carriers["Generator"]),
            ramp_limit_up=plants_regional.ramp_limit_up,
            ramp_limit_down=plants_regional.ramp_limit_down,
            efficiency=plants_regional.efficiency.round(3),
            marginal_cost=plants_regional.marginal_cost,
            capital_cost=plants_regional.annualized_capex_fom,
            build_year=plants_regional.build_year.astype(int).fillna(0),
            lifetime=plants_regional.carrier.map(costs.lifetime),
            committable=unit_commitment,
            **committable_attrs,
        )

        # Add fuel and VOM costs to the network
        n.generators.loc[plants_regional.index, "vom_cost"] = plants_regional.carrier.map(
            costs.opex_variable_per_mwh,
        )
        n.generators.loc[plants_regional.index, "fuel_cost"] = plants_regional.fuel_cost
        # Use the unified column name coming from build_powerplants.py
        n.generators.loc[plants_regional.index, "heat_rate"] = (
            plants_regional.heat_rate
        )

        # ------------------------------------------------------------------
        # Ensure carrier-level CO₂ intensity reflects any retrofit penalties
        # ------------------------------------------------------------------
        if "co2_emissions" in plants_regional.columns:
            carrier_emissions = (
                plants_regional.groupby("carrier")["co2_emissions"].first()
            )
            for carrier, emis in carrier_emissions.items():
                if not np.isnan(emis):
                    n.carriers.at[carrier, "co2_emissions"] = emis
        n.generators.loc[plants_regional.index, "ba_eia"] = plants_regional.balancing_authority_code
        n.generators.loc[plants_regional.index, "ba_ads"] = plants_regional.ads_balancing_area


def normed(s):
    return s / s.sum()


def attach_wind_and_solar(
    n: pypsa.Network,
    costs: pd.DataFrame,
    input_profiles: str,
    carriers: list[str],
    extendable_carriers: dict[str, list[str]],
):
    """
    Attached Atlite Calculated wind and solar capacity factor profiles to the
    network.
    """
    add_missing_carriers(n, carriers)
    for car in carriers:
        if car in ["hydro", "EGS"]:
            continue

        with xr.open_dataset(getattr(input_profiles, "profile_" + car)) as ds:
            if ds.indexes["bus"].empty:
                continue

            capital_cost = costs.at[car, "annualized_capex_fom"]

            bus2sub = (
                pd.read_csv(input_profiles.bus2sub, dtype=str)
                .drop("interconnect", axis=1)
                .rename(columns={"Bus": "bus_id"})
                .drop_duplicates(subset="sub_id")
            )
            bus_list = ds.bus.to_dataframe("sub_id").merge(bus2sub).bus_id.astype(str).values
            p_nom_max_bus = (
                ds["p_nom_max"]
                .to_dataframe()
                .merge(bus2sub[["bus_id", "sub_id"]], left_on="bus", right_on="sub_id")
                .set_index("bus_id")
                .p_nom_max
            )
            weight_bus = (
                ds["weight"]
                .to_dataframe()
                .merge(bus2sub[["bus_id", "sub_id"]], left_on="bus", right_on="sub_id")
                .set_index("bus_id")
                .weight
            )
            bus_profiles = (
                ds["profile"]
                .transpose("time", "bus")
                .to_pandas()
                .T.merge(
                    bus2sub[["bus_id", "sub_id"]],
                    left_on="bus",
                    right_on="sub_id",
                )
                .set_index("bus_id")
                .drop(columns="sub_id")
                .T
            )
            bus_profiles = broadcast_investment_horizons_index(n, bus_profiles)

            logger.info(f"Adding {car} capacity-factor profiles to the network.")

            n.madd(
                "Generator",
                bus_list,
                " " + car,
                bus=bus_list,
                carrier=car,
                p_nom_extendable=car in extendable_carriers["Generator"],
                p_nom_max=p_nom_max_bus,
                weight=weight_bus,
                marginal_cost=costs.at[car, "marginal_cost"],
                capital_cost=capital_cost,
                efficiency=1,
                build_year=n.investment_periods[0],
                lifetime=costs.at[car, "lifetime"],
                p_max_pu=bus_profiles,
            )


def attach_egs(
    n: pypsa.Network,
    costs: pd.DataFrame,
    input_profiles: str,
    carriers: list[str],
    extendable_carriers: dict[str, list[str]],
    line_length_factor=1,
):
    """
    Attached STM Calculated wind and solar capacity factor profiles to the
    network.
    """
    car = "EGS"
    if (car not in carriers) and (car not in extendable_carriers["Generator"]):
        return
    add_missing_carriers(n, carriers)
    capital_recovery_period = 25  # Following EGS supply curves by Aljubran et al. (2024)
    discount_rate = 0.07  # load_costs(snakemake.input.tech_costs).loc["geothermal", "wacc_real"]
    drilling_cost = snakemake.config["renewable"]["EGS"]["drilling_cost"]

    with (
        xr.open_dataset(
            getattr(input_profiles, "specs_egs"),
        ) as ds_specs,
        xr.open_dataset(
            getattr(input_profiles, "profile_egs"),
        ) as ds_profile,
    ):
        bus2sub = (
            pd.read_csv(input_profiles.bus2sub, dtype=str)
            .drop("interconnect", axis=1)
            .rename(columns={"Bus": "bus_id"})
        )

        # IGNORE: Remove dropna(). Rather, apply dropna when creating the original dataset
        df_specs = pd.merge(
            ds_specs.to_dataframe().reset_index().dropna(),
            bus2sub,
            on="sub_id",
            how="left",
        )
        df_specs["bus_id"] = df_specs["bus_id"].astype(str)

        # bus_id must be in index for pypsa to read it
        df_specs = df_specs.set_index("bus_id")

        # columns must be renamed to refer to the right quantities for pypsa to read it correctly
        logger.info(f"Using {drilling_cost} EGS drilling costs.")
        df_specs = df_specs.rename(
            columns={
                ("advanced_capex_usd_kw" if drilling_cost == "advanced" else "capex_usd_kw"): "capital_cost",
                "avail_capacity_mw": "p_nom_max",
                "fixed_om": "fixed_om",
            },
        )

        # TODO: come up with proper values for these params

        df_specs["capital_cost"] = 1000 * (
            df_specs["capital_cost"] * calculate_annuity(capital_recovery_period, discount_rate) + df_specs["fixed_om"]
        )  # convert and annualize USD/kW to USD/MW-year
        df_specs["efficiency"] = 1.0

        df_specs = df_specs.loc[~(df_specs.index == "nan")]

        # TODO: review what qualities need to be included. Currently limited for speedup.
        qualities = [1]  # df_specs.Quality.unique()

        for q in qualities:
            suffix = " " + car  # + f" Q{q}"
            df_q = df_specs[df_specs["Quality"] == q]

            bus_list = df_q.index.values
            capital_cost = df_q["capital_cost"]
            p_nom_max_bus = df_q["p_nom_max"]
            efficiency = df_q["efficiency"]  # for now.

            # IGNORE: Remove dropna(). Rather, apply dropna when creating the original dataset
            df_q_profile = pd.merge(
                ds_profile.sel(Quality=q).to_dataframe().dropna().reset_index(),
                bus2sub,
                on="sub_id",
                how="left",
            )
            bus_profiles = pd.pivot_table(
                df_q_profile,
                columns="bus_id",
                index=["year", "Date"],
                values="capacity_factor",
            )

            logger.info(
                f"Adding EGS (Resource Quality-{q}) capacity-factor profiles to the network.",
            )

            n.madd(
                "Generator",
                bus_list,
                suffix,
                bus=bus_list,
                carrier=car,
                p_nom_extendable=car in extendable_carriers["Generator"],
                p_nom_max=p_nom_max_bus,
                capital_cost=capital_cost,
                efficiency=efficiency,
                p_max_pu=bus_profiles,
                build_year=n.investment_periods[0],
                lifetime=capital_recovery_period,
            )


def attach_battery_storage(
    n: pypsa.Network,
    costs: pd.DataFrame,
    plants: pd.DataFrame,
):
    """Attaches Existing Battery Energy Storage Systems To the Network."""
    plants_filt = plants.query("carrier == 'battery' ")
    plants_filt.index = plants_filt.index.astype(str) + "_" + plants_filt.generator_id.astype(str)
    plants_filt.loc[:, "energy_storage_capacity_mwh"] = plants_filt.energy_storage_capacity_mwh.astype(float)
    plants_filt = plants_filt.dropna(subset=["energy_storage_capacity_mwh"])

    logger.info(
        f"Added Batteries as Storage Units to the network.\n{np.round(plants_filt.p_nom.sum() / 1000, 2)} GW Power Capacity \n{np.round(plants_filt.energy_storage_capacity_mwh.sum() / 1000, 2)} GWh Energy Capacity",
    )

    plants_filt = plants_filt.dropna(subset=["energy_storage_capacity_mwh"])
    n.madd(  # Adds storage units which can retire economically or at their lifetime
        "StorageUnit",
        plants_filt.index,
        carrier="battery",
        bus=plants_filt.bus_assignment,
        p_nom=plants_filt.p_nom,
        p_nom_max=plants_filt.p_nom,
        p_nom_min=0,
        p_nom_extendable=False,  # Only Allow lifetime retirments for existing BESS
        capital_cost=costs.at["4hr_battery_storage", "opex_fixed_per_kw"] * 1e3,
        max_hours=plants_filt.energy_storage_capacity_mwh / plants_filt.p_nom,
        build_year=plants_filt.build_year,
        lifetime=costs.at["4hr_battery_storage", "lifetime"],
        efficiency_store=0.85**0.5,
        efficiency_dispatch=0.85**0.5,
        cyclic_state_of_charge=True,
    )


def broadcast_investment_horizons_index(n: pypsa.Network, df: pd.DataFrame):
    """
    Broadcast the index of a dataframe to match the potentially multi-indexed
    investment periods of a PyPSA network.
    """
    # MultiIndex (period, timestamp)
    sns = n.snapshots

    # Work on a copy to avoid mutating caller's data
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Fast-path: already matches the full set of snapshots
    if len(df.index) == len(sns):
        df.index = sns
        return df

    # Otherwise broadcast the single-year profile to every investment period
    period_dfs: list[pd.DataFrame] = []
    for period in n.investment_periods.to_list():
        # The subset of snapshots belonging to the current period
        period_snapshots = sns[sns.get_level_values(0) == period]

        # Clone the profile and set its year to the planning horizon
        df_period = df.copy()
        df_period.index = df.index.map(lambda ts: ts.replace(year=period))

        # Align to the exact timestamps of the period.  If some hours are
        # missing (e.g. Feb-29 in a leap year) fill them by carrying forward
        # the previous hour.  This is conservative but avoids failures when
        # mixing leap and non-leap calendars.
        df_period = df_period.reindex(
            period_snapshots.get_level_values(1),
            method="ffill",
        )

        # Put the MultiIndex back so it lines up with n.snapshots
        df_period.index = period_snapshots
        period_dfs.append(df_period)

    df_broadcast = pd.concat(period_dfs)

    # Final consistency check – this should now match exactly
    assert len(df_broadcast.index) == len(sns), (
        "broadcast_investment_horizons_index: "
        "mismatch after broadcasting"
    )

    return df_broadcast


def apply_seasonal_capacity_derates(
    n: pypsa.Network,
    plants: pd.DataFrame,
    conventional_carriers: list,
    sns: pd.DatetimeIndex,
):
    """Applies conventional rerate factor p_max_pu based on the seasonal capacity derates defined in eia860."""
    sns_dt = sns.get_level_values(1)
    summer_sns = sns_dt[sns_dt.month.isin([6, 7, 8])]
    winter_sns = sns_dt[~sns_dt.month.isin([6, 7, 8])]

    # conventional_carriers = ['geothermal'] # testing override impact

    conv_plants = plants.query("carrier in @conventional_carriers")
    # keep identical generator IDs to ensure alignment with the network
    conv_gens = n.generators.query("carrier in @conventional_carriers")

    p_max_pu = pd.DataFrame(1.0, index=sns_dt, columns=conv_gens.index)
    p_max_pu.loc[summer_sns, conv_gens.index] *= conv_plants.loc[
        :,
        "summer_derate",
    ].astype(float)
    p_max_pu.loc[winter_sns, conv_gens.index] *= conv_plants.loc[
        :,
        "winter_derate",
    ].astype(float)

    p_max_pu = broadcast_investment_horizons_index(n, p_max_pu)
    n.generators_t.p_max_pu = pd.concat(
        [n.generators_t.p_max_pu, p_max_pu],
        axis=1,
    ).round(3)


def apply_must_run_ratings(
    n: pypsa.Network,
    plants: pd.DataFrame,
    conventional_carriers: list,
    sns: pd.DatetimeIndex,
):
    """Applies Minimum Loading Capacities only to WECC ADS designated Plants."""
    conv_plants = plants.query("carrier in @conventional_carriers").copy()
    # keep identical generator IDs to ensure alignment with the network

    conv_plants.loc[:, "ads_mustrun"] = conv_plants.ads_mustrun.infer_objects(
        copy=False,
    ).fillna(False)

    conv_plants.loc[:, "minimum_load_pu"] = conv_plants.minimum_load_mw / conv_plants.p_nom
    conv_plants.loc[:, "minimum_load_pu"] = (
        conv_plants.minimum_load_pu.clip(
            upper=np.minimum(conv_plants.summer_derate, conv_plants.winter_derate),
            lower=0,
        )
        .astype(float)
        .fillna(0)
    )
    must_run = conv_plants.query("ads_mustrun == True")
    n.generators.loc[must_run.index, "p_min_pu"] = must_run.minimum_load_pu.round(3) * 0.95


def clean_bus_data(n: pypsa.Network):
    """Drops data from the network that are no longer needed in workflow."""
    col_list = [
        # "Pd",
        "load_dissag",
        "LAF",
        "LAF_state",
    ]
    n.buses = n.buses.drop(columns=[col for col in col_list if col in n.buses])


def attach_breakthrough_renewable_plants(
    n,
    fn_plants,
    renewable_carriers,
    extendable_carriers,
    costs,
):
    plants = pd.read_csv(fn_plants)
    plants = plants.rename(columns={"Plant name": "name", "Technology": "carrier"})
    plants.index = plants["name"]

    # Drop any plants that are already in the network
    plants = plants[~plants.index.isin(n.generators.index)]

    for tech in renewable_carriers:
        tech_plants = plants[plants.carrier == tech].copy()

        if tech_plants.empty:
            continue

        p_nom_be = pd.read_csv(snakemake.input[f"{tech}_breakthrough"], index_col=0)

        intersection = set(p_nom_be.columns).intersection(
            tech_plants.index,
        )  # filters by plants ID for the plants of type tech
        p_nom_be = p_nom_be[list(intersection)]

        p_nom_be.columns = p_nom_be.columns.astype(str)

        if (tech_plants.Pmax == 0).any():
            # p_nom is the maximum of {Pmax, dispatch}
            p_nom = pd.concat([p_nom_be.max(axis=0), tech_plants["Pmax"]], axis=1).max(
                axis=1,
            )
            p_max_pu = (p_nom_be[p_nom.index] / p_nom).astype(float).fillna(0)  # some values remain 0
        else:
            p_nom = tech_plants.Pmax
            p_max_pu = p_nom_be[tech_plants.index] / p_nom

        leap_day = p_max_pu.loc["2016-02-29 00:00:00":"2016-02-29 23:00:00"]
        p_max_pu = p_max_pu.drop(leap_day.index)
        p_max_pu = broadcast_investment_horizons_index(n, p_max_pu)

        n.madd(
            "Generator",
            tech_plants.index,
            bus=tech_plants.bus_id,
            p_nom_min=p_nom,
            p_nom=p_nom,
            marginal_cost=0,
            p_max_pu=p_max_pu,  # timeseries of max power output pu
            p_nom_extendable=False,
            carrier=tech,
            weight=1.0,
            build_year=n.investment_periods[0],
            lifetime=np.inf,
        )
    return n


def apply_pudl_fuel_costs(
    n,
    plants,
    costs,
):
    # Apply PuDL Fuel Costs for plants where listed
    pudl_fuel_costs = pd.read_csv(snakemake.input["pudl_fuel_costs"], index_col=0)

    # Check if any of the plants are in the pudl fuel costs
    if not set(plants.index).intersection(pudl_fuel_costs.columns):
        return n

    # Construct the VOM table for each generator by carrier
    vom = pd.DataFrame(index=pudl_fuel_costs.columns)
    for gen in pudl_fuel_costs.columns:
        if gen not in plants.index:
            continue
        carrier = plants.loc[gen, "carrier"]
        if carrier not in costs.index:
            continue
        vom.loc[gen, "VOM"] = costs.at[carrier, "opex_variable_per_mwh"]

    # Apply the VOM to the fuel costs
    pudl_fuel_costs = pudl_fuel_costs + vom.squeeze()
    pudl_fuel_costs = broadcast_investment_horizons_index(n, pudl_fuel_costs)

    # Drop any columns that are not in the network
    # keep original generator IDs for alignment
    pudl_fuel_costs.columns = pudl_fuel_costs.columns
    pudl_fuel_costs = pudl_fuel_costs[[x for x in pudl_fuel_costs.columns if x in n.generators.index]]

    # drop any data that has been assigned at a coarser resolution
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"][
        [x for x in n.generators_t["marginal_cost"] if x not in pudl_fuel_costs]
    ]

    # assign new marginal costs
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(
        pudl_fuel_costs,
    )
    logger.info(
        f"Applied PuDL fuel costs to {len(pudl_fuel_costs.columns)} generators.",
    )
    return n


def main(snakemake):
    params = snakemake.params
    interconnection = snakemake.wildcards["interconnect"]

    n = pypsa.Network(snakemake.input.network)

    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
    regions_offshore = gpd.read_file(snakemake.input.regions_offshore)
    reeds_shapes = gpd.read_file(snakemake.input.reeds_shapes)

    costs = pd.read_csv(snakemake.input.tech_costs)
    costs = costs.pivot(index="pypsa-name", columns="parameter", values="value")
    update_transmission_costs(n, costs, params.length_factor)

    renewable_carriers = set(params.renewable_carriers)
    extendable_carriers = params.extendable_carriers
    conventional_carriers = params.conventional_carriers
    conventional_inputs = {k: v for k, v in snakemake.input.items() if k.startswith("conventional_")}

    plants = load_powerplants(
        snakemake.input["powerplants"],
        n.investment_periods,
        interconnect=interconnection,
    )
    plants = filter_plants_by_region(
        plants,
        regions_onshore,
        regions_offshore,
        reeds_shapes,
    )
    plants = match_plant_to_bus(n, plants)

    attach_egs(
        n,
        costs,
        snakemake.input,
        renewable_carriers,
        extendable_carriers,
        params.length_factor,
    )

    attach_conventional_generators(
        n,
        costs,
        plants,
        conventional_carriers,
        extendable_carriers,
        params.conventional,
        renewable_carriers,
        conventional_inputs,
        unit_commitment=params.conventional["unit_commitment"],
        fuel_price=None,  # update fuel prices later
    )
    apply_seasonal_capacity_derates(
        n,
        plants,
        conventional_carriers,
        n.snapshots,
    )

    if params.conventional.get("must_run", False):
        # TODO (@ktehranchi): In the future the plants that are must-run should
        # not be clustered and instead retire according to lifetime
        apply_must_run_ratings(
            n,
            plants,
            conventional_carriers,
            n.snapshots,
        )

    attach_battery_storage(
        n,
        costs,
        plants,
    )

    attach_wind_and_solar(
        n,
        costs,
        snakemake.input,
        renewable_carriers,
        extendable_carriers,
    )
    renewable_carriers = list(
        set(snakemake.config["electricity"]["renewable_carriers"]).intersection(
            {"onwind", "solar", "offwind", "offwind_floating"},
        ),
    )
    attach_renewable_capacities_to_atlite(
        n,
        plants,
        renewable_carriers,
    )

    # temporarily adding hydro with breakthrough only data until I can correctly import hydro_data
    n = attach_breakthrough_renewable_plants(
        n,
        snakemake.input["plants_breakthrough"],
        ["hydro"],
        extendable_carriers,
        costs,
    )

    update_p_nom_max(n)

    # apply regional multipliers to capital cost data
    for carrier, multiplier_data in const.CAPEX_LOCATIONAL_MULTIPLIER.items():
        if n.generators.query(f"carrier == '{carrier}'").empty:
            continue
        multiplier_file = snakemake.input[f"gen_cost_mult_{multiplier_data}"]
        df_multiplier = pd.read_csv(multiplier_file)
        df_multiplier = clean_locational_multiplier(df_multiplier)
        update_capital_costs(n, carrier, costs, df_multiplier)

    if params.conventional["dynamic_fuel_price"].get("enable", False):
        logger.info("Applying dynamic fuel pricing to conventional generators")
        if params.conventional["dynamic_fuel_price"]["wholesale"]:
            assert params.eia_api, "Must provide EIA API key for dynamic fuel pricing"

            dynamic_fuel_prices = {
                "OCGT": {
                    "state": "state_ng_fuel_prices",
                    "balancing_area": "ba_ng_fuel_prices",  # name of file in snakefile
                },
                "CCGT": {
                    "state": "state_ng_fuel_prices",
                    "balancing_area": "ba_ng_fuel_prices",
                },
                "coal": {"state": "state_coal_fuel_prices"},
            }

            # NOTE: Must go from most to least coarse data (ie. state then ba) to apply the
            # data correctly!
            for carrier, prices in dynamic_fuel_prices.items():
                for area in ("state", "reeds_zone", "balancing_area"):
                    # check if data is supplied for the area
                    try:
                        datafile = prices[area]
                    except KeyError:
                        continue
                    # if data should exist, try to read it in
                    try:
                        df = pd.read_csv(
                            snakemake.input[datafile],
                            index_col="snapshot",
                        )
                        if df.empty:
                            logger.warning(f"No data provided for {datafile}")
                            continue
                    except KeyError:
                        logger.warning(f"Can not find dynamic price file {datafile}")
                        continue

                    vom = costs.at[carrier, "opex_variable_per_mwh"]

                    apply_dynamic_pricing(
                        n=n,
                        carrier=carrier,
                        geography=area,
                        df=df,
                        vom=vom,
                    )
                    logger.info(
                        f"Applied dynamic price data for {carrier} from {datafile}",
                    )

        if params.conventional["dynamic_fuel_price"]["pudl"]:
            n = apply_pudl_fuel_costs(n, plants, costs)

    # fix p_nom_min for extendable generators
    # The "- 0.001" is just to avoid numerical issues
    n.generators["p_nom_min"] = n.generators.apply(
        lambda x: ((x["p_nom"] - 0.001) if (x["p_nom_extendable"] and x["p_nom_min"] == 0) else x["p_nom_min"]),
        axis=1,
    )

    output_folder = os.path.dirname(snakemake.output[0]) + "/base_network"
    export_network_for_gis_mapping(n, output_folder)

    clean_bus_data(n)
    sanitize_carriers(n, snakemake.config)
    n.meta = snakemake.config

    # n.export_to_netcdf(snakemake.output[0])
    pickle.dump(n, open(snakemake.output[0], "wb"))


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_electricity",
            interconnect="texas",
        )
    configure_logging(snakemake)
    main(snakemake)
