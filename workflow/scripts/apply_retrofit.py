import logging
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore
from _helpers import mock_snakemake

if TYPE_CHECKING:
    from snakemake.script import Snakemake  # type: ignore

logger = logging.getLogger(__name__)


def apply_retrofit_penalties(
    plants: pd.DataFrame,
    retrofit_targets: list[dict],
    base_emissions: pd.Series,
) -> pd.DataFrame:
    """
    Applies penalties and emissions changes for retrofitted generator units.
    Also marks retrofitted generators to prevent clustering.
    """

    # Copy carrier default emissions if custom emissions column doesn't exist
    if "co2_emissions" not in plants.columns:
        plants["co2_emissions"] = plants["carrier"].map(base_emissions)

    # Add column to mark retrofitted plants
    if "land_region" not in plants.columns:
        plants["land_region"] = None

    for target in retrofit_targets:
        plant_id = target["plant_id_eia"]
        generator_ids = target["generator_ids"]
        heat_rate_penalty = target.get("heat_rate_penalty", 1.0)
        capture_rate = target.get("co2_capture_rate")

        # Create a mask for the specific generators at the specified plant
        plant_mask = (
            (plants.plant_id_eia == plant_id)
            & (plants.generator_id.isin(generator_ids))
        )

        if not plant_mask.any():
            continue

        num_found = plant_mask.sum()
        msg = (
            f"Applying retrofits to {num_found} generator(s) "
            f"at plant {plant_id}."
        )
        logger.info(msg)

        # Apply heat rate penalty
        plants.loc[plant_mask, "heat_rate"] *= heat_rate_penalty
        # add  a flag for penalized plants
        plants.loc[plant_mask, "retrofit_penalized"] = True
        # mark retrofitted plants to exclude from clustering
        plants.loc[plant_mask, "land_region"] = plants.loc[plant_mask, "bus"]
        logger.info(f"    - Applied heat rate penalty: {heat_rate_penalty}")

        if capture_rate is not None:
            # Apply CO2 capture rate
            plants.loc[plant_mask, "co2_emissions"] *= 1 - capture_rate
            logger.info(f"    - Applied CO2 capture rate: {capture_rate}")

    return plants


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake_ = mock_snakemake(
            "apply_retrofit",
            input={
                "powerplants": "resources/powerplants.csv",
                "tech_costs": "resources/costs/costs_2030.csv",
            },
        )
    else:
        snakemake_ = globals()["snakemake"]  # type: ignore

    # Basic logging setup
    fmt = "%(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    plants_df = pd.read_csv(snakemake_.input.powerplants)
    costs = pd.read_csv(snakemake_.input.tech_costs)

    # Create a series for base emissions rates per carrier
    base_emissions = costs.loc[costs.parameter == "co2_emissions"]
    base_emissions = base_emissions.set_index("pypsa-name")["value"]

    # Get retrofit targets from config
    retrofit_cfg = snakemake_.config.get("retrofits", {})
    retrofit_targets = retrofit_cfg.get("targets", [])

    plants_df = apply_retrofit_penalties(
        plants_df, retrofit_targets, base_emissions
    )

    plants_df.to_csv(snakemake_.output.powerplants, index=False) 