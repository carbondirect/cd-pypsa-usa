import logging
import pandas as pd
from _helpers import mock_snakemake

logger = logging.getLogger(__name__)


def apply_retrofit_penalties(
    plants: pd.DataFrame,
    retrofit_targets: list[dict],
    base_emissions: pd.Series,
) -> pd.DataFrame:
    """
    Applies penalties and emissions changes for retrofitted generator units.
    """

    # Ensure there's a column for custom emissions, copying from carrier default
    if "co2_emissions" not in plants.columns:
        plants["co2_emissions"] = plants["carrier"].map(base_emissions)

    for target in retrofit_targets:
        plant_id = target["plant_id_eia"]
        generator_ids = target["generator_ids"]
        heat_rate_penalty = target.get("heat_rate_penalty", 1.0)
        capture_rate = target.get("co2_capture_rate")

        # Create a mask for the specific generators at the specified plant
        plant_mask = (plants.plant_id_eia == plant_id) & (
            plants.generator_id.isin(generator_ids)
        )

        if not plant_mask.any():
            continue

        num_found = plant_mask.sum()
        logger.info(
            f"Applying retrofits to {num_found} generator(s) at plant {plant_id}."
        )

        # Apply heat rate penalty
        plants.loc[plant_mask, "heat_rate"] *= heat_rate_penalty
        # add  a flag for penalized plants
        plants.loc[plant_mask, "retrofit_penalized"] = True
        logger.info(f"    - Applied heat rate penalty: {heat_rate_penalty}")

        # Apply CO2 capture rate
        if capture_rate is not None:
            # get the default emission rate for the carrier of the first generator
            base_carrier = plants.loc[plant_mask].iloc[0]["carrier"]
            base_emission_rate = base_emissions.get(base_carrier, 0)

            new_emission_rate = base_emission_rate * (1 - capture_rate)
            plants.loc[plant_mask, "co2_emissions"] = new_emission_rate
            logger.info(
                f"    - Set CO2 capture rate to {capture_rate:.0%}, "
                f"new emissions: {new_emission_rate:.4f} tCO2/MWh_th"
            )

        # Update marginal cost to reflect new heat rate
        plants.loc[plant_mask, "marginal_cost"] = (
            plants.loc[plant_mask, "vom"]
            + (
                plants.loc[plant_mask, "fuel_cost"]
                * plants.loc[plant_mask, "heat_rate"]
            )
        ).round(2)

    return plants


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "apply_retrofit",
            input={
                "powerplants": "resources/powerplants.csv",
                "tech_costs": "resources/costs/costs_2030.csv",
            },
        )

    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    plants_df = pd.read_csv(snakemake.input.powerplants)
    costs = pd.read_csv(snakemake.input.tech_costs)

    # Create a series for base emissions rates per carrier
    base_emissions = costs.loc[costs.parameter == "co2_emissions"]
    base_emissions = base_emissions.set_index("pypsa-name")["value"]

    # Define the specific generators to be retrofitted.
    # We want to assume 90% capture rather than 95%
    retrofit_targets = [
        {
            "plant_id_eia": 55327,
            "generator_ids": ["CGT-1", "CGT-2"],  # check the formatting
            "heat_rate_penalty": 1.1,
            "co2_capture_rate": 0.90,
        }
    ]

    plants_df = apply_retrofit_penalties(
        plants_df, retrofit_targets, base_emissions
    )

    plants_df.to_csv(snakemake.output.powerplants, index=False) 