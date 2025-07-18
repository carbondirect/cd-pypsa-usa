"""
Plots static and interactive charts to analyze system results.

**Inputs**

A solved network

**Outputs**

Capacity maps for:
    - Base capacity
    - New capacity
    - Optimal capacity (does not show existing unused capacity)
    - Optimal browfield capacity
    - Renewable potential capacity

    .. image:: _static/plots/capacity-map.png
        :scale: 33 %

Emission charts for:
    - Emissions map by node

    .. image:: _static/plots/emissions-map.png
        :scale: 33 %
"""

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
from _helpers import configure_logging
from add_electricity import sanitize_carriers
from cartopy import crs as ccrs
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from summary import (
    get_capacity_base,
    get_capacity_brownfield,
    get_demand_base,
    get_node_emissions_timeseries,
)

logger = logging.getLogger(__name__)

# Global Plotting Settings
TITLE_SIZE = 16


def get_color_palette(n: pypsa.Network) -> pd.Series:
    """Returns colors based on nice name."""
    colors = (n.carriers.reset_index().set_index("nice_name")).color

    # additional = {
    #     "Battery Charge": n.carriers.loc["battery"].color,
    #     "Battery Discharge": n.carriers.loc["battery"].color,
    #     "battery_discharger": n.carriers.loc["battery"].color,
    #     "battery_charger": n.carriers.loc["battery"].color,
    #     "4hr_battery_storage_discharger": n.carriers.loc["4hr_battery_storage"].color,
    #     "4hr_battery_storage_charger": n.carriers.loc["4hr_battery_storage"].color,
    #     "8hr_PHS_charger": n.carriers.loc["8hr_PHS"].color,
    #     "8hr_PHS_discharger": n.carriers.loc["8hr_PHS"].color,
    #     "10hr_PHS_charger": n.carriers.loc["10hr_PHS"].color,
    #     "10hr_PHS_discharger": n.carriers.loc["10hr_PHS"].color,
    #     "co2": "k",
    # }

    # Initialize the additional dictionary
    additional = {
        "co2": "k",
    }

    # Loop through the carriers DataFrame
    for index, row in n.carriers.iterrows():
        if "battery" in index or "PHS" in index:
            color = row.color
            additional.update(
                {
                    f"{index}_charger": color,
                    f"{index}_discharger": color,
                },
            )

    return pd.concat([colors, pd.Series(additional)]).to_dict()


def get_bus_scale(interconnect: str) -> float:
    """Scales lines based on interconnect size."""
    if interconnect != "usa":
        return 1e5
    else:
        return 4e4


def get_line_scale(interconnect: str) -> float:
    """Scales lines based on interconnect size."""
    if interconnect != "usa":
        return 2e3
    else:
        return 3e3


def create_title(title: str, **wildcards) -> str:
    """
    Standardizes wildcard writing in titles.

    Arguments:
        title: str
            Title of chart to plot
        **wildcards
            any wildcards to add to title
    """
    w = []
    for wildcard, value in wildcards.items():
        if wildcard == "interconnect":
            w.append(f"interconnect = {value}")
        elif wildcard == "clusters":
            w.append(f"#clusters = {value}")
        elif wildcard == "ll":
            w.append(f"ll = {value}")
        elif wildcard == "opts":
            w.append(f"opts = {value}")
        elif wildcard == "sector":
            w.append(f"sectors = {value}")
    wildcards_joined = " | ".join(w)
    return f"{title} \n ({wildcards_joined})"


def remove_sector_buses(df: pd.DataFrame) -> pd.DataFrame:
    """Removes buses for sector coupling."""
    num_levels = df.index.nlevels

    if num_levels > 1:
        condition = (df.index.get_level_values("bus").str.endswith(" gas")) | (
            df.index.get_level_values("bus").str.endswith(" gas storage")
        )
    else:
        condition = (
            (df.index.str.endswith(" gas"))
            | (df.index.str.endswith(" gas storage"))
            | (df.index.str.endswith(" gas import"))
            | (df.index.str.endswith(" gas export"))
        )
    return df.loc[~condition].copy()


def plot_emissions_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    **wildcards,
) -> None:
    # get data

    emissions = (
        get_node_emissions_timeseries(n)
        .groupby(level=0, axis=1)  # group columns
        .sum()
        .sum()  # collaps rows
        .mul(1e-6)  # T -> MT
    )
    emissions = remove_sector_buses(emissions.T).T
    emissions.index.name = "bus"

    # plot data

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )

    bus_scale = 1

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=emissions / bus_scale,
            bus_colors="k",
            bus_alpha=0.7,
            line_widths=0,
            link_widths=0,
            ax=ax,
            margin=0.2,
            color_geomap=None,
        )

    # onshore regions
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )
    ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])

    title = create_title("Emissions (MTonne)", **wildcards)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.tight_layout()
    fig.savefig(save)
    plt.close()


def plot_capacity_map(
    n: pypsa.Network,
    bus_values: pd.DataFrame,
    line_values: pd.DataFrame,
    link_values: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    bus_scale=1,
    line_scale=1,
    title=None,
    flow=None,
    line_colors="teal",
    link_colors="green",
    line_cmap="viridis",
    line_norm=None,
) -> tuple[plt.figure, plt.axes]:
    """Generic network plotting function for capacity pie charts at each node."""
    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )

    line_width = line_values / line_scale
    link_width = link_values / line_scale

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=bus_values / bus_scale,
            bus_colors=n.carriers.color,
            bus_alpha=0.7,
            line_widths=line_width,
            link_widths=0 if link_width.empty else link_width,
            line_colors=line_colors,
            link_colors=link_colors,
            ax=ax,
            margin=0.2,
            color_geomap=True,
            flow=flow,
            line_cmap=line_cmap,
            line_norm=line_norm,
        )

    # onshore regions
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )
    ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])

    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_sizes = [5000, 10e3, 50e3]  # in MW
    line_sizes = [2000, 5000]  # in MW

    add_legend_circles(
        ax,
        [s / bus_scale for s in bus_sizes],
        [f"{s / 1000} GW" for s in bus_sizes],
        legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
    )
    add_legend_lines(
        ax,
        [s / line_scale for s in line_sizes],
        [f"{s / 1000} GW" for s in line_sizes],
        legend_kw={"bbox_to_anchor": (1, 0.8), **legend_kwargs},
    )
    add_legend_patches(
        ax,
        n.carriers.color.fillna("#000000"),
        n.carriers.nice_name,
        legend_kw={"bbox_to_anchor": (1, 0), **legend_kwargs, "loc": "lower left"},
    )
    if not title:
        ax.set_title("Capacity (MW)", fontsize=TITLE_SIZE, pad=20)
    else:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.tight_layout()

    return fig, ax


def plot_demand_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plots map of network nodal demand."""
    # get data

    bus_values = get_demand_base(n).mul(1e-3)
    line_values = n.lines.s_nom
    link_values = n.links[n.links.carrier == "AC"].p_nom.replace(to_replace={pd.NA: 0})

    # plot data
    title = create_title("Network Demand", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )

    line_width = line_values / line_scale
    link_width = link_values / line_scale

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=bus_values / bus_scale,
            # bus_colors=None,
            bus_alpha=0.7,
            line_widths=line_width,
            link_widths=0 if link_width.empty else link_width,
            line_colors="teal",
            ax=ax,
            margin=0.2,
            color_geomap=None,
        )

    # onshore regions
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )
    ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])

    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_sizes = [5000, 10e3, 50e3]  # in MW
    line_sizes = [2000, 5000]  # in MW

    add_legend_circles(
        ax,
        [s / bus_scale for s in bus_sizes],
        [f"{s / 1000} GW" for s in bus_sizes],
        legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
    )
    add_legend_lines(
        ax,
        [s / line_scale for s in line_sizes],
        [f"{s / 1000} GW" for s in line_sizes],
        legend_kw={"bbox_to_anchor": (1, 0.8), **legend_kwargs},
    )
    add_legend_patches(
        ax,
        n.carriers.color.fillna("#000000"),
        n.carriers.nice_name,
        legend_kw={"bbox_to_anchor": (1, 0), **legend_kwargs, "loc": "lower left"},
    )
    if not title:
        ax.set_title("Total Annual Demand (MW)", fontsize=TITLE_SIZE, pad=20)
    else:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.tight_layout()
    fig.savefig(save)
    plt.close()


def plot_base_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plots map of base network capacities."""
    # get data

    bus_values = get_capacity_base(n)
    bus_values = bus_values[bus_values.index.get_level_values(1).isin(carriers)]
    bus_values = remove_sector_buses(bus_values).groupby(by=["bus", "carrier"]).sum()

    line_values = n.lines.s_nom
    link_values = n.links[n.links.carrier == "AC"].p_nom.replace(to_replace={pd.NA: 0})

    # plot data

    title = create_title("Base Network Capacities", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    fig, _ = plot_capacity_map(
        n=n,
        bus_values=bus_values,
        line_values=line_values,
        link_values=link_values,
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=title,
    )
    fig.savefig(save)
    plt.close()


def plot_opt_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plots map of optimal network capacities."""
    # get data
    # capacity = n.statistics()[['Optimal Capacity']]
    # capacity = capacity[capacity.index.get_level_values(0).isin(['Generator', 'StorageUnit'])]
    # capacity.index = capacity.index.droplevel(0)

    bus_values = get_capacity_brownfield(n)
    bus_values = bus_values[bus_values.index.get_level_values("carrier").isin(carriers)]
    bus_values = remove_sector_buses(bus_values).reset_index().groupby(by=["bus", "carrier"]).sum().squeeze()
    line_values = n.lines.s_nom_opt
    link_values = n.links[n.links.carrier == "AC"].p_nom_opt.replace(
        to_replace={pd.NA: 0},
    )

    # plot data
    title = create_title("Optimal Network Capacities", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    fig, _ = plot_capacity_map(
        n=n,
        bus_values=bus_values,
        line_values=line_values,
        link_values=link_values,
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=title,
    )
    fig.savefig(save)
    plt.close()


def plot_new_capacity_map(
    n: pypsa.Network,
    n_base: pypsa.Network,
    tech_colors: dict,
    bus_size_factor: float,
    fn: str,
):
    """
    Plots the new capacity built in the network.
    """
    new_capacity = n.generators.p_nom_opt - n_base.generators.p_nom
    bus_values = new_capacity[new_capacity > 0].groupby(n.generators.bus).sum()

    if bus_values.empty:
        logger.info("No new capacity built. Skipping new capacity map.")
        # Create a placeholder file to satisfy Snakemake
        Path(fn).touch()
        return

    fig, _ = plot_capacity_map(
        n,
        bus_values=bus_values,
        line_values=pd.Series(0, index=n.lines.s_nom.index),
        link_values=pd.Series(0, index=n.links.p_nom.index),
        regions=gpd.GeoDataFrame(), # Assuming regions is not needed for new capacity plot
        bus_scale=bus_size_factor,
        line_scale=1,
        title="New Network Capacities",
        flow=None,
        line_colors="teal",
        link_colors="green",
        line_cmap="viridis",
        line_norm=None,
    )
    fig.savefig(fn)
    plt.close()


def plot_renewable_potential(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    **wildcards,
) -> None:
    """Plots wind and solar resource potential by node."""
    # get data
    renew = n.generators[
        (n.generators.p_nom_max != np.inf)
        & (n.generators.build_year == n.investment_periods[0])
        & (
            n.generators.carrier.isin(
                ["onwind", "offwind", "offwind_floating", "solar", "EGS"],
            )
        )
    ]

    bus_values = renew.groupby(["bus", "carrier"]).p_nom_max.sum()

    # do not show lines or links
    line_values = pd.Series(0, index=n.lines.s_nom.index)
    link_values = pd.Series(0, index=n.links.p_nom.index)

    # plot data
    title = create_title("Renewable Capacity Potential", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1

    bus_scale *= 15  # since potential capacity is so big

    fig, ax = plot_capacity_map(
        n=n,
        bus_values=bus_values,
        line_values=line_values,
        link_values=link_values,
        regions=regions,
        bus_scale=bus_scale,
        title=title,
    )

    # only show renewables in legend
    fig.artists[-2].remove()  # remove line width legend
    fig.artists[-1].remove()  # remove existing colour legend
    renew_carriers = n.carriers[n.carriers.index.isin(["onwind", "offwind", "offwind_floating", "solar", "EGS"])]
    add_legend_patches(
        ax,
        renew_carriers.color,
        renew_carriers.nice_name,
        legend_kw={"bbox_to_anchor": (1, 0), "frameon": False, "loc": "lower left"},
    )

    fig.savefig(save)
    plt.close()


def plot_lmp_map(network: pypsa.Network, save: str, **wildcards):
    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(8, 8),
    )

    lmps = n.buses_t.marginal_price.mean()

    plt.hexbin(
        network.buses.x,
        network.buses.y,
        gridsize=40,
        C=lmps,
        cmap=plt.cm.bwr,
        zorder=3,
    )
    network.plot(ax=ax, line_widths=pd.Series(0.5, network.lines.index), bus_sizes=0)

    cb = plt.colorbar(
        location="bottom",
        pad=0.01,
    )  # Adjust the pad value to move the color bar closer
    cb.set_label("LMP ($/MWh)")
    plt.title(create_title("Locational Marginal Price [$/MWh]", **wildcards))
    plt.tight_layout(
        rect=[0, 0, 1, 0.95],
    )  # Adjust the rect values to make the layout tighter
    plt.savefig(save)
    plt.close()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_network_maps",
            interconnect="western",
            clusters="4m",
            simpl="70",
            ll="v1.0",
            opts="1h-TCT",
            sector="E",
        )
    configure_logging(snakemake)

    # extract shared plotting files
    n = pypsa.Network(snakemake.input.network)
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore)

    sanitize_carriers(n, snakemake.config)

    # mappers
    generating_link_carrier_map = {"fuel cell": "H2", "battery discharger": "battery"}

    # carriers to plot
    carriers = (
        snakemake.params.electricity["conventional_carriers"]
        + snakemake.params.electricity["renewable_carriers"]
        + snakemake.params.electricity["extendable_carriers"]["Generator"]
        + snakemake.params.electricity["extendable_carriers"]["StorageUnit"]
        + snakemake.params.electricity["extendable_carriers"]["Store"]
        + snakemake.params.electricity["extendable_carriers"]["Link"]
    )
    carriers = list(set(carriers))  # remove any duplicates

    # plotting theme
    sns.set_theme("paper", style="darkgrid")

    # create plots
    plot_base_capacity_map(
        n,
        onshore_regions,
        carriers,
        snakemake.output["capacity_map_base.pdf"],
        **snakemake.wildcards,
    )
    plot_opt_capacity_map(
        n,
        onshore_regions,
        carriers,
        **snakemake.wildcards,
        save=snakemake.output["capacity_map_optimized.pdf"],
    )
    plot_new_capacity_map(
        n,
        n, # This is a placeholder for n_base, which is not directly available here.
        {}, # This is a placeholder for tech_colors, which is not directly available here.
        1, # This is a placeholder for bus_size_factor, which is not directly available here.
        snakemake.output["capacity_map_new.pdf"],
    )
    plot_demand_map(
        n,
        onshore_regions,
        carriers,
        snakemake.output["demand_map.pdf"],
        **snakemake.wildcards,
    )
    plot_emissions_map(
        n,
        onshore_regions,
        snakemake.output["emissions_map.pdf"],
        **snakemake.wildcards,
    )
    plot_renewable_potential(
        n,
        onshore_regions,
        snakemake.output["renewable_potential_map.pdf"],
        **snakemake.wildcards,
    )
    # plot_lmp_map(n, snakemake.output["lmp_map.pdf"], **snakemake.wildcards)
