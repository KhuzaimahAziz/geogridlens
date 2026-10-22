"""Bus and branch GeoDataFrames for the IEEE-118 benchmark.

The IEEE-118 case has electrical topology but no coordinates.  The case-study
builder supplies a coordinate per bus; this module turns the pandapower case
into point and line layers whose ``bus_id`` and ``branch_id`` match the
result tables.  Geometry is a drawing of the benchmark, never a real asset.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


# Placeholder CRS for the planar drawing; the case-study builder overrides it
# with the study area's metric CRS.
SCHEMATIC_CRS = "EPSG:3857"


def _bus_ids(net: Any) -> pd.Series:
    """Return human-facing IEEE bus numbers indexed by pandapower bus index."""

    return net.bus["name"].astype(int)


def network_graph(net: Any) -> nx.Graph:
    """Build an undirected topology graph from in-service lines and transformers."""

    graph = nx.Graph()
    graph.add_nodes_from(net.bus.index.tolist())

    for _, line in net.line.loc[net.line["in_service"]].iterrows():
        graph.add_edge(int(line["from_bus"]), int(line["to_bus"]))
    for _, trafo in net.trafo.loc[net.trafo["in_service"]].iterrows():
        graph.add_edge(int(trafo["hv_bus"]), int(trafo["lv_bus"]))

    return graph


def _bus_role_sets(net: Any) -> tuple[set[int], set[int], set[int]]:
    """Collect slack, generation, and load buses for display labels."""

    slack_buses = set(net.ext_grid["bus"].astype(int))
    generation_buses = set(net.gen["bus"].astype(int)) | set(net.sgen["bus"].astype(int))
    load_buses = set(net.load["bus"].astype(int))
    return slack_buses, generation_buses, load_buses


def build_bus_layer(net: Any, coordinates: dict[int, tuple[float, float]]) -> gpd.GeoDataFrame:
    """Build a point layer for every IEEE-118 bus."""

    bus_ids = _bus_ids(net)
    slack_buses, generation_buses, load_buses = _bus_role_sets(net)
    records: list[dict[str, Any]] = []

    for index, row in net.bus.iterrows():
        if index in slack_buses:
            role = "slack"
        elif index in generation_buses:
            role = "generator"
        elif index in load_buses:
            role = "load"
        else:
            role = "junction"

        x, y = coordinates[int(index)]
        records.append(
            {
                "bus_id": int(bus_ids.loc[index]),
                "bus_index": int(index),
                "bus_role": role,
                "base_voltage_kv": float(row["vn_kv"]),
                "x_schematic": x,
                "y_schematic": y,
                "geometry": Point(x, y),
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=SCHEMATIC_CRS)


def _line_capacity_mva(net: Any, line: pd.Series) -> float | None:
    """Estimate line thermal capacity from its pandapower current rating."""

    from_bus = int(line["from_bus"])
    values = [
        line.get("max_i_ka"),
        line.get("df", 1.0),
        line.get("parallel", 1.0),
        line.get("max_loading_percent", 100.0),
        net.bus.at[from_bus, "vn_kv"],
    ]
    if any(pd.isna(value) for value in values):
        return None
    return float(
        np.sqrt(3)
        * values[4]
        * values[0]
        * values[1]
        * values[2]
        * values[3]
        / 100
    )


def _trafo_capacity_mva(trafo: pd.Series) -> float | None:
    """Estimate transformer capacity from its nominal MVA rating."""

    values = [
        trafo.get("sn_mva"),
        trafo.get("df", 1.0),
        trafo.get("parallel", 1.0),
        trafo.get("max_loading_percent", 100.0),
    ]
    if any(pd.isna(value) for value in values):
        return None
    return float(values[0] * values[1] * values[2] * values[3] / 100)


def build_branch_layer(net: Any, coordinates: dict[int, tuple[float, float]]) -> gpd.GeoDataFrame:
    """Build line features for both lines and transformers.

    The ``branch_id`` field deliberately matches AC-result CSVs, enabling a
    direct join from scenario loading metrics to the GIS layer.
    """

    bus_ids = _bus_ids(net)
    records: list[dict[str, Any]] = []

    for index, line in net.line.iterrows():
        from_index = int(line["from_bus"])
        to_index = int(line["to_bus"])
        records.append(
            {
                "branch_id": f"line:{index}",
                "branch_type": "line",
                "from_bus": int(bus_ids.loc[from_index]),
                "to_bus": int(bus_ids.loc[to_index]),
                "in_service": bool(line["in_service"]),
                "rated_capacity_mva": _line_capacity_mva(net, line),
                "geometry": LineString([coordinates[from_index], coordinates[to_index]]),
            }
        )

    for index, trafo in net.trafo.iterrows():
        from_index = int(trafo["hv_bus"])
        to_index = int(trafo["lv_bus"])
        records.append(
            {
                "branch_id": f"trafo:{index}",
                "branch_type": "trafo",
                "from_bus": int(bus_ids.loc[from_index]),
                "to_bus": int(bus_ids.loc[to_index]),
                "in_service": bool(trafo["in_service"]),
                "rated_capacity_mva": _trafo_capacity_mva(trafo),
                "geometry": LineString([coordinates[from_index], coordinates[to_index]]),
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=SCHEMATIC_CRS)
