from collections.abc import Sequence
from importlib import import_module
from typing import NamedTuple, Optional, Union

from Rhino.Geometry import Brep, Curve, GeometryBase, Point3d
from Rhino.Geometry.Intersect import Intersection

TOLERANCE = 0.01


class GeometryInput(NamedTuple):
    shape: Brep
    piece_count: int


class GeometryOutput(NamedTuple):
    result: Sequence[Curve]
    debug_shapes: Sequence[Sequence[Union[GeometryBase, Point3d]]]


def populate_geometry(brep: Brep, count: int, seed: int) -> list[Point3d]:
    module = import_module("ghpythonlib.components")
    func = getattr(module, "PopulateGeometry")
    result = func(brep, count, seed)
    return list(result)


def voronoi_3d(points: Sequence[Point3d]) -> list[Brep]:
    module = import_module("ghpythonlib.components")
    func = getattr(module, "Voronoi3D")
    result = func(points)
    return list(getattr(result, "cells"))


def join_curves(curves: Sequence[Curve]) -> Curve:
    joined = list(Curve.JoinCurves(curves, TOLERANCE))
    if len(joined) != 1:
        raise ValueError
    return joined[0]


def points_to_closed_curve(points: Sequence[Point3d]) -> Curve:
    module = import_module("ghpythonlib.components")
    func = getattr(module, "PointsToClosedCurve")
    result = func(points)
    return result


def main():
    shape: Optional[Brep] = globals()["shape"]
    if not isinstance(shape, Brep):
        raise ValueError

    piece_count: Optional[int] = globals()["piece_count"]
    if not isinstance(piece_count, int):
        raise ValueError

    populated_points = populate_geometry(shape, piece_count, 1)
    voronoi_cells = voronoi_3d(populated_points)
    raw_pieces = [
        join_curves(list(Intersection.BrepBrep(cell, shape, TOLERANCE)[1]))
        for cell in voronoi_cells
    ]

    return GeometryOutput(
        result=raw_pieces,
        debug_shapes=[
            populated_points,
            voronoi_cells,
        ],
    )


if __name__ == "__main__":
    result, debug_shapes = main()
