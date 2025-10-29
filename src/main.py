from collections.abc import Sequence
from importlib import import_module
from typing import NamedTuple, Optional, Union

from Rhino.Geometry import Brep, Curve, GeometryBase, Plane, Point3d, PolylineCurve
from Rhino.Geometry.Intersect import Intersection

TOLERANCE = 0.001


class Hat(NamedTuple):
    """Represents a hat structure with a base curve and an offsetted plane."""

    base_curve: PolylineCurve
    offsetted_plane: Plane


class GeometryInput(NamedTuple):
    """Input parameters for geometry processing."""

    shape: Brep
    piece_count: int


class GeometryOutput(NamedTuple):
    """Output containing the resulting hats and debug shapes."""

    result: Sequence[Hat]
    debug_shapes: Sequence[Sequence[Union[GeometryBase, Point3d]]]


def populate_geometry(brep: Brep, count: int, seed: int) -> list[Point3d]:
    """Populate a Brep geometry with random points."""
    module = import_module("ghpythonlib.components")
    func = getattr(module, "PopulateGeometry")
    result = func(brep, count, seed)
    return list(result)


def voronoi_3d(points: Sequence[Point3d]) -> list[Brep]:
    """Generate 3D Voronoi cells from a sequence of points."""
    module = import_module("ghpythonlib.components")
    func = getattr(module, "Voronoi3D")
    result = func(points)
    return list(getattr(result, "cells"))


def join_curves(curves: Sequence[Curve]) -> Curve:
    """Join multiple curves into a single curve."""
    joined = list(Curve.JoinCurves(curves, TOLERANCE))
    if len(joined) != 1:
        raise ValueError
    return joined[0]


class PolygonBuilder:
    """Builder class for creating polyline curves from curves."""

    def build(self, curve: Curve) -> PolylineCurve:
        """Build a closed polyline curve from a curve."""
        points = self._extract_vertices(curve)
        return self._points_to_closed_polyline_curve(points)

    def _points_to_closed_polyline_curve(
        self, points: Sequence[Point3d]
    ) -> PolylineCurve:
        """Convert a sequence of points to a closed polyline curve."""
        return PolylineCurve([*points, points[0]])

    def _extract_vertices(self, curve: Curve) -> list[Point3d]:
        """Extract vertices from a curve's segments."""
        segments = curve.DuplicateSegments()
        points: list[Point3d] = []
        if not curve.IsClosed:
            points.append(next(iter(segments)).PointAtStart)
        points.extend(segment.PointAtEnd for segment in segments)
        return points


class HatBuilder:
    """Builder class for creating Hat structures from polyline curves."""

    def build(self, curve: PolylineCurve) -> Hat:
        """Build a Hat from a polyline curve."""
        offsetted_plane = self._build_offsetted_plane(curve)
        return Hat(
            base_curve=curve,
            offsetted_plane=offsetted_plane,
        )

    def _build_offsetted_plane(self, curve: PolylineCurve) -> Plane:
        """Build an offsetted plane by fitting a plane to the curve's points."""
        points = curve.ToArray()
        return Plane.FitPlaneToPoints(points)[1]


def main():
    """Main function to generate hat structures from a Brep shape using Voronoi tessellation."""
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

    polygon_builder = PolygonBuilder()
    refined_pieces = [polygon_builder.build(piece) for piece in raw_pieces]

    hat_builder = HatBuilder()
    hats = [hat_builder.build(piece) for piece in refined_pieces]

    return GeometryOutput(
        result=hats,
        debug_shapes=[
            populated_points,
            voronoi_cells,
            raw_pieces,
            refined_pieces,
        ],
    )


if __name__ == "__main__":
    result, debug_shapes = main()
