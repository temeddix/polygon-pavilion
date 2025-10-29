from collections.abc import Sequence
from importlib import import_module
from typing import NamedTuple, Optional, Union

from Rhino.Geometry import (
    Brep,
    Curve,
    GeometryBase,
    Plane,
    Point3d,
    PolylineCurve,
    Vector3d,
)
from Rhino.Geometry.Intersect import Intersection

TOLERANCE = 0.001
OFFSET_DISTANCE_FACTOR = 0.15


class Hat(NamedTuple):
    """
    Represents a hat structure
    with a base curve and an offsetted plane.
    """

    base_curve: PolylineCurve
    offsetted_plane: Plane


class GeometryInput(NamedTuple):
    """Input parameters for geometry processing."""

    shape: Brep
    piece_count: int


class GeometryOutput(NamedTuple):
    """Output containing the resulting hats and debug shapes."""

    result: Sequence[Hat]
    debug_shapes: Sequence[Sequence[Union[GeometryBase, Point3d, Plane]]]


def populate_geometry(brep: Brep, count: int, seed: int) -> list[Point3d]:
    """Populates a Brep geometry with random points."""
    module = import_module("ghpythonlib.components")
    func = getattr(module, "PopulateGeometry")
    result = func(brep, count, seed)
    return list(result)


def voronoi_3d(points: Sequence[Point3d]) -> list[Brep]:
    """Generates 3D Voronoi cells from a sequence of points."""
    module = import_module("ghpythonlib.components")
    func = getattr(module, "Voronoi3D")
    result = func(points)
    return list(getattr(result, "cells"))


def join_curves(curves: Sequence[Curve]) -> Curve:
    """Joins multiple curves into a single curve."""
    joined = list(Curve.JoinCurves(curves, TOLERANCE))
    if len(joined) != 1:
        raise ValueError
    return joined[0]


class PolygonBuilder:
    """Builder class for creating polyline curves from curves."""

    def build(self, curve: Curve) -> PolylineCurve:
        """Builds a closed polyline curve from a curve."""
        points = self._extract_vertices(curve)
        return self._points_to_closed_polyline_curve(points)

    def _points_to_closed_polyline_curve(
        self, points: Sequence[Point3d]
    ) -> PolylineCurve:
        """Converts a sequence of points to a closed polyline curve."""
        return PolylineCurve([*points, points[0]])

    def _extract_vertices(self, curve: Curve) -> list[Point3d]:
        """Extracts vertices from a curve's segments."""
        segments = curve.DuplicateSegments()
        points: list[Point3d] = []
        if not curve.IsClosed:
            points.append(next(iter(segments)).PointAtStart)
        points.extend(segment.PointAtEnd for segment in segments)
        return points


class HatBuilder:
    """Builder class for creating Hat structures from polyline curves."""

    def __init__(self, original_shape: Brep) -> None:
        self._original_shape = original_shape

    def build(self, curve: PolylineCurve) -> Hat:
        """Builds a Hat from a polyline curve."""
        offsetted_plane = self._build_offsetted_plane(curve)
        return Hat(
            base_curve=curve,
            offsetted_plane=offsetted_plane,
        )

    def _build_offsetted_plane(self, curve: PolylineCurve) -> Plane:
        """
        Builds an offsetted plane
        by fitting a plane to the curve's points.
        """
        points = list(curve.ToArray())
        center = Point3d(
            sum(p.X for p in points) / len(points),
            sum(p.Y for p in points) / len(points),
            sum(p.Z for p in points) / len(points),
        )
        plane = Plane.FitPlaneToPoints(points)[1]
        plane.Origin = center

        if self._is_flipped(plane):
            plane.Flip()

        # Calculate the diameter of the polyline curve
        diameter = self._calculate_diameter(points)

        # Offset the plane along its local Z axis by 15% of the diameter
        offset_distance = diameter * OFFSET_DISTANCE_FACTOR
        plane.Origin = plane.Origin + plane.ZAxis * offset_distance

        return plane

    def _calculate_diameter(self, points: Sequence[Point3d]) -> float:
        """
        Calculates the diameter of the polyline curve
        as the maximum distance between any two points.
        """
        max_distance = 0.0
        for i, p1 in enumerate(points):
            for p2 in points[i + 1 :]:
                distance = p1.DistanceTo(p2)
                if distance > max_distance:
                    max_distance = distance
        return max_distance

    def _is_flipped(self, plane: Plane) -> bool:
        """
        Checks if the plane normal is opposite to the original shape's normal.
        Returns True if the plane should be flipped.
        """
        # Find the closest point on the original shape from the plane origin
        closest_point = self._original_shape.ClosestPoint(plane.Origin)

        # Get the normal at the closest point on the original shape
        # Find the face that contains the closest point
        shape_normal = None
        for face in self._original_shape.Faces:
            success, u, v = face.ClosestPoint(closest_point)
            if success:
                shape_normal = face.NormalAt(u, v)
                break

        if shape_normal is None:
            raise ValueError

        # Check if the normals are opposite (dot product < 0)
        return Vector3d.Multiply(plane.ZAxis, shape_normal) < 0


def main():
    """
    Main function to generate pavilion
    from a smooth Brep shape using Voronoi tessellation.
    """
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

    hat_builder = HatBuilder(shape)
    hats = [hat_builder.build(piece) for piece in refined_pieces]

    hat_previews: list[Union[PolylineCurve, Plane]] = []
    for hat in hats:
        hat_previews.append(hat.base_curve)
        hat_previews.append(hat.offsetted_plane)

    return GeometryOutput(
        result=hats,
        debug_shapes=[
            populated_points,
            voronoi_cells,
            raw_pieces,
            hat_previews,
        ],
    )


if __name__ == "__main__":
    result, debug_shapes = main()
