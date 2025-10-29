import math
from collections.abc import Sequence
from importlib import import_module
from typing import NamedTuple, Optional, Union

from Rhino.Geometry import (
    Brep,
    Curve,
    GeometryBase,
    Line,
    Plane,
    Point3d,
    PolylineCurve,
    Transform,
    Vector3d,
)
from Rhino.Geometry.Intersect import Intersection

TOLERANCE = 0.001
OFFSET_DISTANCE_FACTOR = 0.08


class Hat(NamedTuple):
    """
    Represents a hat structure
    with a base curve, an offsetted plane, and a top curve.
    """

    base_curve: PolylineCurve
    offsetted_plane: Plane
    top_curve: PolylineCurve
    debug_rectangles: Sequence[Curve]


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
        top_curve, debug_rectangles = self._build_top_curve(curve, offsetted_plane)
        return Hat(
            base_curve=curve,
            offsetted_plane=offsetted_plane,
            top_curve=top_curve,
            debug_rectangles=debug_rectangles,
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

    def _build_top_curve(
        self, base_curve: PolylineCurve, offsetted_plane: Plane
    ) -> tuple[PolylineCurve, list[Curve]]:
        """
        Builds the top curve on the offsetted plane by:
        1. Extruding each boundary segment to the offsetted plane's Z-axis
        2. Rotating each rectangle 60° toward the center
        3. Intersecting with the offsetted plane
        4. Creating a closed polyline from the intersection points

        Returns a tuple of (top_curve, debug_rectangles).
        """
        base_points = list(base_curve.ToArray())[:-1]  # Exclude duplicate closing point
        center = self._calculate_center(base_points)

        top_points: list[Point3d] = []
        debug_rectangles: list[Curve] = []

        for i in range(len(base_points)):
            p1 = base_points[i]
            p2 = base_points[(i + 1) % len(base_points)]

            # Create a rectangle by extruding the segment to the offsetted plane
            intersection_line, rectangle = self._create_and_rotate_rectangle(
                p1, p2, center, offsetted_plane
            )

            if intersection_line is not None:
                # Add both endpoints of the intersection line
                top_points.append(intersection_line.From)
                top_points.append(intersection_line.To)

            if rectangle is not None:
                debug_rectangles.append(rectangle)

        # Create closed polyline from intersection points
        if len(top_points) > 0:
            return PolylineCurve([*top_points, top_points[0]]), debug_rectangles
        else:
            # Fallback: return a small polyline at the plane origin
            return PolylineCurve(
                [offsetted_plane.Origin, offsetted_plane.Origin]
            ), debug_rectangles

    def _calculate_center(self, points: Sequence[Point3d]) -> Point3d:
        """Calculates the center point of a sequence of points."""
        return Point3d(
            sum(p.X for p in points) / len(points),
            sum(p.Y for p in points) / len(points),
            sum(p.Z for p in points) / len(points),
        )

    def _create_and_rotate_rectangle(
        self,
        p1: Point3d,
        p2: Point3d,
        center: Point3d,
        offsetted_plane: Plane,
    ) -> tuple[Optional[Line], Optional[PolylineCurve]]:
        """
        Creates a rectangle from a segment, rotates it toward center,
        and returns the intersection line with the offsetted plane.

        Returns a tuple of (intersection_line, rectangle_curve).
        """
        from Rhino.Geometry import NurbsSurface

        # Calculate segment midpoint
        segment_mid = Point3d((p1.X + p2.X) / 2, (p1.Y + p2.Y) / 2, (p1.Z + p2.Z) / 2)

        # Create the segment line for rotation axis
        segment_line = Line(p1, p2)
        rotation_axis = segment_line.Direction

        # Calculate vector from segment to center (for rotation direction)
        segment_to_center = Vector3d(center - segment_mid)

        # Determine rotation direction by checking cross product with plane normal
        cross = Vector3d.CrossProduct(rotation_axis, segment_to_center)
        dot_with_normal = Vector3d.Multiply(cross, offsetted_plane.ZAxis)

        # Rotate 60 degrees (to get 30 degree angle with plane)
        # Flip the angle sign to rotate inward instead of outward
        if dot_with_normal < 0:
            angle = math.radians(60)
        else:
            angle = -math.radians(60)

        # Create rotation transform around the segment
        rotation_transform = Transform.Rotation(angle, rotation_axis, segment_mid)

        # Calculate the vector pointing from segment to offsetted plane
        # We'll use the plane's Z-axis direction
        plane_normal = offsetted_plane.ZAxis

        # Calculate a point on the offsetted plane above the segment
        # Project the plane origin direction from segment midpoint
        to_plane = offsetted_plane.Origin - segment_mid
        distance_to_plane = Vector3d.Multiply(to_plane, plane_normal)

        # Create top edge of rectangle (before rotation)
        # The rectangle extends from the segment upward to the plane
        height = abs(distance_to_plane) / math.cos(math.radians(60))  # Adjust for angle

        # Create two points above the segment endpoints
        up_vector = Vector3d(plane_normal)
        up_vector.Unitize()
        top_p1_orig = p1 + up_vector * height
        top_p2_orig = p2 + up_vector * height

        # Apply rotation to the top edge
        top_p1 = Point3d(top_p1_orig)
        top_p2 = Point3d(top_p2_orig)
        top_p1.Transform(rotation_transform)
        top_p2.Transform(rotation_transform)

        # Create the rectangle surface
        rect_surface = NurbsSurface.CreateFromCorners(p1, p2, top_p2, top_p1)

        # Create the rectangle polyline for debugging
        rectangle = PolylineCurve([p1, p2, top_p2, top_p1, p1])

        # Convert rectangle surface to Brep for intersection
        rect_brep = rect_surface.ToBrep()

        # Create a large plane surface for intersection
        plane_size = 10000.0
        plane_corners = [
            offsetted_plane.Origin
            + offsetted_plane.XAxis * plane_size
            + offsetted_plane.YAxis * plane_size,
            offsetted_plane.Origin
            - offsetted_plane.XAxis * plane_size
            + offsetted_plane.YAxis * plane_size,
            offsetted_plane.Origin
            - offsetted_plane.XAxis * plane_size
            - offsetted_plane.YAxis * plane_size,
            offsetted_plane.Origin
            + offsetted_plane.XAxis * plane_size
            - offsetted_plane.YAxis * plane_size,
        ]
        plane_surface_nurbs = NurbsSurface.CreateFromCorners(*plane_corners)
        plane_brep = plane_surface_nurbs.ToBrep()

        # Intersect the rectangle with the plane
        if rect_brep and plane_brep:
            intersection_result = Intersection.BrepBrep(
                rect_brep, plane_brep, TOLERANCE
            )
            intersection_curves = intersection_result[1]

            if intersection_curves:
                # Convert to list to check if there are any curves
                curve_list = list(intersection_curves)
                if len(curve_list) > 0:
                    # Get the first intersection curve
                    intersection_curve = curve_list[0]
                    # Convert to line if possible
                    line_start = intersection_curve.PointAtStart
                    line_end = intersection_curve.PointAtEnd
                    return Line(line_start, line_end), rectangle

        # Fallback: project both endpoints onto the plane
        proj_p1 = offsetted_plane.ClosestPoint(top_p1)
        proj_p2 = offsetted_plane.ClosestPoint(top_p2)
        return Line(proj_p1, proj_p2), rectangle


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

    hat_previews: list[Union[Curve, Plane]] = []

    for hat in hats:
        hat_previews.append(hat.base_curve)
        hat_previews.append(hat.offsetted_plane)
        hat_previews.extend(hat.debug_rectangles)
        hat_previews.append(hat.top_curve)

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
