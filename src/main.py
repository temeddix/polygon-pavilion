import math
from collections.abc import Sequence
from importlib import import_module
from typing import Any, NamedTuple, Protocol, TypeVar, Union

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

T = TypeVar("T")

TOLERANCE = 0.001
OFFSET_DISTANCE_FACTOR = 0.08


class InvalidInputError(Exception):
    """Exception raised for invalid input types."""

    def __init__(self, expected_type: type[Any], actual_value: Any) -> None:
        message = f"Invalid input value: expected {expected_type}, got {actual_value}"
        super().__init__(message)


class UnexpectedShapeError(Exception):
    """Exception raised for unexpected geometry shapes."""

    def __init__(
        self,
        geometry: Sequence[Union[GeometryBase, Plane, Line]],
    ) -> None:
        message = f"Unexpected geometry {geometry}"
        super().__init__(message)


class Hat(NamedTuple):
    """
    Represents a hat structure
    with a base curve, an offsetted plane, and a top curve.
    """

    base_curve: PolylineCurve
    offsetted_plane: Plane
    top_curve: PolylineCurve


class GeometryInput(NamedTuple):
    """Input parameters for geometry processing."""

    smooth_surface: Brep
    piece_count: int
    seed: int


class GeometryOutput(NamedTuple):
    """Output containing the resulting hats and debug shapes."""

    result: Sequence[Hat]
    intermediates: Sequence[Sequence[Union[GeometryBase, Point3d, Plane]]]


class GeometryBuilder(Protocol):
    """Protocol for geometry build step worker classes."""

    def build(self) -> Any:
        """Do the work of this geometry build step."""
        ...

    def get_intermediates(self) -> list[Union[GeometryBase, Point3d, Plane]]:
        """Get intermediate geometries from this step for debugging."""
        ...


class SurfaceSplitter:
    """Builder class for splitting smooth surfaces into pieces."""

    def __init__(self, smooth_surface: Brep, piece_count: int, seed: int) -> None:
        self._smooth_surface = smooth_surface
        self._piece_count = piece_count
        self._seed = seed
        self._populated_points: list[Point3d] = []
        self._voronoi_cells: list[Brep] = []

    def build(self) -> list[Curve]:
        surface = self._smooth_surface
        self._populated_points = self._populate_geometry(surface)
        self._voronoi_cells = self._voronoi_3d(self._populated_points)
        raw_pieces = [
            self._join_curves(list(Intersection.BrepBrep(cell, surface, TOLERANCE)[1]))
            for cell in self._voronoi_cells
        ]
        return raw_pieces

    def get_intermediates(self) -> list[Union[GeometryBase, Point3d, Plane]]:
        """Get intermediate debug geometries from this step."""
        result: list[Union[GeometryBase, Point3d, Plane]] = []
        result.extend(self._populated_points)
        result.extend(self._voronoi_cells)
        return result

    def _populate_geometry(self, brep: Brep) -> list[Point3d]:
        """Populates a Brep geometry with random points."""
        module = import_module("ghpythonlib.components")
        func = getattr(module, "PopulateGeometry")
        result = func(brep, self._piece_count, self._seed)
        return list(result)

    def _voronoi_3d(self, points: Sequence[Point3d]) -> list[Brep]:
        """Generates 3D Voronoi cells from a sequence of points."""
        module = import_module("ghpythonlib.components")
        func = getattr(module, "Voronoi3D")
        result = func(points)
        return list(getattr(result, "cells"))

    def _join_curves(self, curves: Sequence[Curve]) -> Curve:
        """Joins multiple curves into a single curve."""
        joined = list(Curve.JoinCurves(curves, TOLERANCE))
        if len(joined) != 1:
            raise UnexpectedShapeError(curves)
        return joined[0]


class PolygonBuilder:
    """Builder class for creating polyline curves from curves."""

    def __init__(self, raw_pieces: Sequence[Curve]) -> None:
        self._raw_pieces = raw_pieces
        self._refined_pieces: list[PolylineCurve] = []

    def build(self) -> list[PolylineCurve]:
        """Builds closed polyline curves from raw curves."""
        return [self._build_polygon(p) for p in self._raw_pieces]

    def _build_polygon(self, curve: Curve) -> PolylineCurve:
        """Builds a closed polyline curve from a curve."""
        points = self._extract_vertices(curve)
        polyline = self._points_to_closed_polyline_curve(points)
        self._refined_pieces.append(polyline)
        return polyline

    def get_intermediates(self) -> list[Union[GeometryBase, Point3d, Plane]]:
        """Get intermediate debug geometries from this step."""
        return list(self._refined_pieces)

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

    def __init__(
        self, original_shape: Brep, refined_pieces: Sequence[PolylineCurve]
    ) -> None:
        self._original_shape = original_shape
        self._refined_pieces = refined_pieces
        self._hat_previews: list[Union[Curve, Plane]] = []

    def build(self) -> list[Hat]:
        """Builds Hats from a sequence of polyline curves."""
        return [self._build_hat(p) for p in self._refined_pieces]

    def _build_hat(self, curve: PolylineCurve) -> Hat:
        """Builds a Hat from a polyline curve."""
        offsetted_plane = self._build_offsetted_plane(curve)
        top_curve = self._build_top_curve(curve, offsetted_plane)

        # Store intermediates for debugging
        self._hat_previews.append(curve)
        self._hat_previews.append(offsetted_plane)
        self._hat_previews.append(top_curve)

        return Hat(
            base_curve=curve,
            offsetted_plane=offsetted_plane,
            top_curve=top_curve,
        )

    def get_intermediates(self) -> list[Union[GeometryBase, Point3d, Plane]]:
        """Get intermediate debug geometries from this step."""
        return list(self._hat_previews)

    def _extract_vertices(self, curve: PolylineCurve) -> list[Point3d]:
        """Extracts vertices from a curve's segments."""
        if not curve.IsClosed:
            raise UnexpectedShapeError([curve])
        return list(curve.ToArray())[:-1]  # Exclude duplicate closing point

    def _build_offsetted_plane(self, curve: PolylineCurve) -> Plane:
        """
        Builds an offsetted plane
        by fitting a plane to the curve's points.
        """
        points = self._extract_vertices(curve)
        center = self._calculate_center(points)
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
            raise UnexpectedShapeError([plane, self._original_shape])

        # Check if the normals are opposite (dot product < 0)
        return Vector3d.Multiply(plane.ZAxis, shape_normal) < 0

    def _build_top_curve(
        self, base_curve: PolylineCurve, offsetted_plane: Plane
    ) -> PolylineCurve:
        """
        Builds the top curve on the offsetted plane by:
        1. Extruding each boundary segment to the offsetted plane's Z-axis
        2. Rotating each rectangle 60° toward the center
        3. Intersecting with the offsetted plane
        4. Creating a closed polyline from the intersection points

        Returns a tuple of (top_curve, debug_rectangles).
        """
        base_points = self._extract_vertices(base_curve)
        center = self._calculate_center(base_points)

        intersection_lines: list[Line] = []

        for i in range(len(base_points)):
            p1 = base_points[i]
            p2 = base_points[(i + 1) % len(base_points)]

            # Create a rectangle by extruding the segment to the offsetted plane
            intersection_line = self._create_and_rotate_rectangle(
                p1, p2, center, offsetted_plane
            )
            intersection_lines.append(intersection_line)

        # Build top curve by finding intersection points between adjacent lines
        if len(intersection_lines) < 3:
            raise UnexpectedShapeError(intersection_lines)

        top_points: list[Point3d] = []

        for i in range(len(intersection_lines)):
            current_line = intersection_lines[i]
            next_line = intersection_lines[(i + 1) % len(intersection_lines)]

            # Find where current line meets the next line
            intersection_point = self._find_line_intersection_point(
                current_line, next_line, offsetted_plane
            )
            top_points.append(intersection_point)

        # Create closed polyline
        return PolylineCurve([*top_points, top_points[0]])

    def _calculate_center(self, points: Sequence[Point3d]) -> Point3d:
        """Calculates the center point of a sequence of points."""
        return Point3d(
            sum(p.X for p in points) / len(points),
            sum(p.Y for p in points) / len(points),
            sum(p.Z for p in points) / len(points),
        )

    def _find_line_intersection_point(
        self, line1: Line, line2: Line, plane: Plane
    ) -> Point3d:
        """
        Finds the intersection point between two lines on a plane.
        Projects line1's end point onto line2 to find where they meet.
        """
        # Find closest points between the two lines
        success, t1, _ = Intersection.LineLine(line1, line2, TOLERANCE, True)

        if success:
            # Lines intersect - return the point on line1
            return line1.PointAt(t1)

        # Lines don't intersect - project line1's end onto line2
        line2_param = line2.ClosestParameter(line1.To)
        projected = line2.PointAt(line2_param)

        # Project back onto the plane to ensure it's on the plane
        return plane.ClosestPoint(projected)

    def _create_and_rotate_rectangle(
        self,
        p1: Point3d,
        p2: Point3d,
        center: Point3d,
        offsetted_plane: Plane,
    ) -> Line:
        """
        Projects segment endpoints along a 60-degree rotated direction
        to find their intersection with the offsetted plane.
        """
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

        # Start with the plane's normal direction
        plane_normal = Vector3d(offsetted_plane.ZAxis)
        plane_normal.Unitize()

        # Apply rotation to get the projection direction
        projection_direction = Vector3d(plane_normal)
        projection_direction.Transform(rotation_transform)

        # Project p1 and p2 to the offsetted plane along the rotated direction
        top_p1 = self._project_point_to_plane(p1, projection_direction, offsetted_plane)
        top_p2 = self._project_point_to_plane(p2, projection_direction, offsetted_plane)

        # Create the intersection line
        intersection_line = Line(top_p1, top_p2)

        return intersection_line

    def _project_point_to_plane(
        self, point: Point3d, direction: Vector3d, plane: Plane
    ) -> Point3d:
        """
        Projects a point along a direction vector to intersect with a plane.
        """
        # Create a ray from the point in the given direction
        ray = Line(point, direction)

        # Find intersection parameter with the plane
        success, t = Intersection.LinePlane(ray, plane)

        if not success:
            raise UnexpectedShapeError([ray, plane])

        # Return the point at the intersection
        return ray.PointAt(t)


def ensure_type(obj: Any, expected_type: type[T]) -> T:
    """
    Type checks and guarantees return type as T.
    Raises error if obj is not of expected_type.
    """
    if not isinstance(obj, expected_type):
        raise InvalidInputError(expected_type, obj)
    return obj


def main(geo_input: GeometryInput) -> GeometryOutput:
    """
    Main function to generate pavilion
    from a smooth Brep shape using Voronoi tessellation.
    """
    shape, piece_count, seed = geo_input

    surface_splitter = SurfaceSplitter(shape, piece_count, seed)
    raw_pieces = surface_splitter.build()

    polygon_builder = PolygonBuilder(raw_pieces)
    refined_pieces = polygon_builder.build()

    hat_builder = HatBuilder(shape, refined_pieces)
    hats = hat_builder.build()

    geo_builders: Sequence[GeometryBuilder] = [
        surface_splitter,
        polygon_builder,
        hat_builder,
    ]

    return GeometryOutput(
        result=hats,
        intermediates=[b.get_intermediates() for b in geo_builders],
    )


if __name__ == "__main__":
    geo_input = GeometryInput(
        smooth_surface=ensure_type(globals()["smooth_surface"], Brep),
        piece_count=ensure_type(globals()["piece_count"], int),
        seed=ensure_type(globals()["seed"], int),
    )
    result, debug_shapes = main(geo_input)
