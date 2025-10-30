import math
from collections.abc import Sequence
from importlib import import_module
from typing import Any, NamedTuple, Optional, Protocol, TypeVar, Union

from Rhino.Geometry import (
    AreaMassProperties,
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


def populate_geometry(brep: Brep, piece_count: int, seed: int) -> list[Point3d]:
    """Populates a Brep geometry with random points."""
    module = import_module("ghpythonlib.components")
    func = getattr(module, "PopulateGeometry")
    result = func(brep, piece_count, seed)
    return list(result)


def voronoi_3d(points: Sequence[Point3d]) -> list[Brep]:
    """Generates 3D Voronoi cells from a sequence of points."""
    module = import_module("ghpythonlib.components")
    func = getattr(module, "Voronoi3D")
    result = func(points)
    return list(getattr(result, "cells"))


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
        self.preview = geometry
        message = f"Unexpected geometry {geometry}"
        super().__init__(message)


class Hat(NamedTuple):
    """
    Represents a hat structure
    with a base curve, an offsetted plane, a top curve, and surfaces.
    """

    base_curve: PolylineCurve
    offsetted_plane: Plane
    top_curve: PolylineCurve
    top_surface: Brep
    side_surfaces: list[Brep]


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
        self._populated_points = populate_geometry(
            surface, self._piece_count, self._seed
        )
        self._voronoi_cells = voronoi_3d(self._populated_points)
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
        self._hat_previews: list[Union[Curve, Plane, Brep]] = []

    def build(self) -> list[Hat]:
        """Builds Hats from a sequence of polyline curves."""
        return [self._build_hat(p) for p in self._refined_pieces]

    def _build_hat(self, curve: PolylineCurve) -> Hat:
        """Builds a Hat from a polyline curve."""
        offsetted_plane = self._build_offsetted_plane(curve)
        top_curve = self._build_top_curve(curve, offsetted_plane)

        # Create surfaces
        top_surface = self._create_surface_from_curve(top_curve)
        if top_surface is None:
            raise UnexpectedShapeError([top_curve])
        side_surfaces = self._create_side_surfaces(curve, top_curve)

        # Store intermediates for debugging
        self._hat_previews.append(curve)
        self._hat_previews.append(offsetted_plane)
        self._hat_previews.append(top_curve)
        self._hat_previews.append(top_surface)
        self._hat_previews.extend(side_surfaces)

        return Hat(
            base_curve=curve,
            offsetted_plane=offsetted_plane,
            top_curve=top_curve,
            top_surface=top_surface,
            side_surfaces=side_surfaces,
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
        Builds the top curve on the offsetted plane.
        """
        base_points = self._extract_vertices(base_curve)
        center = self._calculate_center(base_points)

        projected_lines: list[Line] = []

        for i in range(len(base_points)):
            p1 = base_points[i]
            p2 = base_points[(i + 1) % len(base_points)]
            intersection_line = self._project_segment(p1, p2, center, offsetted_plane)
            projected_lines.append(intersection_line)

        # Build top curve by finding intersection points between adjacent lines
        if len(projected_lines) < 3:
            raise UnexpectedShapeError(projected_lines)

        top_points: list[Point3d] = []

        for i in range(len(projected_lines)):
            current_line = projected_lines[i]
            next_line = projected_lines[(i + 1) % len(projected_lines)]

            # Find where current line meets the next line
            intersection_point = self._find_line_intersection_point(
                current_line, next_line
            )
            top_points.append(intersection_point)

        # Create closed polyline
        polyline = PolylineCurve([*top_points, top_points[0]])

        # Check for self-intersections and clean them
        polyline = self._clean_self_intersections(polyline)

        return polyline

    def _calculate_center(self, points: Sequence[Point3d]) -> Point3d:
        """Calculates the center point of a sequence of points."""
        return Point3d(
            sum(p.X for p in points) / len(points),
            sum(p.Y for p in points) / len(points),
            sum(p.Z for p in points) / len(points),
        )

    def _get_area(self, curve: Curve) -> float:
        """Calculates the area of a closed curve."""
        if not curve.IsClosed:
            raise UnexpectedShapeError([curve])

        area_props = AreaMassProperties.Compute(curve)
        return area_props.Area

    def _clean_self_intersections(self, polyline: PolylineCurve) -> PolylineCurve:
        """
        Detects and removes self-intersections from a polyline curve.
        Returns the largest piece if self-intersections are found.
        """
        # Check for self-intersections
        intersections = Intersection.CurveSelf(polyline, TOLERANCE)

        # If there are no self-intersections, return the original
        if intersections.Count == 0:
            return polyline

        # Split the curve at all self-intersection points
        t_params: list[float] = []
        for intersection in intersections:
            t_params.append(intersection.ParameterA)
            t_params.append(intersection.ParameterB)

        # Split the curve
        split_curves = [c for c in polyline.Split(t_params) if c.IsValid]

        # Check if all split curves are closed
        all_closed = all(c.IsClosed for c in split_curves)

        # If all are closed, return the largest one
        if all_closed:
            largest_curve = max(split_curves, key=self._get_area)
            return PolygonBuilder([largest_curve]).build()[0]

        # Try to join the unclosed curves
        open_curves = [c for c in split_curves if not c.IsClosed]
        joined = list(Curve.JoinCurves(open_curves, TOLERANCE))

        # Check if we got a single closed curve
        if len(joined) == 1 and joined[0].IsClosed:
            joined_curve = joined[0]
            return PolygonBuilder([joined_curve]).build()[0]

        raise UnexpectedShapeError(split_curves)

    def _create_surface_from_curve(self, curve: PolylineCurve) -> Optional[Brep]:
        """Creates a planar surface from a closed curve."""
        if not curve.IsClosed:
            raise UnexpectedShapeError([curve])

        # Create a planar Brep from the closed curve
        breps = list(Brep.CreatePlanarBreps(curve, TOLERANCE) or [])

        if len(breps) == 0:
            return None

        return breps[0]

    def _create_side_surfaces(
        self, base_curve: PolylineCurve, top_curve: PolylineCurve
    ) -> list[Brep]:
        """
        Creates side surfaces by connecting base curve segments to top curve vertices.
        """
        base_points = self._extract_vertices(base_curve)
        top_points = self._extract_vertices(top_curve)

        side_surfaces: list[Brep] = []

        for i in range(len(base_points)):
            # Get segment start and end points
            base_start = base_points[i]
            base_end = base_points[(i + 1) % len(base_points)]

            # Find closest top vertices
            closest_start = self._find_closest_point(base_start, top_points)
            closest_end = self._find_closest_point(base_end, top_points)

            # Skip if both points map to the same top vertex (would create a triangle)
            if closest_start.DistanceTo(closest_end) < TOLERANCE:
                continue

            # Create four-sided polygon
            quad_points = [base_start, base_end, closest_end, closest_start]
            quad_curve = PolylineCurve([*quad_points, quad_points[0]])

            # Create surface from the quad
            side_surface = self._create_surface_from_curve(quad_curve)
            if side_surface is not None:
                side_surfaces.append(side_surface)

        return side_surfaces

    def _find_closest_point(
        self, point: Point3d, candidates: Sequence[Point3d]
    ) -> Point3d:
        """Finds the closest point from a list of candidate points."""
        if len(candidates) == 0:
            raise ValueError("No candidate points provided")

        closest = candidates[0]
        min_distance = point.DistanceTo(closest)

        for candidate in candidates[1:]:
            distance = point.DistanceTo(candidate)
            if distance < min_distance:
                min_distance = distance
                closest = candidate

        return closest

    def _find_line_intersection_point(self, line1: Line, line2: Line) -> Point3d:
        """
        Finds the intersection point between two lines on a plane.
        Projects line1's starting point along line1's direction onto an infinite line2.
        """
        # Treat line2 as infinite by using LineLine without bounded constraint
        success, t1, _ = Intersection.LineLine(line1, line2, TOLERANCE, False)

        if not success:
            raise UnexpectedShapeError([line1, line2])

        return line1.PointAt(t1)

    def _project_segment(
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
    try:
        result, intermediates = main(geo_input)
    except Exception as e:
        error = e
