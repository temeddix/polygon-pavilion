"""Generate pavilion structures from smooth Brep shapes using Voronoi tessellation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib import import_module
from typing import Any, NamedTuple, Protocol, TypeVar

from Rhino.Geometry import (
    AreaMassProperties,
    Brep,
    Curve,
    GeometryBase,
    Line,
    Plane,
    Point3d,
    PolylineCurve,
    TextDot,
    Transform,
    Vector3d,
)
from Rhino.Geometry.Intersect import Intersection

T = TypeVar("T")

TOLERANCE = 0.001
OFFSET_DISTANCE_FACTOR = 0.08
COLLAPSE_PRECISION = 2
HAT_SIDE_ANGLE = math.pi / 6
MIN_POLYGON_VERTICES = 3
MIN_TOP_POINTS = 2


def populate_geometry(brep: Brep, piece_count: int, seed: int) -> list[Point3d]:
    """Populate a Brep geometry with random points."""
    module = import_module("ghpythonlib.components")
    func = getattr(module, "PopulateGeometry")
    result = func(brep, piece_count, seed)
    return list(result)


def voronoi_3d(points: list[Point3d]) -> list[Brep]:
    """Generate 3D Voronoi cells from a sequence of points."""
    module = import_module("ghpythonlib.components")
    func = getattr(module, "Voronoi3D")
    result = func(points)
    return list(getattr(result, "cells"))


def join_adjacent_breps(breps: list[Brep]) -> Brep:
    """Join the top and all side surfaces into a single polysurface."""
    joined = list(Brep.JoinBreps(breps, TOLERANCE))
    if len(joined) != 1:
        raise UnexpectedShapeError(breps)
    return joined[0]


def create_planar_brep_from_curve(curve: Curve) -> Brep:
    """Create a planar Brep from a closed curve."""
    if not curve.IsClosed:
        raise UnexpectedShapeError([curve])
    breps = list(Brep.CreatePlanarBreps(curve, TOLERANCE) or [])
    if len(breps) != 1:
        raise UnexpectedShapeError([curve])
    return breps[0]


class InvalidInputError(Exception):
    """Exception raised for invalid input types."""

    def __init__(self, expected_type: type[Any], actual_value: object) -> None:
        """Initialize the exception with expected type and actual value."""
        message = f"Invalid input value: expected {expected_type}, got {actual_value}"
        super().__init__(message)


class UnexpectedShapeError(Exception):
    """Exception raised for unexpected geometry shapes."""

    def __init__(
        self,
        geometry: list[Any],
    ) -> None:
        """Initialize the exception with the unexpected geometry."""
        self.preview = geometry
        message = f"Unexpected geometry {geometry}"
        super().__init__(message)


class Hat(NamedTuple):
    """Represent a hat structure with base curve, top plane, and surfaces."""

    base_curve: PolylineCurve
    top_plane: Plane
    top: Brep
    sides: list[Brep]


class GeometryOutput(NamedTuple):
    """Output containing the resulting hats and debug shapes."""

    cut_lines: list[Curve]
    score_lines: list[Curve]
    intermediates: list[Sequence[GeometryBase | Point3d | Brep]]
    labels: list[TextDot]


class UnrolledHat(NamedTuple):
    """Represent an unrolled, flat hat shape."""

    top: Brep
    sides: list[Brep]


class Flag(NamedTuple):
    """Represent a flag structure with base curve and surface."""

    rect: Brep
    flap: Brep


class LaserLines(NamedTuple):
    """Dataclass for storing laser cutting lines."""

    cut_lines: list[Curve]
    score_lines: list[Curve]


class GeometryBuilder(Protocol):
    """Protocol for geometry build step worker classes."""

    def build(self) -> object:
        """Do the work of this geometry build step."""
        ...

    def get_intermediates(self) -> Sequence[GeometryBase | Point3d | Brep]:
        """Get intermediate geometries from this step for debugging."""
        ...


class SurfaceSplitter:
    """Builder class for splitting smooth surfaces into pieces."""

    def __init__(self, smooth_surface: Brep, piece_count: int, seed: int) -> None:
        """Initialize the surface splitter with parameters."""
        self._smooth_surface = smooth_surface
        self._piece_count = piece_count
        self._seed = seed
        self._populated_points: list[Point3d] = []
        self._voronoi_cells: list[Brep] = []

    def build(self) -> list[Curve]:
        """Build list of curves by splitting the surface."""
        surface = self._smooth_surface
        self._populated_points = populate_geometry(
            surface,
            self._piece_count,
            self._seed,
        )
        self._voronoi_cells = voronoi_3d(self._populated_points)
        return [
            self._join_curves(list(Intersection.BrepBrep(cell, surface, TOLERANCE)[1]))
            for cell in self._voronoi_cells
        ]

    def get_intermediates(self) -> Sequence[GeometryBase | Point3d | Brep]:
        """Get intermediate debug geometries from this step."""
        result: Sequence[GeometryBase | Point3d | Brep] = []
        result.extend(self._populated_points)
        result.extend(self._voronoi_cells)
        return result

    def _join_curves(self, curves: list[Curve]) -> Curve:
        """Join multiple curves into a single curve."""
        joined = list(Curve.JoinCurves(curves, TOLERANCE))
        if len(joined) != 1:
            raise UnexpectedShapeError(curves)
        return joined[0]


class PolygonBuilder:
    """Builder class for creating polyline curves from curves."""

    def __init__(self, raw_pieces: list[Curve], collapse_length: float) -> None:
        """Initialize the polygon builder with raw pieces and collapse length."""
        self._raw_pieces = raw_pieces
        self._collapse_length = collapse_length
        self._refined_pieces: list[PolylineCurve] = []
        self._vertex_map: dict[tuple[float, float, float], Point3d] = {}

    def build(self) -> list[PolylineCurve]:
        """Build closed polyline curves from raw curves."""
        # First pass: extract all vertices and build collapse mapping
        all_points: list[Point3d] = []
        for piece in self._raw_pieces:
            points = self._extract_vertices(piece)
            all_points.extend(points)

        # Build vertex collapse map
        self._build_vertex_collapse_map(all_points)

        # Second pass: build polygons using the collapsed vertices
        return [self._build_polygon(p) for p in self._raw_pieces]

    def _build_vertex_collapse_map(self, points: list[Point3d]) -> None:
        """Build a map from original vertex positions to collapsed positions.

        Vertices that are close together will map to the same collapsed position.
        """
        # Sort points to ensure consistent processing
        sorted_points = sorted(points, key=lambda p: (p.X, p.Y, p.Z))

        for point in sorted_points:
            point_key = (
                round(point.X, COLLAPSE_PRECISION),
                round(point.Y, COLLAPSE_PRECISION),
                round(point.Z, COLLAPSE_PRECISION),
            )

            # Check if this point is close to any already mapped point
            found_cluster = False
            for mapped_point in self._vertex_map.values():
                if point.DistanceTo(mapped_point) < self._collapse_length:
                    # Use the existing mapped point
                    self._vertex_map[point_key] = mapped_point
                    found_cluster = True
                    break

            if not found_cluster:
                # This is a new cluster center
                self._vertex_map[point_key] = point

    def _get_collapsed_point(self, point: Point3d) -> Point3d:
        """Get the collapsed version of a point using the vertex map."""
        point_key = (
            round(point.X, COLLAPSE_PRECISION),
            round(point.Y, COLLAPSE_PRECISION),
            round(point.Z, COLLAPSE_PRECISION),
        )
        return self._vertex_map.get(point_key, point)

    def _build_polygon(self, curve: Curve) -> PolylineCurve:
        """Build a closed polyline curve from a curve."""
        points = self._extract_vertices(curve)
        collapsed_points = self._collapse_small_segments(points)
        polyline = self._points_to_closed_polyline_curve(collapsed_points)
        self._refined_pieces.append(polyline)
        return polyline

    def get_intermediates(self) -> Sequence[GeometryBase | Point3d | Brep]:
        """Get intermediate debug geometries from this step."""
        return list(self._refined_pieces)

    def _collapse_small_segments(self, points: list[Point3d]) -> list[Point3d]:
        """Collapse small segments by using the global vertex collapse map."""
        if len(points) < MIN_POLYGON_VERTICES:
            return list(points)

        # Apply global vertex collapse mapping
        collapsed: list[Point3d] = []

        for point in points:
            mapped_point = self._get_collapsed_point(point)

            # Only add if it's different from the last point
            if not collapsed or mapped_point.DistanceTo(collapsed[-1]) >= TOLERANCE:
                collapsed.append(mapped_point)

        # Check if the last point is the same as the first point (closing the loop)
        if len(collapsed) > 1 and collapsed[-1].DistanceTo(collapsed[0]) < TOLERANCE:
            collapsed.pop()

        # Need at least 3 points for a polygon
        if len(collapsed) < MIN_POLYGON_VERTICES:
            # If collapsing would result in too few points, return original
            return list(points)

        return collapsed

    def _points_to_closed_polyline_curve(self, points: list[Point3d]) -> PolylineCurve:
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

    def __init__(
        self,
        original_shape: Brep,
        refined_pieces: list[PolylineCurve],
    ) -> None:
        """Initialize the hat builder with original shape and refined pieces."""
        self._original_shape = original_shape
        self._refined_pieces = refined_pieces
        self._hat_previews: list[Curve | Brep] = []

    def build(self) -> list[Hat]:
        """Build Hats from a sequence of polyline curves."""
        return [self._build_hat(p) for p in self._refined_pieces]

    def _build_hat(self, curve: PolylineCurve) -> Hat:
        """Build a Hat from a polyline curve."""
        top_plane = self._build_top_plane(curve)
        curve = self._orient_curve(curve, top_plane.ZAxis)
        top_curve = self._build_top_curve(curve, top_plane)

        # Create surfaces
        top_surface = create_planar_brep_from_curve(top_curve)
        side_surfaces = self._create_hat_sides(curve, top_curve)

        # Store intermediates for debugging
        self._hat_previews.append(curve)
        self._hat_previews.append(top_curve)
        self._hat_previews.append(top_surface)
        self._hat_previews.extend(side_surfaces)

        return Hat(
            base_curve=curve,
            top_plane=top_plane,
            top=top_surface,
            sides=side_surfaces,
        )

    def get_intermediates(self) -> Sequence[GeometryBase | Point3d | Brep]:
        """Get intermediate debug geometries from this step."""
        return list(self._hat_previews)

    def _extract_vertices(self, curve: PolylineCurve) -> list[Point3d]:
        """Extract vertices from a curve's segments."""
        if not curve.IsClosed:
            raise UnexpectedShapeError([curve])
        return list(curve.ToArray())[:-1]  # Exclude duplicate closing point

    def _build_top_plane(self, curve: PolylineCurve) -> Plane:
        """Build a top plane by fitting a plane to the curve's points."""
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

    def _orient_curve(
        self,
        curve: PolylineCurve,
        plane_normal: Vector3d,
    ) -> PolylineCurve:
        """Orient the curve to be clockwise when viewed from the plane normal direction.

        If the curve is clockwise, it reverses the point order.
        """
        points = self._extract_vertices(curve)

        if len(points) < MIN_POLYGON_VERTICES:
            return curve

        # Calculate the signed area using the plane normal
        # Positive area means clockwise, negative means clockwise
        signed_area = 0.0

        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            # Create edge vector
            edge = Vector3d(p2 - p1)

            # Vector from origin to midpoint of edge
            mid = Point3d(
                (p1.X + p2.X) / 2,
                (p1.Y + p2.Y) / 2,
                (p1.Z + p2.Z) / 2,
            )
            to_mid = Vector3d(mid)

            # Cross product gives area contribution
            cross = Vector3d.CrossProduct(to_mid, edge)

            # Dot with plane normal gives signed contribution
            signed_area += Vector3d.Multiply(cross, plane_normal)

        # If signed area is negative, curve is clockwise - reverse it
        if signed_area > 0:
            reversed_points = list(reversed(points))
            return PolylineCurve([*reversed_points, reversed_points[0]])

        return curve

    def _calculate_diameter(self, points: list[Point3d]) -> float:
        """Calculate the diameter as the maximum distance between any two points."""
        max_distance = 0.0
        for i, p1 in enumerate(points):
            for p2 in points[i + 1 :]:
                distance = p1.DistanceTo(p2)
                max_distance = max(max_distance, distance)
        return max_distance

    def _is_flipped(self, plane: Plane) -> bool:
        """Check if the plane normal is opposite to the original shape's normal."""
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
        self,
        base_curve: PolylineCurve,
        top_plane: Plane,
    ) -> PolylineCurve:
        """Build the top curve on the top plane."""
        base_points = self._extract_vertices(base_curve)
        center = self._calculate_center(base_points)

        projected_lines: list[Line] = []

        for i in range(len(base_points)):
            p1 = base_points[i]
            p2 = base_points[(i + 1) % len(base_points)]
            intersection_line = self._project_segment(p1, p2, center, top_plane)
            projected_lines.append(intersection_line)

        # Build top curve by finding intersection points between adjacent lines
        if len(projected_lines) < MIN_POLYGON_VERTICES:
            raise UnexpectedShapeError(projected_lines)

        top_points: list[Point3d] = []

        for i in range(len(projected_lines)):
            current_line = projected_lines[i]
            next_line = projected_lines[(i + 1) % len(projected_lines)]

            # Find where current line meets the next line
            intersection_point = self._find_line_intersection_point(
                current_line,
                next_line,
            )
            top_points.append(intersection_point)

        # Create closed polyline
        polyline = PolylineCurve([*top_points, top_points[0]])

        # Check for self-intersections and clean them
        return self._clean_self_intersections(polyline)

    def _calculate_center(self, points: list[Point3d]) -> Point3d:
        """Calculate the center point of a sequence of points."""
        return Point3d(
            sum(p.X for p in points) / len(points),
            sum(p.Y for p in points) / len(points),
            sum(p.Z for p in points) / len(points),
        )

    def _get_area(self, curve: Curve) -> float:
        """Calculate the area of a closed curve."""
        if not curve.IsClosed:
            raise UnexpectedShapeError([curve])

        area_props = AreaMassProperties.Compute(curve)
        return area_props.Area

    def _clean_self_intersections(self, polyline: PolylineCurve) -> PolylineCurve:
        """Detect and remove self-intersections from a polyline curve.

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
            return PolygonBuilder([largest_curve], 0.0).build()[0]

        # Try to join the unclosed curves
        open_curves = [c for c in split_curves if not c.IsClosed]
        joined = list(Curve.JoinCurves(open_curves, TOLERANCE))

        # Check if we got a single closed curve
        if len(joined) == 1 and joined[0].IsClosed:
            joined_curve = joined[0]
            return PolygonBuilder([joined_curve], 0.0).build()[0]

        raise UnexpectedShapeError(split_curves)

    def _create_hat_sides(
        self,
        base_curve: PolylineCurve,
        top_curve: PolylineCurve,
    ) -> list[Brep]:
        """Create side surfaces connecting base and top curve segments."""
        base_points = self._extract_vertices(base_curve)
        top_points = self._extract_vertices(top_curve)

        side_surfaces: list[Brep] = []

        for i in range(len(base_points)):
            # Get base segment
            base_start = base_points[i]
            base_end = base_points[(i + 1) % len(base_points)]

            # Find the two closest consecutive points on the top curve
            # This ensures we create a proper quad that connects to the top surface edge
            closest_indices = self._find_two_closest_consecutive_points(
                base_start,
                base_end,
                top_points,
            )

            if closest_indices is None:
                continue

            idx1, idx2 = closest_indices
            top_start = top_points[idx1]
            top_end = top_points[idx2]

            # Create four-sided polygon connecting base segment to top segment
            # Order: base_start -> base_end -> top_end -> top_start
            quad_points = [base_start, base_end, top_end, top_start]
            quad_curve = PolylineCurve([*quad_points, quad_points[0]])

            # Create surface from the quad
            side_surface = create_planar_brep_from_curve(quad_curve)
            side_surfaces.append(side_surface)

        return side_surfaces

    def _find_two_closest_consecutive_points(
        self,
        base_start: Point3d,
        base_end: Point3d,
        top_points: list[Point3d],
    ) -> tuple[int, int] | None:
        """Find two consecutive points on top curve closest to base segment."""
        if len(top_points) < MIN_TOP_POINTS:
            return None

        # Calculate the midpoint of the base segment
        base_mid = Point3d(
            (base_start.X + base_end.X) / 2,
            (base_start.Y + base_end.Y) / 2,
            (base_start.Z + base_end.Z) / 2,
        )

        # Find the edge on the top curve closest to the base midpoint
        min_distance = float("inf")
        best_idx = 0

        for i in range(len(top_points)):
            p1 = top_points[i]
            p2 = top_points[(i + 1) % len(top_points)]

            # Calculate midpoint of this top edge
            edge_mid = Point3d((p1.X + p2.X) / 2, (p1.Y + p2.Y) / 2, (p1.Z + p2.Z) / 2)

            # Distance from base midpoint to this edge midpoint
            distance = base_mid.DistanceTo(edge_mid)

            if distance < min_distance:
                min_distance = distance
                best_idx = i

        return (best_idx, (best_idx + 1) % len(top_points))

    def _find_line_intersection_point(self, line1: Line, line2: Line) -> Point3d:
        """Find the intersection point between two lines on a plane.

        Projects line1's starting point along line1's direction onto an infinite line2.
        """
        # Treat line2 as infinite by using LineLine without bounded constraint
        bounded = False
        success, t1, _ = Intersection.LineLine(line1, line2, TOLERANCE, bounded)

        if not success:
            raise UnexpectedShapeError([line1, line2])

        return line1.PointAt(t1)

    def _project_segment(
        self,
        p1: Point3d,
        p2: Point3d,
        center: Point3d,
        top_plane: Plane,
    ) -> Line:
        """Project segment endpoints along a 60-degree rotated direction.

        Finds their intersection with the top plane.
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
        dot_with_normal = Vector3d.Multiply(cross, top_plane.ZAxis)

        # Rotate 60 degrees (to get 30 degree angle with plane)
        # Flip the angle sign to rotate inward instead of outward
        angle = math.pi / 2 - HAT_SIDE_ANGLE
        if dot_with_normal > 0:
            angle = -angle

        # Create rotation transform around the segment
        rotation_transform = Transform.Rotation(angle, rotation_axis, segment_mid)

        # Start with the plane's normal direction
        plane_normal = Vector3d(top_plane.ZAxis)
        plane_normal.Unitize()

        # Apply rotation to get the projection direction
        projection_direction = Vector3d(plane_normal)
        projection_direction.Transform(rotation_transform)

        # Project p1 and p2 to the top plane along the rotated direction
        top_p1 = self._project_point_to_plane(p1, projection_direction, top_plane)
        top_p2 = self._project_point_to_plane(p2, projection_direction, top_plane)

        # Create the intersection line
        return Line(top_p1, top_p2)

    def _project_point_to_plane(
        self,
        point: Point3d,
        direction: Vector3d,
        plane: Plane,
    ) -> Point3d:
        """Project a point along a direction vector to intersect with a plane."""
        # Create a ray from the point in the given direction
        ray = Line(point, direction)

        # Find intersection parameter with the plane
        success, t = Intersection.LinePlane(ray, plane)

        if not success:
            raise UnexpectedShapeError([ray, plane])

        # Return the point at the intersection
        return ray.PointAt(t)


class HatUnroller:
    """Builder class for unrolling Hat structures into flat 2D patterns."""

    def __init__(self, hats: list[Hat], smooth_surface: Brep) -> None:
        """Initialize the hat unroller with hats and original shape."""
        self._smooth_surface = smooth_surface
        self._hats = hats
        self._unrolled_hats: list[Brep] = []
        self._text_dots: list[TextDot] = []

    def build(self) -> list[UnrolledHat]:
        """Build unrolled flat patterns from Hat structures."""
        unrolled = [self._unroll_hat(hat) for hat in self._hats]
        return self._arrange_in_grid(unrolled)

    def _unroll_hat(self, hat: Hat) -> UnrolledHat:
        """Unroll a single Hat into a flat 2D pattern using geometric properties.

        The top face is placed on the world XY plane, and side faces are rotated
        30 degrees outward around their shared edges with the top.
        """
        # Transform to align top to world plane
        xform_to_world = self._prepare_top_transform(hat)

        # Transform and collect top surface
        unrolled_top = hat.top.DuplicateBrep()
        unrolled_top.Transform(xform_to_world)

        # Transform the top plane's Z axis to determine rotation direction
        top_normal = Vector3d(hat.top_plane.ZAxis)
        top_normal.Transform(xform_to_world)

        # Unfold all side surfaces using their stored top_edge information
        unfolded_sides = self._unfold_all_sides(hat.sides, xform_to_world)
        self._unrolled_hats.append(join_adjacent_breps([unrolled_top, *unfolded_sides]))

        return UnrolledHat(top=unrolled_top, sides=unfolded_sides)

    def _prepare_top_transform(self, hat: Hat) -> Transform:
        """Prepare the transformation to move the top surface to the world XY plane."""
        # Get plane from the top surface
        top_plane = hat.top_plane

        # Create transform to align top surface to world XY plane
        world_plane = Plane.WorldXY
        world_plane.Origin = Point3d(0, 0, 0)

        return Transform.PlaneToPlane(top_plane, world_plane)

    def _unfold_all_sides(
        self,
        hat_sides: list[Brep],
        xform_to_world: Transform,
    ) -> list[Brep]:
        """Unfolds all side surfaces by rotating them around their shared edges."""
        unfolded_sides: list[Brep] = []

        for hat_side in hat_sides:
            unfolded = self._unfold_single_side(hat_side, xform_to_world)
            if unfolded is not None:
                unfolded_sides.append(unfolded)

        return unfolded_sides

    def _unfold_single_side(
        self,
        hat_side: Brep,
        xform_to_world: Transform,
    ) -> Brep | None:
        """Unfolds a single side surface around its shared edge with the top."""
        # Get the hinge
        vertices = list(hat_side.Vertices)
        hinge_start = vertices[3].Location
        hinge_end = vertices[2].Location

        # Transform the hinge to world coordinates
        hinge_start.Transform(xform_to_world)
        hinge_end.Transform(xform_to_world)

        # Calculate rotation transform using the top normal
        rotation_xform = self._calculate_side_rotation(hinge_start, hinge_end)

        # Apply transforms: first to world plane, then rotate
        unfolded_side = hat_side.DuplicateBrep()
        unfolded_side.Transform(xform_to_world)
        unfolded_side.Transform(rotation_xform)

        return unfolded_side

    def _calculate_side_rotation(
        self,
        hinge_start: Point3d,
        hinge_end: Point3d,
    ) -> Transform:
        """Calculate the rotation transform for unfolding a side surface.

        Rotates 30 degrees around the shared edge (hinge).
        Uses the top plane's normal to determine rotation direction.
        """
        # Calculate rotation axis (along the shared edge)
        hinge_vector = Vector3d(hinge_end - hinge_start)
        hinge_vector.Unitize()

        # Create rotation transform around the hinge edge
        return Transform.Rotation(HAT_SIDE_ANGLE, hinge_vector, hinge_start)

    def _arrange_in_grid(self, breps: list[UnrolledHat]) -> list[UnrolledHat]:
        """Arrange the unrolled Breps in a grid layout with spacing.

        Based on the largest bounding box dimensions, positioned next to the
        original shape. Also creates TextDots for labeling original and
        unrolled pieces.
        """
        # Get original shape bounding box
        accurate = True
        original_bbox = self._smooth_surface.GetBoundingBox(accurate)
        original_max_x = original_bbox.Max.X
        original_min_y = original_bbox.Min.Y

        # Analyze all bounding boxes to find max width and height
        max_width = 0.0
        max_height = 0.0

        for unrolled_hat in breps:
            # Get combined bounding box of top and all sides
            all_breps = [unrolled_hat.top, *unrolled_hat.sides]

            # Calculate combined bounding box
            min_x = min(b.GetBoundingBox(accurate).Min.X for b in all_breps)
            max_x = max(b.GetBoundingBox(accurate).Max.X for b in all_breps)
            min_y = min(b.GetBoundingBox(accurate).Min.Y for b in all_breps)
            max_y = max(b.GetBoundingBox(accurate).Max.Y for b in all_breps)

            width = max_x - min_x
            height = max_y - min_y
            max_width = max(max_width, width)
            max_height = max(max_height, height)

        # Calculate grid dimensions (roughly square grid)
        grid_cols = math.ceil(math.sqrt(len(breps)))

        # Start grid offset from the original shape's boundaries with some spacing
        grid_start_x = original_max_x + max_width * 1.0
        grid_start_y = original_min_y

        # Arrange breps in grid and create labels
        arranged_hats: list[UnrolledHat] = []

        for i, unrolled_hat in enumerate(breps):
            row = i // grid_cols
            col = i % grid_cols

            # Calculate translation
            x_offset = grid_start_x + col * max_width * 1.2  # 20% spacing
            y_offset = grid_start_y + row * max_height * 1.2  # 20% spacing

            # Create translation transform
            translation = Transform.Translation(x_offset, y_offset, 0)

            # Apply translation to top and all sides
            arranged_top = unrolled_hat.top.DuplicateBrep()
            arranged_top.Transform(translation)

            arranged_sides: list[Brep] = []
            for side in unrolled_hat.sides:
                arranged_side = side.DuplicateBrep()
                arranged_side.Transform(translation)
                arranged_sides.append(arranged_side)

            arranged_hat = UnrolledHat(top=arranged_top, sides=arranged_sides)
            arranged_hats.append(arranged_hat)

            # Create TextDots for original and unrolled pieces
            label = str(i + 1)  # 1-based numbering

            # TextDot for original Hat piece (at its centroid)
            original_hat = self._hats[i]
            original_centroid = self._get_brep_centroid(original_hat.top)
            original_dot = TextDot(label, original_centroid)
            self._text_dots.append(original_dot)

            # TextDot for unrolled piece (at its centroid)
            unrolled_centroid = self._get_brep_centroid(arranged_top)
            unrolled_dot = TextDot(label, unrolled_centroid)
            self._text_dots.append(unrolled_dot)

        return arranged_hats

    def _get_brep_centroid(self, brep: Brep) -> Point3d:
        """Calculate the centroid of a Brep using area mass properties."""
        props = AreaMassProperties.Compute(brep)
        return props.Centroid

    def get_intermediates(self) -> Sequence[GeometryBase | Point3d | Brep]:
        """Get intermediate debug geometries from this step."""
        return list(self._unrolled_hats)

    def get_text_dots(self) -> list[TextDot]:
        """Get TextDot labels for original and unrolled pieces."""
        return list(self._text_dots)


class HatSettler:
    """Builder class for settling hats for gluing and assembly."""

    def __init__(
        self,
        unrolled_hats: list[UnrolledHat],
        glue_width: float,
        glue_inset: float,
        flap_width: float,
    ) -> None:
        """Initialize the hat settler."""
        self._unrolled_hats = unrolled_hats
        self._glue_width = glue_width
        self._glue_inset = glue_inset
        self._flap_width = flap_width
        self._settled_hats: list[Brep] = []

    def build(self) -> LaserLines:
        """Build the hat settling step."""
        # Join all brep faces (top, sides, flags) into single brep per hat
        for unrolled_hat in self._unrolled_hats:
            # Collect all flag breps (rect and flap)
            brep_faces = [unrolled_hat.top, *unrolled_hat.sides]
            for side in unrolled_hat.sides:
                flag = self._extrude_flags(side)
                brep_faces.extend(flag)
            joined_brep = join_adjacent_breps(brep_faces)
            self._settled_hats.append(joined_brep)

        cut_lines: list[Curve] = []
        score_lines: list[Curve] = []
        for settled_hat in self._settled_hats:
            laser_lines = self._brep_to_laser_lines(settled_hat)
            cut_lines.extend(laser_lines.cut_lines)
            score_lines.extend(laser_lines.score_lines)

        return LaserLines(cut_lines=cut_lines, score_lines=score_lines)

    def _brep_to_laser_lines(self, brep: Brep) -> LaserLines:
        """Convert a Brep into laser cutting and scoring lines.

        Edges longer than score_length are treated as cut lines,
        while shorter edges are treated as score lines.
        """
        cut_lines: list[Curve] = []
        score_lines: list[Curve] = []

        for edge in brep.Edges:
            curve = edge.DuplicateCurve()
            should_cut = len(list(edge.AdjacentFaces())) == 1
            if should_cut:
                cut_lines.append(curve)
            else:
                score_lines.append(curve)

        return LaserLines(cut_lines=cut_lines, score_lines=score_lines)

    def get_intermediates(self) -> Sequence[GeometryBase | Point3d | Brep]:
        """Get intermediate debug geometries from this step."""
        return list(self._settled_hats)

    def _extrude_flags(self, side_brep: Brep) -> Flag:
        """Create a rectangle extruded from the bottom line of a side Brep."""
        # Find the bottom edge (the first edge in the brep)
        bottom_edge = next(iter(side_brep.Edges))

        # Calculate the bottom edge direction vector
        bottom_edge_vector = bottom_edge.TangentAtStart
        bottom_edge_vector.Unitize()

        # Get the start and end points of the bottom and top edges
        bottom_pt_a = bottom_edge.PointAtStart + bottom_edge_vector * self._glue_inset
        bottom_pt_b = bottom_edge.PointAtEnd - bottom_edge_vector * self._glue_inset

        # Calculate perpendicular direction on XY plane using cross product with Z axis
        # Cross product of Z axis with edge gives perpendicular direction in XY plane
        outward_direction = Vector3d.CrossProduct(Vector3d.ZAxis, bottom_edge_vector)
        outward_direction *= self._glue_width

        # Create the four corners of the rectangle
        corner_a = Point3d(bottom_pt_a)
        corner_b = Point3d(bottom_pt_b)
        corner_c = Point3d(bottom_pt_b + outward_direction)
        corner_d = Point3d(bottom_pt_a + outward_direction)

        # Create the flap point
        flap_pt = corner_d - bottom_edge_vector * self._flap_width

        # Create a closed polyline curve for the rectangle
        rect_curve = PolylineCurve([corner_a, corner_b, corner_c, corner_d, corner_a])
        flap_curve = PolylineCurve([corner_a, corner_d, flap_pt, corner_a])

        # Create planar surfaces
        rect = create_planar_brep_from_curve(rect_curve)
        flap = create_planar_brep_from_curve(flap_curve)

        return Flag(rect=rect, flap=flap)


def extract_input(name: str, expected_type: type[T]) -> T:
    """Extract and validate an input value from globals.

    The input value is removed from globals after extraction.
    """
    obj = globals().pop(name)
    if not isinstance(obj, expected_type):
        raise InvalidInputError(expected_type, obj)
    return obj


def main() -> GeometryOutput:
    """Generate pavilion from a smooth Brep shape using Voronoi tessellation."""
    # Extract Grasshopper component inputs
    smooth_surface = extract_input("smooth_surface", Brep)
    piece_count = extract_input("piece_count", int)
    seed = extract_input("seed", int)
    collapse_length = extract_input("collapse_length", float)
    glue_width = extract_input("glue_width", float)
    glue_inset = extract_input("glue_inset", float)
    flap_width = extract_input("flap_width", float)

    # Build geometry using builders
    surface_splitter = SurfaceSplitter(smooth_surface, piece_count, seed)
    raw_pieces = surface_splitter.build()
    polygon_builder = PolygonBuilder(raw_pieces, collapse_length)
    refined_pieces = polygon_builder.build()
    hat_builder = HatBuilder(smooth_surface, refined_pieces)
    hats = hat_builder.build()
    hat_unroller = HatUnroller(hats, smooth_surface)
    unrolled_hats = hat_unroller.build()
    hat_settler = HatSettler(unrolled_hats, glue_width, glue_inset, flap_width)
    settled_hats = hat_settler.build()

    # Collect intermediates for debugging
    geo_builders: list[GeometryBuilder] = [
        surface_splitter,
        polygon_builder,
        hat_builder,
        hat_unroller,
        hat_settler,
    ]
    intermediates = [b.get_intermediates() for b in geo_builders]

    # Return final output
    return GeometryOutput(
        cut_lines=settled_hats.cut_lines,
        score_lines=settled_hats.score_lines,
        intermediates=intermediates,
        labels=hat_unroller.get_text_dots(),
    )


if __name__ == "__main__":
    try:
        cut_lines, score_lines, intermediates, labels = main()
    except Exception as e:
        error = e
