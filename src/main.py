"""Module for converting smooth Brep surfaces into polygonal Breps."""

from typing import Optional

from Rhino.Geometry import Brep, Mesh, MeshFace, MeshingParameters, Point3d, Polyline


class BrepPolygonizer:
    """Converts smooth Brep surfaces into polygonal polysurfaces."""

    def __init__(self, density: float) -> None:
        """Initialize the BrepPolygonizer.

        Args:
            density: Mesh density parameter. Higher values create more polygons.
                    Typical range: 0.01 (coarse) to 10.0 (fine).
        """
        self.density = density

    def polygonize(self, brep: Brep) -> Optional[Brep]:
        """Convert a smooth Brep into a polygonal polysurface.

        Args:
            brep: The input Brep to polygonize.

        Returns:
            A new Brep composed of planar polygon faces, or None if conversion fails.
        """
        if not brep.IsValid:
            return None

        # Create mesh from the Brep
        mesh = self._brep_to_mesh(brep)
        if not mesh or not mesh.IsValid:
            return None

        # Convert mesh to polygonal Brep
        polygonal_brep = self._mesh_to_brep(mesh)
        return polygonal_brep

    def _brep_to_mesh(self, brep: Brep) -> Optional[Mesh]:
        """Convert a Brep to a mesh using the density parameter.

        Args:
            brep: The Brep to mesh.

        Returns:
            A mesh representation of the Brep, or None if meshing fails.
        """
        # Create mesh parameters based on density
        mesh_params = self._create_mesh_parameters()

        # Generate mesh from Brep
        meshes = Mesh.CreateFromBrep(brep, mesh_params)
        mesh_list = list(meshes)
        if not mesh_list:
            return None

        # Join all mesh pieces into a single mesh
        joined_mesh = Mesh()
        for mesh in mesh_list:
            joined_mesh.Append(mesh)

        if joined_mesh.Faces.Count == 0:
            return None

        # Optimize the mesh
        joined_mesh.Compact()
        joined_mesh.Vertices.CombineIdentical(True, True)
        joined_mesh.Vertices.CullUnused()

        return joined_mesh

    def _create_mesh_parameters(self) -> MeshingParameters:
        """Create meshing parameters based on the density value.

        Returns:
            MeshingParameters configured for the specified density.
        """
        params = MeshingParameters.Default

        # Scale parameters based on density
        # Higher density = smaller edge length = more faces
        edge_length = 1.0 / self.density if self.density > 0 else 1.0

        params.MaximumEdgeLength = edge_length
        params.MinimumEdgeLength = edge_length * 0.1
        params.GridAspectRatio = 6.0
        params.GridAmplification = 1.0
        params.SimplePlanes = False
        params.RefineGrid = True
        params.JaggedSeams = False
        params.GridMinCount = 16

        return params

    def _mesh_to_brep(self, mesh: Mesh) -> Optional[Brep]:
        """Convert a mesh into a Brep polysurface.

        Args:
            mesh: The mesh to convert.

        Returns:
            A Brep where each mesh face becomes a planar surface, or None if fails.
        """
        if not mesh or mesh.Faces.Count == 0:
            return None

        # Collect all planar surfaces from mesh faces
        surfaces: list[Brep] = []

        for i in range(mesh.Faces.Count):
            face_brep = self._create_face_brep(mesh, i)
            if face_brep and face_brep.IsValid:
                surfaces.append(face_brep)

        if len(surfaces) == 0:
            return None

        # Join all surfaces into a single Brep
        joined_breps_iterable = Brep.JoinBreps(surfaces, 0.001)
        if joined_breps_iterable is None:  # type: ignore[comparison-overlap]
            return None

        joined_breps = list(joined_breps_iterable)
        if not joined_breps:
            return None

        return joined_breps[0]

    def _create_face_brep(self, mesh: Mesh, face_index: int) -> Optional[Brep]:
        """Create a Brep surface from a single mesh face.

        Args:
            mesh: The source mesh.
            face_index: Index of the face to convert.

        Returns:
            A planar Brep surface representing the mesh face, or None if creation fails.
        """
        face = mesh.Faces[face_index]

        # Get face vertices
        vertices = self._get_face_vertices(mesh, face)
        if len(vertices) < 3:
            return None

        # Create a planar surface from the vertices
        return self._create_planar_surface(vertices)

    def _get_face_vertices(self, mesh: Mesh, face: MeshFace) -> list[Point3d]:
        """Extract vertices from a mesh face.

        Args:
            mesh: The source mesh.
            face: The mesh face.

        Returns:
            List of vertices defining the face (3 for triangles, 4 for quads).
        """
        vertices: list[Point3d] = []

        # Convert Point3f to Point3d
        vertices.append(Point3d(mesh.Vertices[face.A]))
        vertices.append(Point3d(mesh.Vertices[face.B]))
        vertices.append(Point3d(mesh.Vertices[face.C]))

        # Check if it's a quad (4 vertices) or triangle (3 vertices)
        if face.IsQuad:
            vertices.append(Point3d(mesh.Vertices[face.D]))

        return vertices

    def _create_planar_surface(self, vertices: list[Point3d]) -> Optional[Brep]:
        """Create a planar Brep surface from a list of vertices.

        Args:
            vertices: The vertices defining the polygon (ordered).

        Returns:
            A planar Brep surface, or None if creation fails.
        """
        if len(vertices) < 3:
            return None

        # Create a polyline from vertices (closed loop)
        points = list(vertices)
        points.append(vertices[0])  # Close the loop
        polyline = Polyline(points)

        if not polyline.IsValid or polyline.Count < 4:
            return None

        # Convert polyline to curve
        curve = polyline.ToNurbsCurve()
        if not curve or not curve.IsValid:
            return None

        # Try to create a planar surface from the closed curve
        breps_iterable = Brep.CreatePlanarBreps(curve, 0.001)
        if breps_iterable is None:  # type: ignore[comparison-overlap]
            return None

        breps = list(breps_iterable)
        if not breps:
            return None

        return breps[0]


inputs = globals()
shape: Optional[Brep] = inputs["shape"]
density: Optional[float] = inputs["density"]

if shape is None or density is None:
    result = None
else:
    polygonizer = BrepPolygonizer(density)
    result = polygonizer.polygonize(shape)
