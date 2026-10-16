"""Small NumPy/SciPy geometry helpers used inside CloudCompare.

These operations use the host's pinned NumPy/SciPy dependencies without adding
another native geometry runtime to the CloudCompare compatibility matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _xyz_array(points) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Expected an N x 3 array of XYZ coordinates.")
    return array


def write_ascii_ply_xyz(path, points) -> None:
    """Write an N x 3 point array as an ASCII PLY file."""
    xyz = _xyz_array(points)
    output = Path(path)
    with output.open("w", encoding="ascii", newline="\n") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(xyz)}\n")
        ply_file.write("property double x\n")
        ply_file.write("property double y\n")
        ply_file.write("property double z\n")
        ply_file.write("end_header\n")
        np.savetxt(ply_file, xyz, fmt="%.17g")


def read_ascii_ply_xyz(path) -> np.ndarray:
    """Read XYZ coordinates from an ASCII PLY file."""
    input_path = Path(path)
    vertex_count = None
    vertex_properties = []
    current_element = None

    with input_path.open("r", encoding="ascii") as ply_file:
        if ply_file.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {input_path}")

        while True:
            line = ply_file.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {input_path}")
            fields = line.strip().split()
            if not fields or fields[0] == "comment":
                continue
            if fields[:2] == ["format", "ascii"]:
                continue
            if fields[0] == "format":
                raise ValueError("Only ASCII PLY files are supported.")
            if fields[0] == "element":
                current_element = fields[1]
                if current_element == "vertex":
                    vertex_count = int(fields[2])
                    vertex_properties = []
                continue
            if fields[0] == "property" and current_element == "vertex":
                vertex_properties.append(fields[-1])
                continue
            if fields[0] == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"PLY file has no vertex element: {input_path}")
        missing = {"x", "y", "z"} - set(vertex_properties)
        if missing:
            raise ValueError(f"PLY vertex data is missing: {', '.join(sorted(missing))}")
        if vertex_count == 0:
            return np.empty((0, 3), dtype=np.float64)

        data = np.loadtxt(ply_file, max_rows=vertex_count, ndmin=2)

    if data.shape[0] != vertex_count:
        raise ValueError(
            f"PLY declares {vertex_count} vertices but contains {data.shape[0]}."
        )
    indices = [vertex_properties.index(axis) for axis in ("x", "y", "z")]
    return np.asarray(data[:, indices], dtype=np.float64)


def rotate_points_around_z(points, angle_radians, center) -> np.ndarray:
    """Rotate XYZ points around a Z axis passing through *center*."""
    xyz = _xyz_array(points)
    rotation_center = np.asarray(center, dtype=np.float64)
    if rotation_center.shape != (3,):
        raise ValueError("Rotation center must contain exactly three values.")

    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    rotation = np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )
    return (xyz - rotation_center) @ rotation.T + rotation_center


def write_voxel_mesh_ply(path, points, voxel_size) -> None:
    """Create an ASCII triangle-mesh PLY containing occupied voxel cubes."""
    xyz = _xyz_array(points)
    size = float(voxel_size)
    if size <= 0:
        raise ValueError("Voxel size must be greater than zero.")
    if len(xyz) == 0:
        raise ValueError("Cannot voxelize an empty point cloud.")

    origin = xyz.min(axis=0) - (size * 0.5)
    voxel_indices = np.unique(np.floor((xyz - origin) / size).astype(np.int64), axis=0)
    corner_offsets = np.array(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        dtype=np.int64,
    )
    cube_triangles = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )

    vertex_lookup = {}
    lattice_vertices = []
    triangles = []
    for voxel_index in voxel_indices:
        cube_vertex_indices = []
        for corner in voxel_index + corner_offsets:
            key = tuple(int(value) for value in corner)
            vertex_index = vertex_lookup.get(key)
            if vertex_index is None:
                vertex_index = len(lattice_vertices)
                vertex_lookup[key] = vertex_index
                lattice_vertices.append(corner.copy())
            cube_vertex_indices.append(vertex_index)
        triangles.extend(
            tuple(cube_vertex_indices[index] for index in triangle)
            for triangle in cube_triangles
        )

    vertices = origin + np.asarray(lattice_vertices, dtype=np.float64) * size
    output = Path(path)
    with output.open("w", encoding="ascii", newline="\n") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(vertices)}\n")
        ply_file.write("property double x\n")
        ply_file.write("property double y\n")
        ply_file.write("property double z\n")
        ply_file.write(f"element face {len(triangles)}\n")
        ply_file.write("property list uchar int vertex_indices\n")
        ply_file.write("end_header\n")
        np.savetxt(ply_file, vertices, fmt="%.17g")
        for triangle in triangles:
            ply_file.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")
