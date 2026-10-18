"""Pure binary PLY loading, without viewer dependencies."""
import numpy as np
from pathlib import Path

PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def read_ply_header(path):
    """Read the binary PLY header and return its vertex information."""

    with open(path, "rb") as f:
        if f.readline().decode("ascii").strip() != "ply":
            raise ValueError("This file is not a valid PLY file.")

        file_format = None
        vertex_count = None
        vertex_properties = []
        current_element = None

        while True:
            raw_line = f.readline()

            if not raw_line:
                raise ValueError("PLY header ended before end_header.")

            line = raw_line.decode("ascii").strip()

            if not line or line.startswith(("comment ", "obj_info ")):
                continue

            fields = line.split()

            if fields[0] == "format":
                file_format = fields[1]

            elif fields[0] == "element":
                current_element = fields[1]

                if current_element == "vertex":
                    vertex_count = int(fields[2])

            elif fields[0] == "property" and current_element == "vertex":
                if fields[1] == "list":
                    raise ValueError("List properties are not supported.")

                type_name, property_name = fields[1], fields[2]

                if type_name not in PLY_TYPES:
                    raise ValueError(
                        f"Unsupported PLY property type: {type_name}"
                    )

                vertex_properties.append(
                    (property_name, PLY_TYPES[type_name])
                )

            elif fields[0] == "end_header":
                data_offset = f.tell()
                break

    if file_format != "binary_little_endian":
        raise ValueError(
            f"Unsupported format: {file_format}. "
            "Expected binary_little_endian."
        )

    if vertex_count is None or vertex_count < 0:
        raise ValueError("Invalid PLY vertex count.")

    vertex_dtype = np.dtype(vertex_properties)

    return vertex_count, vertex_dtype, data_offset


def load_ply(path, max_points=500_000):
    """Load XYZ and RGB, optionally sampling large point clouds."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"PLY file not found: {path}")

    count, dtype, offset = read_ply_header(path)

    expected_bytes = count * dtype.itemsize
    actual_bytes = path.stat().st_size - offset

    if actual_bytes < expected_bytes:
        raise ValueError(
            f"PLY data is truncated: expected {expected_bytes:,} bytes, "
            f"found {actual_bytes:,}."
        )

    vertices = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=(count,),
    )

    fields = set(vertices.dtype.names or ())
    required = {"x", "y", "z"}

    if not required.issubset(fields):
        raise ValueError(
            f"PLY requires x, y, z fields. Found: {sorted(fields)}"
        )

    # Randomly sample for browser performance, without changing the PLY.
    if max_points > 0 and count > max_points:
        rng = np.random.default_rng(42)
        indices = np.sort(
            rng.choice(count, size=max_points, replace=False)
        )
        vertices = vertices[indices]

    points = np.column_stack(
        (vertices["x"], vertices["y"], vertices["z"])
    ).astype(np.float32)

    if {"red", "green", "blue"}.issubset(fields):
        colors = np.column_stack(
            (vertices["red"], vertices["green"], vertices["blue"])
        ).astype(np.uint8)
    else:
        print("No RGB fields found; using gray.")
        colors = np.full((len(points), 3), 180, dtype=np.uint8)

    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid]

    if len(points) == 0:
        raise ValueError("No finite 3D points found in the PLY.")

    print("\n" + "=" * 60)
    print("POINT CLOUD")
    print("=" * 60)
    print(f"File: {path.resolve()}")
    print(f"Vertices in file: {count:,}")
    print(f"Points displayed: {len(points):,}")
    print(f"XYZ minimum: {points.min(axis=0)}")
    print(f"XYZ maximum: {points.max(axis=0)}")
    print("=" * 60)

    return points, colors


# ============================================================
# VISER VIEWER
# ============================================================
