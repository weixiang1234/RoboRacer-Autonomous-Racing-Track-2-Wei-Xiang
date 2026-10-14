#!/usr/bin/env python3
"""Reproduce the Track 2 'racing_v1' path from the supplied occupancy map.

The line is an *offline* optimization: the ROS 2 driver only reads its CSV at
runtime. Keep this script in team_driver for the hackathon's reproducibility
requirement, and commit the generated racing-line CSV under maps/.

Default algorithm and parameters match the version that produced
icra26_compete_racing_v1.csv (validated 10 laps without collisions).

Example (from the Track 2 repository root):

    python3 race_ws/src/team_driver/team_driver/tools/generate_racing_line.py \\
        --map maps/icra26_compete.yaml \\
        --centerline maps/icra26_compete_centerline.csv \\
        --output maps/icra26_compete_racing_v1.csv

Dependencies: numpy, scipy, Pillow, PyYAML (available in the judging image).
The generator runs *offline*, not inside the real-time control loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize
import yaml


def load_inputs(map_yaml: Path, centerline_csv: Path):
    with map_yaml.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    if meta.get("mode", "trinary") != "trinary":
        raise ValueError("The generator expects a trinary occupancy map")
    if len(meta["origin"]) != 3 or abs(float(meta["origin"][2])) > 1e-12:
        raise ValueError("The map origin must have zero yaw")

    image_path = (map_yaml.parent / meta["image"]).resolve()
    pixels = np.array(Image.open(image_path))
    if pixels.ndim != 2:
        raise ValueError("The map must be 8-bit grayscale")

    resolution = float(meta["resolution"])
    origin = np.array(meta["origin"][:2], dtype=float)
    negate = bool(int(meta.get("negate", 0)))
    free_thresh = float(meta.get("free_thresh", 0.196))

    occupancy = pixels.astype(float) / 255.0 if negate else (255 - pixels) / 255.0
    free = occupancy < free_thresh
    # Occupied AND unknown are non-drivable. With the official map, only 254 is free.

    points = []
    with centerline_csv.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            x, y = map(float, line.split(",")[:2])
            points.append((x, y))

    points = np.asarray(points, dtype=float)
    if len(points) < 30:
        raise ValueError("Expected a closed racing circuit with >= 30 waypoints")

    return pixels, free, resolution, origin, points


def generate(map_yaml: Path, centerline_csv: Path, output_csv: Path):
    pixels, free, resolution, origin, pts = load_inputs(map_yaml, centerline_csv)
    h, w = pixels.shape
    n = len(pts)

    clearance_grid = distance_transform_edt(free) * resolution

    def clearance_xy(q):
        q = np.asarray(q)
        row = (h - 1) - ((q[..., 1] - origin[1]) / resolution)
        col = (q[..., 0] - origin[0]) / resolution
        return map_coordinates(
            clearance_grid,
            [row.ravel(), col.ravel()],
            order=1,
            mode="constant",
            cval=0,
        ).reshape(q.shape[:-1])

    # The normal of the supplied centerline defines the lateral search direction.
    k = 3
    tangent = np.roll(pts, -k, axis=0) - np.roll(pts, k, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    # Sample the connected, drivable lateral corridor at each path waypoint.
    margin = 0.50  # m, clearance from the point to the nearest non-drivable cell
    offset_samples = np.arange(-1.20, 1.201, 0.01)
    lo = np.zeros(n)
    hi = np.zeros(n)

    for i in range(n):
        candidates = pts[i] + offset_samples[:, None] * normal[i]
        good = clearance_xy(candidates) >= margin
        j0 = int(np.argmin(np.abs(offset_samples)))

        if not good[j0]:
            lo[i] = hi[i] = 0.0
            continue

        a = b = j0
        while a > 0 and good[a - 1]:
            a -= 1
        while b < len(offset_samples) - 1 and good[b + 1]:
            b += 1

        lo[i] = offset_samples[a]
        hi[i] = offset_samples[b]

    # Minimize deviations around the spawn/start/finish region.
    for i in list(range(0, 16)) + list(range(n - 8, n)):
        lo[i] = max(lo[i], -0.02)
        hi[i] = min(hi[i], 0.02)

    def metrics(d):
        q = pts + d[:, None] * normal
        q_prev = np.roll(q, 1, axis=0)
        q_next = np.roll(q, -1, axis=0)

        a = q - q_prev
        b = q_next - q
        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)
        chord = np.linalg.norm(q_next - q_prev, axis=1)

        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        curvature = 2.0 * cross / np.maximum(la * lb * chord, 1e-8)

        seg = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
        return q, curvature, seg.sum(), seg

    def objective(d):
        _, curvature, length, seg = metrics(d)
        curvature_cost = np.sum(
            curvature**2 * 0.5 * (seg + np.roll(seg, 1))
        )
        offset_smoothness = np.sum((np.roll(d, -1) - d) ** 2)
        return curvature_cost + 0.002 * offset_smoothness + 0.0005 * length

    result = minimize(
        objective,
        np.zeros(n),
        method="L-BFGS-B",
        bounds=list(zip(lo, hi)),
        options={"maxiter": 120, "maxfun": 36000, "ftol": 1e-10, "maxls": 30},
    )

    offset = result.x
    racing, k_race, racing_length, _ = metrics(offset)
    _, k_center, center_length, _ = metrics(np.zeros(n))

    # Check the connecting line segments, not just the discrete waypoints.
    dense = []
    for i in range(n):
        a = racing[i]
        b = racing[(i + 1) % n]
        length = np.linalg.norm(b - a)
        count = max(2, int(np.ceil(length / 0.01)) + 1)
        t = np.linspace(0.0, 1.0, count, endpoint=False)
        dense.append(a + t[:, None] * (b - a))
    dense = np.vstack(dense)
    min_clearance = float(clearance_xy(dense).min())

    # Fail closed: never export a path that enters the non-drivable region.
    # 0.49 tolerates 1-cm interpolation / occupancy-grid effects around 0.50 m.
    if min_clearance < 0.49:
        raise RuntimeError(
            f"Generated path minimum clearance {min_clearance:.3f} m < 0.49 m; "
            "aborting without writing a CSV"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8") as f:
        f.write("# x_m, y_m\n")
        f.write("# Conservative minimum-curvature candidate derived from icra26_compete map.\n")
        f.write("# Keeps approximately >= 0.50 m centre-point clearance in the supplied occupancy grid.\n")
        f.write("# Start/finish region kept close to the supplied centreline.\n")
        for x, y in racing:
            f.write(f"{x:.4f}, {y:.4f}\n")

    print(f"Waypoints:                 {n}")
    print(f"Centreline length:         {center_length:.3f} m")
    print(f"Candidate length:          {racing_length:.3f} m")
    print(f"Centreline max curvature:  {np.max(np.abs(k_center)):.3f} 1/m")
    print(f"Candidate max curvature:   {np.max(np.abs(k_race)):.3f} 1/m")
    print(f"Minimum map clearance:     {min_clearance:.3f} m")
    print(f"Maximum lateral shift:     {np.max(np.abs(offset)):.3f} m")
    print(f"Optimizer:                 {result.message}")
    print(f"Wrote:                     {output_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path, help="Input occupancy YAML")
    parser.add_argument("--centerline", required=True, type=Path, help="Input centerline CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output racing-line CSV")
    args = parser.parse_args()
    generate(args.map, args.centerline, args.output)


if __name__ == "__main__":
    main()
