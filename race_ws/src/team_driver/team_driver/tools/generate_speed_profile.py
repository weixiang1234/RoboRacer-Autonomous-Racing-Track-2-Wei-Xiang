#!/usr/bin/env python3
"""Generate an OFFLINE curvature-based speed-profile prototype for Track 2.

Not connected to the ROS 2 driver. It does not modify a racing-line CSV or
protected judging files. Numerical limits are trial parameters, NOT measured
vehicle grip/acceleration limits. Validate with the simulator before use.

Example:
  python generate_speed_profile.py \
    --line maps/icra26_compete_racing_v1.csv \
    --output maps/icra26_compete_speed_profile_v1.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


def estimate_curvature(points, window):
    """Signed curvature of a closed, approximately equally spaced XY loop."""
    n = len(points)
    if window < 5 or window % 2 == 0 or window >= n:
        raise ValueError("--smooth-window must be odd, >= 5, and < waypoint count")

    segment_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if np.any(segment_lengths < 1e-5):
        raise ValueError("Consecutive waypoints must have nonzero separation")
    spacing = float(np.mean(segment_lengths))

    dx = savgol_filter(points[:, 0], window, 3, deriv=1, delta=spacing, mode="wrap")
    dy = savgol_filter(points[:, 1], window, 3, deriv=1, delta=spacing, mode="wrap")
    d2x = savgol_filter(points[:, 0], window, 3, deriv=2, delta=spacing, mode="wrap")
    d2y = savgol_filter(points[:, 1], window, 3, deriv=2, delta=spacing, mode="wrap")

    denom = (dx * dx + dy * dy) ** 1.5
    if np.any(denom < 1e-8):
        raise ValueError("Near-zero derivative; inspect waypoint ordering and smoothing")
    curvature = (dx * d2y - dy * d2x) / denom
    return curvature, segment_lengths


def apply_longitudinal_limits(cap, ds, a_accel, a_brake):
    """Limit attainable speed along a closed route in both travel directions."""
    v = cap.copy()
    n = len(v)
    for _ in range(50):
        previous = v.copy()

        # Forward acceleration: cannot instantly accelerate after a corner.
        for i in range(n):
            j = (i + 1) % n
            allowable = np.sqrt(v[i] ** 2 + 2.0 * a_accel * ds[i])
            v[j] = min(v[j], allowable)

        # Backward braking: must slow before entering a corner.
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            allowable = np.sqrt(v[j] ** 2 + 2.0 * a_brake * ds[i])
            v[i] = min(v[i], allowable)

        if np.max(np.abs(v - previous)) < 1e-10:
            break
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", required=True, help="Existing racing-line XY CSV")
    parser.add_argument("--output", required=True, help="New analysis CSV to create")
    parser.add_argument("--vmax", type=float, default=3.30, help="Absolute planning speed cap [m/s]")
    parser.add_argument("--alat", type=float, default=6.50, help="Assumed lateral acceleration limit [m/s^2]")
    parser.add_argument("--accel", type=float, default=1.50, help="Assumed forward acceleration limit [m/s^2]")
    parser.add_argument("--brake", type=float, default=1.00, help="Assumed braking deceleration [m/s^2]")
    parser.add_argument("--smooth-window", type=int, default=11, help="Odd closed-loop Savitzky-Golay window")
    args = parser.parse_args()

    for name in ("vmax", "alat", "accel", "brake"):
        if getattr(args, name) <= 0.0:
            parser.error("--%s must be positive" % name)

    points = np.loadtxt(args.line, comments="#", delimiter=",", ndmin=2)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 20:
        raise ValueError("Expected at least 20 two-column XY waypoints")
    if not np.all(np.isfinite(points)):
        raise ValueError("Waypoints must be finite")

    curvature, ds = estimate_curvature(points, args.smooth_window)
    v_cap = np.minimum(args.vmax, np.sqrt(args.alat / np.maximum(np.abs(curvature), 1e-9)))
    v_profile = apply_longitudinal_limits(v_cap, ds, args.accel, args.brake)

    # Verify cyclic acceleration/braking constraints are satisfied.
    ahead = np.roll(v_profile, -1)
    acc_ok = np.all(ahead * ahead - v_profile * v_profile <= 2 * args.accel * ds + 1e-8)
    brake_ok = np.all(v_profile * v_profile - ahead * ahead <= 2 * args.brake * ds + 1e-8)
    if not (acc_ok and brake_ok):
        raise RuntimeError("Longitudinal envelope validation failed")

    s = np.concatenate(([0.0], np.cumsum(ds[:-1])))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "x_m", "y_m", "s_m", "curvature_1pm", "v_curvature_mps", "v_profile_mps"])
        for i, (x, y) in enumerate(points):
            writer.writerow([i, f"{x:.4f}", f"{y:.4f}", f"{s[i]:.5f}",
                             f"{curvature[i]:.6f}", f"{v_cap[i]:.5f}", f"{v_profile[i]:.5f}"])

    print("OFFLINE PLANNING ONLY — ROS controller unchanged")
    print("Waypoints:               %d" % len(points))
    print("Closed-loop length:      %.3f m" % ds.sum())
    print("Peak |curvature|:        %.3f 1/m" % np.max(np.abs(curvature)))
    print("Profile speed (min/max): %.3f / %.3f m/s" % (v_profile.min(), v_profile.max()))
    print("Profile average speed:   %.3f m/s" % v_profile.mean())
    print("Forward/braking checks:  PASS / PASS")
    print("Assumptions: alat=%.2f, accel=%.2f, brake=%.2f m/s^2" % (args.alat, args.accel, args.brake))
    print("Wrote:                   %s" % out)


if __name__ == "__main__":
    main()
