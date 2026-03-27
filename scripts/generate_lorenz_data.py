#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def generate_lorenz(n_steps: int, dt: float, sigma: float, rho: float, beta: float, x0: float, y0: float, z0: float) -> pd.DataFrame:
    xs = [float(x0)]
    ys = [float(y0)]
    zs = [float(z0)]
    for _ in range(1, int(n_steps)):
        x, y, z = xs[-1], ys[-1], zs[-1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        xs.append(x + dt * dx)
        ys.append(y + dt * dy)
        zs.append(z + dt * dz)
    return pd.DataFrame({"t": range(int(n_steps)), "x": xs, "y": ys, "z": zs})


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic Lorenz attractor CSV for smoke tests.")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--n_steps", type=int, default=512)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--rho", type=float, default=28.0)
    ap.add_argument("--beta", type=float, default=2.6666666666666665)
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--y0", type=float, default=1.0)
    ap.add_argument("--z0", type=float, default=1.05)
    args = ap.parse_args()
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = generate_lorenz(args.n_steps, args.dt, args.sigma, args.rho, args.beta, args.x0, args.y0, args.z0)
    df.to_csv(out_csv, index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
