from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def write_result(result: dict, output: str | Path, manifest: dict) -> None:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"time", "states", "estimates", "controls"}
    metrics = {key: value for key, value in result.items() if key not in arrays}
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    matrix = np.column_stack(
        (result["time"], result["states"], result["estimates"], result["controls"])
    )
    with (path / "timeseries.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "x",
                "z",
                "vx",
                "vz",
                "theta",
                "q",
                "x_hat",
                "z_hat",
                "vx_hat",
                "vz_hat",
                "theta_hat",
                "q_hat",
                "thrust",
                "torque",
            ]
        )
        writer.writerows(matrix)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    axes[0].plot(result["states"][:, 0], result["states"][:, 1], label="true")
    axes[0].plot(result["estimates"][:, 0], result["estimates"][:, 1], "--", label="EKF")
    axes[0].scatter([1], [1], marker="x", label="target")
    axes[0].set(xlabel="x [m]", ylabel="z [m]")
    axes[0].grid()
    axes[0].legend()
    axes[1].plot(result["time"], result["states"][:, 4])
    axes[1].set(xlabel="time [s]", ylabel="pitch [rad]")
    axes[1].grid()
    fig.tight_layout()
    fig.savefig(path / "response.png", dpi=150)
    plt.close(fig)
