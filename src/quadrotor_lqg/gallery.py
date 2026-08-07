"""Generate website-ready figures, animation, and metadata from fault scenarios."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .core import run_episode

INK = "#0f172a"
BLUE = "#2563eb"
ORANGE = "#f59e0b"
SLATE = "#64748b"
GRID = "#cbd5e1"
PAPER = "#f8fafc"
SCENARIOS = {
    "nominal": {},
    "GPS dropout": {"dropout": (3.0, 4.0)},
    "wind gust": {"wind_force_n": 0.8},
    "GPS bias": {"gps_bias_m": 0.08},
}


def _style(axis: plt.Axes) -> None:
    axis.set_facecolor(PAPER)
    axis.grid(color=GRID, linewidth=0.8, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=SLATE)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=125, facecolor=PAPER)
    plt.close(figure)


def scenario_results() -> dict[str, dict]:
    return {name: run_episode(**settings) for name, settings in SCENARIOS.items()}


def _hero(results: dict[str, dict], output: Path) -> None:
    nominal = results["nominal"]
    dropout = results["GPS dropout"]
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    figure.patch.set_facecolor(PAPER)
    figure.suptitle("LQG tracking remains stable through GPS loss", color=INK, fontsize=20)
    figure.text(
        0.5,
        0.925,
        "True state and EKF estimate are separated by line style, not color alone.",
        ha="center",
        color=SLATE,
        fontsize=11,
    )
    axes[0].plot(
        nominal["states"][:, 0],
        nominal["states"][:, 1],
        color=BLUE,
        linewidth=2.4,
        label="nominal true",
    )
    axes[0].plot(
        dropout["states"][:, 0],
        dropout["states"][:, 1],
        color=ORANGE,
        linewidth=2.2,
        label="dropout true",
    )
    axes[0].plot(
        dropout["estimates"][:, 0],
        dropout["estimates"][:, 1],
        color=INK,
        linestyle="--",
        linewidth=1.6,
        label="dropout EKF",
    )
    axes[0].scatter([1], [1], color=INK, marker="x", s=90, linewidth=2, label="target")
    axes[0].set(xlabel="horizontal position x [m]", ylabel="altitude z [m]")
    axes[0].legend(frameon=False, loc="lower right")
    position_error = np.linalg.norm(dropout["states"][:, :2] - np.array([1.0, 1.0]), axis=1)
    estimate_error = np.linalg.norm(dropout["states"][:, :2] - dropout["estimates"][:, :2], axis=1)
    axes[1].axvspan(3.0, 4.0, color=ORANGE, alpha=0.16, label="GPS unavailable")
    axes[1].plot(dropout["time"], position_error, color=BLUE, linewidth=2.2, label="target error")
    axes[1].plot(
        dropout["time"], estimate_error, color=INK, linestyle="--", linewidth=1.8, label="EKF error"
    )
    axes[1].set(xlabel="time [s]", ylabel="position error [m]")
    axes[1].legend(frameon=False, loc="upper right")
    for axis in axes:
        _style(axis)
    figure.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.12, wspace=0.24)
    _save(figure, output / "hero.png")


def _benchmark(results: dict[str, dict], output: Path) -> None:
    labels = list(results)
    rmse = [results[name]["position_rmse_m"] for name in labels]
    final = [results[name]["final_position_error_m"] for name in labels]
    y = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    figure.patch.set_facecolor(PAPER)
    figure.suptitle("Fault-injection benchmark", color=INK, fontsize=20)
    figure.text(
        0.5,
        0.925,
        "Deterministic seed 0; lower is better. Acceptance references are shown explicitly.",
        ha="center",
        color=SLATE,
        fontsize=11,
    )
    axes[0].barh(y, rmse, color=BLUE, edgecolor=INK, linewidth=0.7)
    axes[0].axvline(0.20, color=ORANGE, linestyle="--", linewidth=2, label="0.20 m target")
    axes[0].set(yticks=y, yticklabels=labels, xlabel="position RMSE [m]")
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False)
    axes[1].barh(y, final, color="#93c5fd", edgecolor=INK, linewidth=0.7)
    axes[1].axvline(0.30, color=ORANGE, linestyle="--", linewidth=2, label="0.30 m target")
    axes[1].set(yticks=y, yticklabels=labels, xlabel="final position error [m]")
    axes[1].invert_yaxis()
    axes[1].legend(frameon=False)
    for axis, values in zip(axes, [rmse, final], strict=True):
        _style(axis)
        for index, value in enumerate(values):
            axis.text(value + 0.004, index, f"{value:.3f}", va="center", color=INK)
    figure.subplots_adjust(left=0.12, right=0.96, top=0.86, bottom=0.12, wspace=0.36)
    _save(figure, output / "benchmark.png")


def _animation(result: dict, output: Path) -> None:
    indices = np.linspace(0, len(result["time"]) - 1, 100, dtype=int)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    figure.patch.set_facecolor(PAPER)
    axis.set_facecolor(PAPER)
    axis.set(xlim=(-0.15, 1.3), ylim=(-0.05, 1.25), xlabel="x [m]", ylabel="z [m]")
    axis.grid(color=GRID, linewidth=0.8)
    axis.scatter([1], [1], marker="x", color=INK, s=90, linewidth=2)
    (true_path,) = axis.plot([], [], color=BLUE, linewidth=1.8, label="true")
    (estimate_path,) = axis.plot([], [], color=INK, linestyle="--", linewidth=1.3, label="EKF")
    (body,) = axis.plot([], [], color=ORANGE, linewidth=5, marker="o", markersize=5)
    status = axis.text(0.03, 0.93, "", transform=axis.transAxes, color=INK, fontsize=11)
    axis.legend(frameon=False, loc="lower right")

    def update(frame: int):
        index = indices[frame]
        state = result["states"][index]
        theta = state[4]
        dx, dz = 0.09 * np.cos(theta), 0.09 * np.sin(theta)
        true_path.set_data(result["states"][: index + 1, 0], result["states"][: index + 1, 1])
        estimate_path.set_data(
            result["estimates"][: index + 1, 0], result["estimates"][: index + 1, 1]
        )
        body.set_data([state[0] - dx, state[0] + dx], [state[1] - dz, state[1] + dz])
        fault = "GPS DROPOUT" if 3.0 <= result["time"][index] <= 4.0 else "GPS available"
        status.set_text(f"t = {result['time'][index]:4.1f} s   {fault}")
        status.set_color(ORANGE if fault == "GPS DROPOUT" else INK)
        return true_path, estimate_path, body, status

    animation = FuncAnimation(figure, update, frames=len(indices), interval=65, blit=True)
    animation.save(output / "demo.gif", writer=PillowWriter(fps=15), dpi=90)
    plt.close(figure)


def _architecture(output: Path) -> None:
    nodes = [
        (40, "Sensors", "GPS · altimeter · IMU"),
        (280, "Partial-update EKF", "state + covariance"),
        (550, "LQG feedback", "hover linearization"),
        (790, "Actuator limits", "thrust + torque"),
        (1030, "2D rigid body", "wind + faults"),
    ]
    elements = []
    for index, (x, title, subtitle) in enumerate(nodes):
        elements.append(
            f'<rect x="{x}" y="84" width="190" height="88" rx="14" fill="white" '
            f'stroke="{BLUE if index in {1, 2} else GRID}" stroke-width="2"/>'
        )
        elements.append(
            f'<text x="{x + 95}" y="119" text-anchor="middle" fill="{INK}" '
            f'font-family="Arial" font-size="15">{html.escape(title)}</text>'
        )
        elements.append(
            f'<text x="{x + 95}" y="145" text-anchor="middle" fill="{SLATE}" '
            f'font-family="Arial" font-size="12">{html.escape(subtitle)}</text>'
        )
        if index < len(nodes) - 1:
            next_x = nodes[index + 1][0]
            elements.append(
                f'<line x1="{x + 190}" y1="128" x2="{next_x - 12}" y2="128" '
                f'stroke="{INK}" stroke-width="2" marker-end="url(#arrow)"/>'
            )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="270" viewBox="0 0 1260 270">
<rect width="1260" height="270" fill="{PAPER}"/><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="{INK}"/></marker></defs>
<text x="40" y="42" fill="{INK}" font-family="Arial" font-size="22" font-weight="700">Fault-tolerant estimation and control</text>
{"".join(elements)}
<path d="M1125 185 C1125 232, 375 232, 375 184" fill="none" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#arrow)"/>
<text x="750" y="253" text-anchor="middle" fill="{SLATE}" font-family="Arial" font-size="13">measured state feedback with missing-channel updates</text></svg>'''
    (output / "architecture.svg").write_text(svg)


def gallery_contract(results: dict[str, dict]) -> dict:
    return {
        "schema_version": 1,
        "repository": "quadrotor-lqg-fault-tolerance",
        "title": "Quadrotor LQG Fault Tolerance",
        "tagline": "EKF and LQG flight control under dropout, bias, and wind.",
        "accent": BLUE,
        "highlights": [
            {"label": "nominal RMSE", "value": f"{results['nominal']['position_rmse_m']:.3f} m"},
            {
                "label": "1 s dropout RMSE",
                "value": f"{results['GPS dropout']['position_rmse_m']:.3f} m",
            },
            {
                "label": "dropout final error",
                "value": f"{results['GPS dropout']['final_position_error_m']:.3f} m",
            },
        ],
        "assets": [
            {
                "path": "hero.png",
                "role": "hero",
                "width": 1600,
                "height": 900,
                "alt": "Quadrotor trajectories and estimator error through GPS dropout.",
            },
            {
                "path": "benchmark.png",
                "role": "analysis",
                "width": 1600,
                "height": 900,
                "alt": "Position RMSE and final error across four injected faults.",
            },
            {
                "path": "demo.gif",
                "role": "animation",
                "width": 720,
                "height": 405,
                "alt": "Animated true and estimated quadrotor flight during GPS loss.",
            },
            {
                "path": "architecture.svg",
                "role": "diagram",
                "width": 1260,
                "height": 270,
                "alt": "Sensor, EKF, LQG, actuator, and nonlinear plant architecture.",
            },
        ],
        "reproduce": "python -m quadrotor_lqg.gallery --output artifacts/gallery",
    }


def generate_gallery(output: str | Path, animation: bool = True) -> dict:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    results = scenario_results()
    _hero(results, path)
    _benchmark(results, path)
    _architecture(path)
    if animation:
        _animation(results["GPS dropout"], path)
    rows = [
        {
            "scenario": name,
            "position_rmse_m": result["position_rmse_m"],
            "final_position_error_m": result["final_position_error_m"],
        }
        for name, result in results.items()
    ]
    with (path / "scenario_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    contract = gallery_contract(results)
    (path / "showcase.json").write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/gallery")
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate_gallery(args.output, not args.no_animation)["highlights"], indent=2))


if __name__ == "__main__":
    main()
