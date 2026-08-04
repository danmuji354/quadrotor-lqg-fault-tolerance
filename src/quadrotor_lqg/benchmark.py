import argparse
import json
from pathlib import Path

from .core import run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="standard", choices=["standard"])
    parser.add_argument("--output", default="artifacts/benchmark")
    args = parser.parse_args()
    scenarios = {
        "nominal": {},
        "gps_dropout": {"dropout": (3.0, 4.0)},
        "wind_gust": {"wind_force_n": 0.8},
        "gps_bias": {"gps_bias_m": 0.08},
    }
    metrics = {
        name: {k: v for k, v in run_episode(**settings).items() if not hasattr(v, "shape")}
        for name, settings in scenarios.items()
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "manifest.json").write_text(
        json.dumps({"suite": args.suite, "scenarios": list(scenarios)}, indent=2) + "\n"
    )
    print(metrics)


if __name__ == "__main__":
    main()
