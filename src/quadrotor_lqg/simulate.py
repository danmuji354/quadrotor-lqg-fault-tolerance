import argparse
from pathlib import Path

import yaml

from .core import run_episode
from .reporting import write_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nominal.yaml")
    parser.add_argument("--output", default="artifacts/nominal")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    fault = config.get("fault", {})
    dropout = tuple(fault["gps_dropout_s"]) if fault.get("gps_dropout_s") else None
    result = run_episode(
        config["seed"],
        config["duration_s"],
        config["sample_time_s"],
        dropout,
        fault.get("gps_bias_m", 0.0),
        config.get("wind_force_n", 0.0),
    )
    write_result(result, args.output, {"config": config, "controller": "lqg", "estimator": "ekf"})
    print({k: v for k, v in result.items() if not hasattr(v, "shape")})


if __name__ == "__main__":
    main()
