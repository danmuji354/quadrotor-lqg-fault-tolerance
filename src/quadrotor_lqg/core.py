"""Nonlinear planar flight model, EKF, and LQG controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

Array = np.ndarray


@dataclass(frozen=True)
class QuadParams:
    mass: float = 1.0
    inertia: float = 0.025
    gravity: float = 9.81
    linear_drag: float = 0.12
    max_thrust: float = 18.0
    max_torque: float = 0.8


class PlanarQuadrotor:
    """Six-state x-z-pitch rigid-body model."""

    def __init__(self, params: QuadParams | None = None):
        self.params = QuadParams() if params is None else params

    def derivative(self, state: Array, control: Array, wind_force: float = 0.0) -> Array:
        _x, _z, vx, vz, theta, pitch_rate = np.asarray(state, dtype=float)
        thrust = float(np.clip(control[0], 0.0, self.params.max_thrust))
        torque = float(np.clip(control[1], -self.params.max_torque, self.params.max_torque))
        p = self.params
        return np.array(
            [
                vx,
                vz,
                (-thrust * np.sin(theta) - p.linear_drag * vx + wind_force) / p.mass,
                (thrust * np.cos(theta) - p.mass * p.gravity - p.linear_drag * vz) / p.mass,
                pitch_rate,
                torque / p.inertia,
            ]
        )

    def step(self, state: Array, control: Array, dt: float, wind_force: float = 0.0) -> Array:
        f = lambda s: self.derivative(s, control, wind_force)
        k1 = f(state)
        k2 = f(state + dt * k1 / 2)
        k3 = f(state + dt * k2 / 2)
        k4 = f(state + dt * k3)
        return np.asarray(state) + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def linear_model(self) -> tuple[Array, Array]:
        p = self.params
        a = np.zeros((6, 6))
        b = np.zeros((6, 2))
        a[0, 2] = 1
        a[1, 3] = 1
        a[4, 5] = 1
        a[2, 2] = -p.linear_drag / p.mass
        a[2, 4] = -p.gravity
        a[3, 3] = -p.linear_drag / p.mass
        b[3, 0] = 1 / p.mass
        b[5, 1] = 1 / p.inertia
        return a, b


class EKF:
    """EKF with partial measurement updates during GPS dropout."""

    def __init__(self, initial: Array, covariance: Array | None = None):
        self.state = np.asarray(initial, dtype=float).copy()
        self.covariance = np.eye(6) * 0.1 if covariance is None else covariance.copy()
        self.process_noise = np.diag([1e-5, 1e-5, 2e-3, 2e-3, 1e-5, 1e-3])
        self.measurement_noise = np.diag([0.03**2, 0.03**2, 0.008**2])

    def predict(self, plant: PlanarQuadrotor, control: Array, dt: float) -> None:
        a, _ = plant.linear_model()
        transition = np.eye(6) + dt * a
        self.state = plant.step(self.state, control, dt)
        self.covariance = transition @ self.covariance @ transition.T + self.process_noise

    def update(self, measurement: Array) -> None:
        full_h = np.zeros((3, 6))
        full_h[0, 0] = 1
        full_h[1, 1] = 1
        full_h[2, 4] = 1
        valid = np.isfinite(measurement)
        if not np.any(valid):
            return
        h = full_h[valid]
        r = self.measurement_noise[np.ix_(valid, valid)]
        innovation = measurement[valid] - h @ self.state
        s = h @ self.covariance @ h.T + r
        gain = np.linalg.solve(s, h @ self.covariance).T
        self.state += gain @ innovation
        identity = np.eye(6)
        self.covariance = (identity - gain @ h) @ self.covariance @ (
            identity - gain @ h
        ).T + gain @ r @ gain.T


class LQGController:
    def __init__(self, plant: PlanarQuadrotor):
        a, b = plant.linear_model()
        q = np.diag([14, 18, 5, 6, 35, 4])
        r = np.diag([0.8, 0.12])
        riccati = solve_continuous_are(a, b, q, r)
        self.gain = np.linalg.solve(r, b.T @ riccati)
        self.params = plant.params

    def command(self, estimate: Array, target: Array) -> Array:
        error = np.asarray(estimate) - np.asarray(target)
        delta = -self.gain @ error
        return np.array(
            [
                np.clip(
                    self.params.mass * self.params.gravity + delta[0], 0, self.params.max_thrust
                ),
                np.clip(delta[1], -self.params.max_torque, self.params.max_torque),
            ]
        )


def run_episode(
    seed: int = 0,
    duration_s: float = 10.0,
    sample_time_s: float = 0.01,
    dropout: tuple[float, float] | None = None,
    gps_bias_m: float = 0.0,
    wind_force_n: float = 0.0,
) -> dict:
    rng = np.random.default_rng(seed)
    plant = PlanarQuadrotor()
    controller = LQGController(plant)
    state = np.zeros(6)
    target = np.array([1.0, 1.0, 0, 0, 0, 0])
    estimator = EKF(state)
    time = np.arange(0.0, duration_s + sample_time_s / 2, sample_time_s)
    states = np.zeros((len(time), 6))
    estimates = np.zeros_like(states)
    controls = np.zeros((len(time), 2))
    for index, now in enumerate(time):
        measurement = np.array([state[0] + gps_bias_m, state[1] + gps_bias_m, state[4]])
        measurement += rng.normal(0, [0.03, 0.03, 0.008])
        if dropout and dropout[0] <= now <= dropout[1]:
            measurement[:2] = np.nan
        estimator.update(measurement)
        control = controller.command(estimator.state, target)
        states[index] = state
        estimates[index] = estimator.state
        controls[index] = control
        if index + 1 < len(time):
            gust = wind_force_n if 2.0 <= now <= 4.0 else 0.0
            state = plant.step(state, control, sample_time_s, gust)
            estimator.predict(plant, control, sample_time_s)
    position_error = np.linalg.norm(states[:, :2] - target[:2], axis=1)
    settled = time >= duration_s / 2
    return {
        "time": time,
        "states": states,
        "estimates": estimates,
        "controls": controls,
        "position_rmse_m": float(np.sqrt(np.mean(position_error[settled] ** 2))),
        "final_position_error_m": float(position_error[-1]),
        "minimum_altitude_m": float(np.min(states[:, 1])),
        "maximum_covariance_eigenvalue": float(np.max(np.linalg.eigvalsh(estimator.covariance))),
    }
