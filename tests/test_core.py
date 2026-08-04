import numpy as np

from quadrotor_lqg.core import EKF, LQGController, PlanarQuadrotor, run_episode


def test_hover_is_equilibrium():
    plant = PlanarQuadrotor()
    hover = np.array([plant.params.mass * plant.params.gravity, 0])
    assert np.allclose(plant.derivative(np.zeros(6), hover), 0)


def test_closed_loop_linearization_is_stable():
    plant = PlanarQuadrotor()
    a, b = plant.linear_model()
    gain = LQGController(plant).gain
    assert np.all(np.real(np.linalg.eigvals(a - b @ gain)) < 0)


def test_ekf_covariance_remains_positive_semidefinite():
    plant = PlanarQuadrotor()
    ekf = EKF(np.zeros(6))
    hover = np.array([9.81, 0])
    for _ in range(50):
        ekf.predict(plant, hover, 0.01)
        ekf.update(np.array([0.0, np.nan, 0.0]))
    assert np.min(np.linalg.eigvalsh(ekf.covariance)) > -1e-10


def test_nominal_tracking_and_dropout_safety():
    nominal = run_episode(duration_s=6.0)
    dropout = run_episode(duration_s=6.0, dropout=(2.0, 3.0))
    assert nominal["final_position_error_m"] < 0.3
    assert dropout["minimum_altitude_m"] > -0.15
