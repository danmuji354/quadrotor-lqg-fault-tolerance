# Flight and sensor model

The plant state is `[x, z, vx, vz, pitch, pitch_rate]`. Collective thrust acts along the body vertical axis and torque controls pitch. The EKF observes GPS x/z and an attitude channel. During GPS dropout it retains only the valid attitude row in the correction step.

