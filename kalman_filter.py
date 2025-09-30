import numpy as np

class MaskKalmanFilter:
    def __init__(self, shape: tuple, q_val: float, r_val: float):
        self.Q = q_val
        self.R = r_val
        self.state = np.zeros(shape, dtype=np.float32)
        self.covariance = np.ones(shape, dtype=np.float32)

    def update(self, measurement: np.ndarray) -> np.ndarray:
        # Prediction step
        predicted_state = self.state
        predicted_covariance = self.covariance + self.Q

        # Update step
        innovation = measurement - predicted_state
        innovation_covariance = predicted_covariance + self.R
        kalman_gain = predicted_covariance / innovation_covariance

        # Update state and covariance
        self.state = predicted_state + kalman_gain * innovation
        self.covariance = (1 - kalman_gain) * predicted_covariance

        return self.state