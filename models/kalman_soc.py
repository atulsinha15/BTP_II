import numpy as np
from filterpy.kalman import KalmanFilter


class SoCKalmanFilter:

    def __init__(self):

        self.kf = KalmanFilter(dim_x=1, dim_z=1)

        # State transition
        self.kf.F = np.array([[1]])

        # Measurement function
        self.kf.H = np.array([[1]])

        # Initial state estimate
        self.kf.x = np.array([[0.5]])

        # Initial uncertainty
        self.kf.P *= 0.1

        # Process noise
        self.kf.Q = np.array([[0.0005]])

        # Measurement noise
        self.kf.R = np.array([[0.01]])

    def predict(self):
        self.kf.predict()

    def update(self, measurement):
        self.kf.update(np.array([[measurement]]))

    def get_state(self):
        return float(self.kf.x[0][0])
