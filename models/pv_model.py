import numpy as np

class PVModel:
    def __init__(self, capacity=500):
        self.capacity = capacity  # kW peak capacity

    def generate_irradiance(self, steps):
        # Normalized irradiance curve (0 to 1)
        return np.maximum(0, np.sin(np.linspace(0, np.pi, steps)))

    def compute_power(self, irradiance):
        return self.capacity * irradiance
