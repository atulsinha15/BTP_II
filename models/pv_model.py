import numpy as np

class PVModel:
    def __init__(self, capacity=500, noise_std=20):
        self.capacity = capacity
        self.noise_std = noise_std  # standard deviation in kW

    def generate_irradiance(self, steps):
        return np.maximum(0, np.sin(np.linspace(0, np.pi, steps)))

    def compute_power(self, irradiance):
        base_power = self.capacity * irradiance
        noise = np.random.normal(0, self.noise_std, len(irradiance))
        noisy_power = base_power + noise
        return np.maximum(0, noisy_power)
