import numpy as np

class LoadModel:
    def generate_load(self, steps):
        base = 400 + 100 * np.sin(np.linspace(0, 2*np.pi, steps))
        noise = np.random.normal(0, 20, steps)
        return base + noise
