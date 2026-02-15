class BatteryModel:
    def __init__(self, capacity=800, max_power=200, efficiency=0.95, dt=0.25):
        """
        capacity   : Total battery capacity in kWh
        max_power  : Maximum charge/discharge power in kW
        efficiency : Charge/discharge efficiency (0-1)
        dt         : Timestep duration in hours (0.25 = 15 min)
        """
        self.capacity = capacity
        self.max_power = max_power
        self.efficiency = efficiency
        self.dt = dt

        # Start at 50% SoC
        self.energy = 0.5 * capacity

    def charge(self, power):
        """
        Charge battery with given power (kW).
        Returns actual charging power used.
        """

        # Limit power to max rating
        power = min(power, self.max_power)

        # Convert power to energy (kWh)
        energy_added = power * self.dt * self.efficiency

        # Prevent overcharging
        available_space = self.capacity - self.energy
        energy_added = min(energy_added, available_space)

        self.energy += energy_added

        return energy_added / self.dt  # return actual power used

    def discharge(self, power):
        """
        Discharge battery with given power (kW).
        Returns actual discharging power delivered.
        """

        # Limit power to max rating
        power = min(power, self.max_power)

        # Convert power to energy (kWh)
        energy_removed = power * self.dt / self.efficiency

        # Prevent over-discharging
        energy_removed = min(energy_removed, self.energy)

        self.energy -= energy_removed

        return energy_removed / self.dt  # return actual power delivered

    def get_soc(self):
        """
        Return current State of Charge (0 to 1).
        """
        return self.energy / self.capacity
