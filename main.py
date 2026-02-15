import numpy as np
import matplotlib.pyplot as plt

from models.pv_model import PVModel
from models.load_model import LoadModel
from models.battery_model import BatteryModel
from models.kalman_soc import SoCKalmanFilter


# =====================================================
# Simulation Function (Used for Monte Carlo SoC)
# =====================================================
def run_simulation(pv_power, load, steps, dt):

    battery = BatteryModel(capacity=800, max_power=200, efficiency=0.95, dt=dt)
    soc_values = []

    for t in range(steps):

        net = pv_power[t] - load[t]

        if net > 0:
            battery.charge(net)
        else:
            battery.discharge(abs(net))

        soc_values.append(battery.get_soc())

    return soc_values


# =====================================================
# Parameters
# =====================================================
steps = 96
dt = 0.25
num_scenarios = 50


# =====================================================
# Initialize Models
# =====================================================
pv = PVModel(capacity=500)
load_model = LoadModel()

irradiance = pv.generate_irradiance(steps)

# IMPORTANT: Generate load BEFORE Monte Carlo
load = load_model.generate_load(steps)


# =====================================================
# Generate PV Scenarios (Uncertainty)
# =====================================================
pv_scenarios = []

for i in range(num_scenarios):
    pv_scenarios.append(pv.compute_power(irradiance))

pv_scenarios = np.array(pv_scenarios)

pv_mean = np.mean(pv_scenarios, axis=0)
pv_std = np.std(pv_scenarios, axis=0)

pv_upper = pv_mean + 2 * pv_std
pv_lower = pv_mean - 2 * pv_std
pv_lower = np.maximum(0, pv_lower)


# =====================================================
# Monte Carlo SoC Propagation
# =====================================================
soc_scenarios = []

for i in range(num_scenarios):
    soc_path = run_simulation(pv_scenarios[i], load, steps, dt)
    soc_scenarios.append(soc_path)

soc_scenarios = np.array(soc_scenarios)

soc_mean = np.mean(soc_scenarios, axis=0)
soc_std = np.std(soc_scenarios, axis=0)

soc_upper = soc_mean + 2 * soc_std
soc_lower = soc_mean - 2 * soc_std


# =====================================================
# Plot PV Uncertainty
# =====================================================
plt.figure(figsize=(10,6))
plt.plot(pv_mean, label="PV Mean")
plt.fill_between(range(steps), pv_lower, pv_upper,
                 alpha=0.3, label="Confidence Band (±2σ)")
plt.legend()
plt.title("PV Forecast with Uncertainty")
plt.show()


# =====================================================
# Plot SoC Uncertainty
# =====================================================
plt.figure(figsize=(10,6))
plt.plot(soc_mean, label="SoC Mean")
plt.fill_between(range(steps), soc_lower, soc_upper,
                 alpha=0.3, label="SoC Confidence Band (±2σ)")
plt.ylim(0,1)
plt.legend()
plt.title("Battery SoC with Uncertainty Propagation")
plt.show()

# Kalman Filter for SoC Estimation
kf = SoCKalmanFilter()

true_soc = []
measured_soc = []
estimated_soc = []


# =====================================================
# Deterministic Simulation (Using One Scenario)
# =====================================================
pv_power = pv_scenarios[0]
battery = BatteryModel(capacity=800, max_power=200, efficiency=0.95, dt=dt)

soc_values = []
grid_import = []
grid_export = []

initial_battery_energy = battery.energy

for t in range(steps):

    net = pv_power[t] - load[t]

    if net > 0:
        charge_power = battery.charge(net)
        grid_export.append(max(0, net - charge_power))
        grid_import.append(0)
    else:
        deficit = abs(net)
        discharge_power = battery.discharge(deficit)
        grid_import.append(max(0, deficit - discharge_power))
        grid_export.append(0)

    soc_values.append(battery.get_soc())
    # True SoC
    current_soc = battery.get_soc()
    true_soc.append(current_soc)

    # Simulate noisy measurement
    measurement = current_soc + np.random.normal(0, 0.05)
    measurement = max(0, min(1, measurement))
    measured_soc.append(measurement)

    # Kalman filter update
    kf.predict()
    kf.update(measurement)

    estimated_soc.append(kf.get_state())



# =====================================================
# Energy Summary
# =====================================================
print("\n--- DAILY ENERGY SUMMARY ---")

total_pv_energy = sum(pv_power) * dt
total_load_energy = sum(load) * dt
total_import_energy = sum(grid_import) * dt
total_export_energy = sum(grid_export) * dt

final_battery_energy = battery.energy
energy_change_battery = final_battery_energy - initial_battery_energy

print("Total PV Energy (kWh):", round(total_pv_energy, 2))
print("Total Load Energy (kWh):", round(total_load_energy, 2))
print("Total Grid Import Energy (kWh):", round(total_import_energy, 2))
print("Total Grid Export Energy (kWh):", round(total_export_energy, 2))
print("Battery Energy Change (kWh):", round(energy_change_battery, 2))
print("Final Battery SoC:", round(battery.get_soc(), 3))


# =====================================================
# Energy Conservation Check
# =====================================================
print("\n--- ENERGY CONSERVATION CHECK ---")

lhs = total_pv_energy + total_import_energy
rhs = total_load_energy + total_export_energy + energy_change_battery

print("LHS (PV + Grid Import):", round(lhs, 4))
print("RHS (Load + Export + Battery ΔE):", round(rhs, 4))
print("Energy Difference :", round(lhs - rhs, 6))


# =====================================================
# Plot Deterministic Power Flows
# =====================================================
plt.figure(figsize=(12,8))

plt.subplot(2,1,1)
plt.plot(pv_power, label="PV Power")
plt.plot(load, label="Load")
plt.plot(grid_import, label="Grid Import")
plt.plot(grid_export, label="Grid Export")
plt.legend()
plt.title("Power Flows (kW)")

plt.subplot(2,1,2)
plt.plot(soc_values)
plt.ylim(0,1)
plt.title("Battery State of Charge")

plt.tight_layout()
plt.show()

# plot Kalman Filter SoC Estimation

plt.figure(figsize=(10,6))
plt.plot(true_soc, label="True SoC")
plt.plot(measured_soc, label="Measured SoC", alpha=0.5)
plt.plot(estimated_soc, label="Kalman Estimated SoC")
plt.legend()
plt.title("Kalman Filter State Estimation")
plt.show()
