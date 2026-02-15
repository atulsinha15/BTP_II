import numpy as np
import matplotlib.pyplot as plt

from models.pv_model import PVModel
from models.load_model import LoadModel
from models.battery_model import BatteryModel

# -----------------------------
# Simulation parameters
# -----------------------------
steps = 96
dt = 0.25  # 15 minutes in hours

# -----------------------------
# Initialize models
# -----------------------------
pv = PVModel(capacity=500)
load_model = LoadModel()
battery = BatteryModel(capacity=800, max_power=200, efficiency=0.95, dt=dt)

irradiance = pv.generate_irradiance(steps)
pv_power = pv.compute_power(irradiance)
load = load_model.generate_load(steps)

soc_values = []
grid_import = []
grid_export = []
battery_charge_power = []
battery_discharge_power = []

# Store initial battery energy for validation
initial_battery_energy = battery.energy

# -----------------------------
# Simulation loop
# -----------------------------
for t in range(steps):

    pv_t = pv_power[t]
    load_t = load[t]

    net = pv_t - load_t

    actual_charge_power = 0
    actual_discharge_power = 0

    # -------------------------
    # Surplus case (PV > Load)
    # -------------------------
    if net > 0:

        actual_charge_power = battery.charge(net)
        remaining = net - actual_charge_power

        grid_export.append(max(0, remaining))
        grid_import.append(0)

    # -------------------------
    # Deficit case (Load > PV)
    # -------------------------
    else:

        deficit = abs(net)

        actual_discharge_power = battery.discharge(deficit)
        remaining = deficit - actual_discharge_power

        grid_import.append(max(0, remaining))
        grid_export.append(0)

    battery_charge_power.append(actual_charge_power)
    battery_discharge_power.append(actual_discharge_power)

    soc_values.append(battery.get_soc())

# -----------------------------
# DAILY ENERGY SUMMARY
# -----------------------------
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

# -----------------------------
# ENERGY CONSERVATION CHECK
# -----------------------------
print("\n--- ENERGY CONSERVATION CHECK ---")

lhs = total_pv_energy + total_import_energy
rhs = total_load_energy + total_export_energy + energy_change_battery

print("LHS (PV + Grid Import):", round(lhs, 4))
print("RHS (Load + Export + Battery ΔE):", round(rhs, 4))
print("Energy Difference :", round(lhs - rhs, 6))

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(14,10))

# Power flows
plt.subplot(3,1,1)
plt.plot(pv_power, label="PV Power")
plt.plot(load, label="Load")
plt.plot(grid_import, label="Grid Import")
plt.plot(grid_export, label="Grid Export")
plt.legend()
plt.title("Power Flows (kW)")

# Battery behavior
plt.subplot(3,1,2)
plt.plot(battery_charge_power, label="Battery Charging Power")
plt.plot(battery_discharge_power, label="Battery Discharging Power")
plt.legend()
plt.title("Battery Charge/Discharge (kW)")

# SoC
plt.subplot(3,1,3)
plt.plot(soc_values)
plt.ylim(0,1)
plt.title("Battery State of Charge")

plt.tight_layout()
plt.show()
