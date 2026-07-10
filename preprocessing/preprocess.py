"""
=============================================================
  GreenEnergy — Smart EV Assistant
  Step 2: Preprocessing — Fixed for real column names
=============================================================
"""

import json
import random
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── System Prompt ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are GreenEnergy, a Smart EV Assistant with expert knowledge of "
    "electric vehicle technology, battery systems, charging infrastructure, "
    "grid optimization, vehicle specifications, performance metrics, costs, "
    "and sustainability. Provide accurate, detailed, and helpful answers "
    "about all aspects of electric vehicles. Always support your answers "
    "with specific data and metrics where available."
)

print("=" * 60)
print("  GreenEnergy — Preprocessing All 4 Datasets")
print("=" * 60)

all_samples = []


# ══════════════════════════════════════════════════════════
#  HELPER — create one training sample
# ══════════════════════════════════════════════════════════
def make_sample(user: str, assistant: str) -> dict:
    return {
        "system":    SYSTEM_PROMPT,
        "user":      user,
        "assistant": assistant
    }


# ══════════════════════════════════════════════════════════
#  DATASET 1 — EV Analytics
#  Real columns: vehicle_id, make, model, year, region,
#  battery_capacity_kwh, battery_health_%, range_km,
#  charging_power_kw, charging_time_hr, charge_cycles,
#  energy_consumption_kwh_per_100km, mileage_km,
#  avg_speed_kmh, max_speed_kmh, acceleration_0_100_kmh_sec,
#  temperature_c, co2_saved_tons, maintenance_cost_usd,
#  insurance_cost_usd, monthly_charging_cost_usd,
#  resale_value_usd
# ══════════════════════════════════════════════════════════
def process_ev_analytics():
    print("\n[1/4] Processing EV Analytics Dataset...")
    path = RAW_DIR / "ev_analytics" / "electric_vehicle_analytics.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    samples = []

    for _, row in df.iterrows():
        make = row.get("make",  "Unknown")
        model = row.get("model", "Unknown")
        year = row.get("year",  "")
        name = f"{year} {make} {model}".strip()

        battery = row.get("battery_capacity_kwh", None)
        health = row.get("battery_health_%", None)
        range_v = row.get("range_km", None)
        charge_pw = row.get("charging_power_kw", None)
        charge_t = row.get("charging_time_hr", None)
        cycles = row.get("charge_cycles", None)
        energy = row.get("energy_consumption_kwh_per_100km", None)
        mileage = row.get("mileage_km", None)
        accel = row.get("acceleration_0_100_kmh_sec", None)
        co2 = row.get("co2_saved_tons", None)
        maint = row.get("maintenance_cost_usd", None)
        monthly = row.get("monthly_charging_cost_usd", None)
        resale = row.get("resale_value_usd", None)
        temp = row.get("temperature_c", None)
        region = row.get("region", None)

        if battery and health:
            samples.append(make_sample(
                user=f"What is the battery status of the {name}?",
                assistant=f"The {name} has a battery capacity of {battery} kWh "
                f"with a current battery health of {health}%. "
                f"{'The battery is in excellent condition.' if float(str(health)) > 80 else 'The battery shows some degradation and may benefit from optimization.'}"
            ))

        if range_v and battery:
            samples.append(make_sample(
                user=f"What is the driving range of the {name}?",
                assistant=f"The {name} offers a driving range of {range_v} km "
                f"on a full charge from its {battery} kWh battery pack. "
                f"Range may vary based on driving conditions, speed, and temperature."
            ))

        if charge_pw and charge_t:
            samples.append(make_sample(
                user=f"How long does it take to charge the {name}?",
                assistant=f"The {name} charges at {charge_pw} kW and takes approximately "
                f"{charge_t} hours for a full charge. "
                f"Fast charging at higher power levels can significantly reduce this time."
            ))

        if energy:
            samples.append(make_sample(
                user=f"How energy efficient is the {name}?",
                assistant=f"The {name} consumes {energy} kWh per 100 km, "
                f"making it {'highly efficient' if float(str(energy)) < 20 else 'moderately efficient'} "
                f"compared to the average EV. Lower consumption means lower running costs and longer range."
            ))

        if co2:
            samples.append(make_sample(
                user=f"How much CO2 does the {name} save compared to a petrol car?",
                assistant=f"The {name} has saved approximately {co2} tons of CO2 emissions "
                f"compared to an equivalent petrol vehicle. "
                f"This represents a significant environmental benefit of switching to electric mobility."
            ))

        if monthly and maint:
            samples.append(make_sample(
                user=f"What are the running costs of the {name}?",
                assistant=f"The {name} has a monthly charging cost of ${monthly} USD "
                f"and maintenance costs of ${maint} USD. "
                f"EVs typically have lower running costs than petrol vehicles "
                f"due to cheaper electricity vs fuel and fewer moving parts requiring maintenance."
            ))

        if resale:
            samples.append(make_sample(
                user=f"What is the resale value of the {name}?",
                assistant=f"The {name} has an estimated resale value of ${resale} USD. "
                f"EV resale values are influenced by battery health, mileage, "
                f"model year, and the availability of newer models with improved range."
            ))

        if cycles:
            samples.append(make_sample(
                user=f"How many charge cycles has the {name} completed?",
                assistant=f"The {name} has completed {cycles} charge cycles. "
                f"Most modern EV batteries are designed to retain over 80% capacity "
                f"after 1000-1500 charge cycles, depending on charging habits and temperature."
            ))

        if accel:
            samples.append(make_sample(
                user=f"How fast does the {name} accelerate?",
                assistant=f"The {name} accelerates from 0 to 100 km/h in {accel} seconds. "
                f"Electric motors deliver instant torque, giving EVs strong acceleration "
                f"performance compared to equivalent combustion engine vehicles."
            ))

        if temp and range_v:
            samples.append(make_sample(
                user=f"How does cold weather affect the {name}'s range?",
                assistant=f"At {temp}°C, the {name} achieves a range of {range_v} km. "
                f"Cold temperatures reduce battery efficiency — typically by 15-30% "
                f"in very cold conditions. Preconditioning the battery while plugged in "
                f"can help maintain optimal range in cold weather."
            ))

    print(f"   Rows: {len(df)} | ✅ Generated {len(samples)} training samples")
    return samples


# ══════════════════════════════════════════════════════════
#  DATASET 2 — EV Specs & Trends (Registration Data)
#  Real columns: model_year, make, model, electric_vehicle_type,
#  electric_range, base_msrp, state, city, county
# ══════════════════════════════════════════════════════════
def process_ev_specs_trends():
    print("\n[2/4] Processing EV Specs & Trends Dataset...")
    path = RAW_DIR / "ev_specs_trends" / "Electric_Vehicle_Data.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(
        " ", "_").str.replace("(", "").str.replace(")", "")

    # Drop rows with no useful info
    df = df.dropna(subset=["make", "model"])
    df = df[df["electric_range"] > 0] if "electric_range" in df.columns else df

    # Remove duplicates — keep unique make+model+year combos
    if "model_year" in df.columns:
        df = df.drop_duplicates(subset=["make", "model", "model_year"])
    else:
        df = df.drop_duplicates(subset=["make", "model"])

    samples = []

    for _, row in df.iterrows():
        make = row.get("make",       "Unknown")
        model = row.get("model",      "Unknown")
        year = row.get("model_year", "")
        name = f"{year} {make} {model}".strip()

        range_v = row.get("electric_range", None)
        msrp = row.get("base_msrp",      None)
        ev_type = row.get("electric_vehicle_type", None)
        state = row.get("state", None)
        city = row.get("city",  None)
        cafv = row.get("clean_alternative_fuel_vehicle_cafv_eligibility", None)

        if range_v and float(str(range_v)) > 0:
            samples.append(make_sample(
                user=f"What is the electric range of the {name}?",
                assistant=f"The {name} has an EPA-rated electric range of {range_v} miles. "
                f"This makes it {'suitable for long distance travel' if float(str(range_v)) > 200 else 'well suited for daily commuting and city driving'}."
            ))

        if ev_type:
            samples.append(make_sample(
                user=f"What type of electric vehicle is the {name}?",
                assistant=f"The {name} is classified as a {ev_type}. "
                f"Battery Electric Vehicles (BEV) run entirely on electricity, "
                f"while Plug-in Hybrid Electric Vehicles (PHEV) combine an electric motor "
                f"with a combustion engine for extended range."
            ))

        if msrp and float(str(msrp)) > 0:
            samples.append(make_sample(
                user=f"What is the base price of the {name}?",
                assistant=f"The {name} has a base MSRP of ${msrp:,.0f} USD. "
                f"After federal tax credits and state incentives, the effective price "
                f"may be significantly lower depending on your location."
            ))

        if cafv:
            samples.append(make_sample(
                user=f"Is the {name} eligible for clean fuel vehicle incentives?",
                assistant=f"The {name} has the following clean alternative fuel vehicle eligibility: "
                f"{cafv}. EV incentives vary by country and region — check your local "
                f"government website for the most up-to-date rebates and tax credits available."
            ))

        if state and city and range_v:
            samples.append(make_sample(
                user=f"Is the {name} a good choice for driving in {city}, {state}?",
                assistant=f"The {name} with its {range_v} mile range is "
                f"{'well suited' if float(str(range_v)) > 150 else 'adequate'} for driving in {city}, {state}. "
                f"Consider the local charging infrastructure and your typical daily mileage "
                f"when evaluating if this range meets your needs."
            ))

    print(
        f"   Rows: {len(df)} unique models | ✅ Generated {len(samples)} training samples")
    return samples


# ══════════════════════════════════════════════════════════
#  DATASET 3 — EV Grid Optimization
#  Real columns: timestamp, station_id, location,
#  charging_type, num_chargers, voltage_level, current_flow,
#  power_consumed, power_loss, voltage_fluctuation,
#  battery_capacity, charging_time, charging_power,
#  charging_cost, predicted_power_demand,
#  optimized_charging_power, grid_stability_score,
#  reduced_power_loss_category, voltage_stability_category
# ══════════════════════════════════════════════════════════
def process_ev_grid():
    print("\n[3/4] Processing EV Grid Optimization Dataset...")
    path = RAW_DIR / "ev_grid" / "EV_Charging_Grid_Optimization_Categorical.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    samples = []

    for _, row in df.iterrows():
        station = row.get("station_id",              "Unknown station")
        location = row.get("location",                None)
        charge_type = row.get("charging_type",           None)
        num_chargers = row.get("num_chargers",            None)
        voltage = row.get("voltage_level",           None)
        power = row.get("power_consumed",          None)
        power_loss = row.get("power_loss",              None)
        opt_power = row.get("optimized_charging_power", None)
        grid_score = row.get("grid_stability_score",    None)
        charge_cost = row.get("charging_cost",           None)
        pred_demand = row.get("predicted_power_demand",  None)
        volt_stab = row.get("voltage_stability_category", None)
        loss_cat = row.get("reduced_power_loss_category", None)
        charge_time = row.get("charging_time",           None)
        bat_cap = row.get("battery_capacity",        None)

        if station and grid_score:
            samples.append(make_sample(
                user=f"What is the grid stability at charging station {station}?",
                assistant=f"Charging station {station} has a grid stability score of {grid_score}. "
                f"{'The grid is highly stable at this location.' if float(str(grid_score)) > 0.8 else 'Some instability detected — smart charging is recommended to optimize load.'}"
            ))

        if station and opt_power and power:
            samples.append(make_sample(
                user=f"How can charging be optimized at station {station}?",
                assistant=f"At station {station}, current power consumption is {power} kW. "
                f"The AI-optimized charging power recommendation is {opt_power} kW, "
                f"which reduces grid stress while maintaining efficient charging speed."
            ))

        if charge_type and voltage:
            samples.append(make_sample(
                user=f"What charging technology does station {station} use?",
                assistant=f"Station {station} uses {charge_type} charging at {voltage} voltage level. "
                f"This configuration supports {'fast DC charging for rapid top-ups' if 'dc' in str(charge_type).lower() else 'standard AC charging suitable for overnight or workplace charging'}."
            ))

        if power_loss and loss_cat:
            samples.append(make_sample(
                user=f"What are the power losses at station {station}?",
                assistant=f"Station {station} experiences a power loss of {power_loss} kW, "
                f"categorized as {loss_cat}. "
                f"Minimizing power loss through smart grid management improves "
                f"overall charging efficiency and reduces electricity costs."
            ))

        if charge_cost and charge_time:
            samples.append(make_sample(
                user=f"How much does it cost to charge at station {station}?",
                assistant=f"Charging at station {station} costs {charge_cost} USD "
                f"for a session of {charge_time} hours. "
                f"Charging costs vary by time of day, location, and charging speed. "
                f"Off-peak charging is typically cheaper and better for grid stability."
            ))

        if num_chargers and location:
            samples.append(make_sample(
                user=f"How many chargers are available at the {location} station?",
                assistant=f"The charging station at {location} has {num_chargers} chargers available. "
                f"Multiple chargers reduce wait times and support higher throughput "
                f"for EV fleets and public charging needs."
            ))

        if pred_demand and opt_power:
            samples.append(make_sample(
                user=f"What is the predicted power demand at station {station}?",
                assistant=f"The predicted power demand at station {station} is {pred_demand} kW. "
                f"Based on this forecast, the optimized charging power is set to {opt_power} kW "
                f"to balance demand with grid capacity and minimize peak load stress."
            ))

    # Add domain knowledge Q&A about grid and V2G topics
    domain_qa = [
        ("What is Vehicle-to-Grid (V2G) technology?",
         "Vehicle-to-Grid (V2G) technology enables electric vehicles to send stored energy "
         "back to the power grid during peak demand periods. EVs act as distributed energy "
         "storage units, helping stabilize the grid, reduce electricity costs for owners, "
         "and support the integration of renewable energy sources like solar and wind. "
         "V2G systems communicate with grid operators in real time to decide when to "
         "charge, discharge, or hold energy."),

        ("How does smart charging reduce electricity costs for EV owners?",
         "Smart charging systems automatically schedule EV charging during off-peak hours "
         "when electricity rates are lowest — typically late at night or early morning. "
         "By avoiding peak demand periods, owners can reduce charging costs by 30-60% "
         "compared to unmanaged charging. Smart chargers also integrate with renewable "
         "energy sources to maximize the use of clean, cheap electricity."),

        ("What is the impact of many EVs charging simultaneously on the power grid?",
         "When many EVs charge simultaneously — especially during evening peak hours — "
         "it creates demand spikes that can overload local transformers and distribution networks. "
         "Smart charging and V2G technologies address this by spreading charging load "
         "across time, using EV batteries as grid buffers, and coordinating with "
         "grid operators through demand response programs."),

        ("What is grid stability and why does it matter for EV charging?",
         "Grid stability refers to the power grid's ability to maintain consistent voltage "
         "and frequency despite fluctuating supply and demand. Unstable grids can cause "
         "slow or interrupted charging sessions and damage EV charging equipment. "
         "AI-based optimization systems monitor grid stability in real time and adjust "
         "charging power to prevent instability while keeping EVs charged efficiently."),

        ("How does optimized EV charging routing work?",
         "Optimized EV charging routing uses real-time data about battery state, "
         "charging station availability, charging speeds, electricity prices, and "
         "traffic conditions to calculate the most efficient route for an EV trip. "
         "Advanced algorithms minimize total travel time and charging cost while "
         "eliminating range anxiety by ensuring the vehicle always reaches the "
         "next charging point with sufficient battery remaining.")
    ]

    for user_q, assistant_a in domain_qa:
        samples.append(make_sample(user=user_q, assistant=assistant_a))

    print(f"   Rows: {len(df)} | ✅ Generated {len(samples)} training samples")
    return samples


# ══════════════════════════════════════════════════════════
#  DATASET 4 — HuggingFace EV Specs 2025
#  Real columns: brand, model, top_speed_kmh,
#  battery_capacity_kwh, battery_type, number_of_cells,
#  torque_nm, efficiency_wh_per_km, range_km,
#  acceleration_0_100_s, fast_charging_power_kw_dc,
#  fast_charge_port, towing_capacity_kg, cargo_volume_l,
#  seats, drivetrain, segment, car_body_type
# ══════════════════════════════════════════════════════════
def process_ev_huggingface():
    print("\n[4/4] Processing HuggingFace EV Specs 2025 Dataset...")
    path = RAW_DIR / "ev_huggingface" / "ev_specs_2025_train.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    samples = []

    for _, row in df.iterrows():
        brand = row.get("brand", "Unknown")
        model = row.get("model", "Unknown")
        name = f"{brand} {model}".strip()

        top_speed = row.get("top_speed_kmh",            None)
        battery = row.get("battery_capacity_kwh",     None)
        bat_type = row.get("battery_type",             None)
        cells = row.get("number_of_cells",          None)
        torque = row.get("torque_nm",                None)
        efficiency = row.get("efficiency_wh_per_km",     None)
        range_v = row.get("range_km",                 None)
        accel = row.get("acceleration_0_100_s",     None)
        fast_chg = row.get("fast_charging_power_kw_dc", None)
        chg_port = row.get("fast_charge_port",         None)
        towing = row.get("towing_capacity_kg",       None)
        cargo = row.get("cargo_volume_l",           None)
        seats = row.get("seats",                    None)
        drivetrain = row.get("drivetrain",               None)
        segment = row.get("segment",                  None)
        body_type = row.get("car_body_type",            None)

        if range_v and battery:
            eff_km = round(float(str(range_v)) / float(str(battery)), 1)
            samples.append(make_sample(
                user=f"What is the range and efficiency of the {name}?",
                assistant=f"The {name} achieves a range of {range_v} km from its "
                f"{battery} kWh battery, giving it an efficiency of "
                f"{eff_km} km/kWh. "
                f"{'This places it among the most efficient EVs available.' if eff_km > 6 else 'This is competitive efficiency for its class.'}"
            ))

        if bat_type and cells:
            samples.append(make_sample(
                user=f"What type of battery does the {name} use?",
                assistant=f"The {name} uses a {bat_type} battery pack consisting of "
                f"{cells} individual cells. {bat_type} batteries offer "
                f"{'excellent energy density and fast charging capability' if 'nmc' in str(bat_type).lower() or 'nca' in str(bat_type).lower() else 'excellent thermal stability, long cycle life, and enhanced safety'} "
                f"compared to other battery chemistries."
            ))

        if fast_chg and chg_port:
            samples.append(make_sample(
                user=f"What is the fast charging capability of the {name}?",
                assistant=f"The {name} supports DC fast charging at up to {fast_chg} kW "
                f"via {chg_port} connector. At maximum charging speed, "
                f"it can add approximately {round(float(str(fast_chg)) * 0.25, 0):.0f} km "
                f"of range in just 15 minutes at a compatible fast charger."
            ))

        if top_speed and accel:
            samples.append(make_sample(
                user=f"What is the performance of the {name}?",
                assistant=f"The {name} delivers impressive performance with a top speed of "
                f"{top_speed} km/h and acceleration from 0 to 100 km/h in just "
                f"{accel} seconds. Electric motors provide instant torque delivery, "
                f"giving EVs a performance advantage over equivalent combustion vehicles."
            ))

        if torque and drivetrain:
            samples.append(make_sample(
                user=f"What is the torque and drivetrain of the {name}?",
                assistant=f"The {name} produces {torque} Nm of torque with a {drivetrain} drivetrain. "
                f"{'All-wheel drive provides superior traction and handling in all weather conditions.' if 'awd' in str(drivetrain).lower() or 'all' in str(drivetrain).lower() else 'Rear-wheel drive delivers a sporty, dynamic driving experience.'}"
            ))

        if towing and cargo:
            samples.append(make_sample(
                user=f"What is the practicality of the {name} for hauling and storage?",
                assistant=f"The {name} offers a towing capacity of {towing} kg and "
                f"{cargo} litres of cargo volume. "
                f"This makes it {'suitable for towing trailers, caravans, or small boats' if float(str(towing)) > 1500 else 'suitable for everyday cargo needs'} "
                f"while providing generous storage space for passengers' luggage."
            ))

        if seats and segment:
            samples.append(make_sample(
                user=f"What segment does the {name} belong to and how many passengers can it carry?",
                assistant=f"The {name} is a {segment} segment {body_type or 'vehicle'} "
                f"with seating for {seats} passengers. "
                f"This makes it ideal for {'families and group travel' if int(str(seats)) >= 5 else 'individuals and couples or small families'}."
            ))

        if efficiency:
            samples.append(make_sample(
                user=f"How energy efficient is the {name} in Wh per km?",
                assistant=f"The {name} consumes {efficiency} Wh per km of energy. "
                f"{'This is exceptionally efficient, minimizing running costs and maximizing range.' if float(str(efficiency)) < 180 else 'This efficiency is typical for its size and performance class.'} "
                f"Lower Wh/km means lower electricity bills and better environmental impact."
            ))

    print(f"   Rows: {len(df)} | ✅ Generated {len(samples)} training samples")
    return samples


# ══════════════════════════════════════════════════════════
#  COMBINE & SAVE
# ══════════════════════════════════════════════════════════
def combine_and_save(all_samples):
    print(f"\n{'='*60}")
    print(f"  Combining all datasets...")
    print(f"{'='*60}")

    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.9)
    train_data = all_samples[:split_idx]
    val_data = all_samples[split_idx:]

    for filename, data in [
        ("master_dataset.json", all_samples),
        ("train.json",          train_data),
        ("val.json",            val_data)
    ]:
        with open(PROCESSED_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Master dataset : {len(all_samples):,} samples")
    print(f"✅ Training set   : {len(train_data):,} samples")
    print(f"✅ Validation set : {len(val_data):,} samples")
    print(f"\n📂 Saved to: {PROCESSED_DIR}")
    print(f"\n▶  Next step: run training/finetune.py")
    print("=" * 60)

    print("\n📌 Example training samples:")
    for i, s in enumerate(all_samples[:3], 1):
        print(f"\n  Sample {i}:")
        print(f"  USER     : {s['user']}")
        print(f"  ASSISTANT: {s['assistant'][:150]}...")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    s1 = process_ev_analytics()
    s2 = process_ev_specs_trends()
    s3 = process_ev_grid()
    s4 = process_ev_huggingface()

    all_samples = s1 + s2 + s3 + s4
    print(f"\n{'='*60}")
    print(f"  Total from all 4 datasets: {len(all_samples):,} samples")
    print(f"{'='*60}")

    combine_and_save(all_samples)
