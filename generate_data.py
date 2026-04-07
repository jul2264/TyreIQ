import pandas as pd
import numpy as np

def simulate_lap_time(lap, tire, temp, track, add_noise=True):
    # Base offset (Lap 1 time)
    if tire == 0: # Soft
        base_time = 90.0
        a_factor = -0.05 # Initial grip advantage
        b_factor = 0.008 # Quadratic degradation
    elif tire == 1: # Medium
        base_time = 90.8
        a_factor = -0.04
        b_factor = 0.005
    else: # Hard
        base_time = 92.0
        a_factor = -0.03
        b_factor = 0.003
        
    # Adjust wear based on track conditions
    temp_factor = (temp - 30.0) * 0.0001
    
    # Specific Circuit wear factor
    if track == 0: track_factor = 0.001 # Silverstone High wear
    elif track == 1: track_factor = -0.0005 # Monza Low wear
    elif track == 2: track_factor = 0.000 # Monaco Street
    elif track == 3: track_factor = 0.0015 # Singapore Street High Wear
    else: track_factor = 0.0005 # Interlagos Medium wear
    
    total_b = b_factor + temp_factor + track_factor
    if total_b < 0.0001: total_b = 0.0001
    
    # Fuel burn and Track Evolution
    fuel_burn = lap * 0.03
    track_gain = lap * 0.02
    total_lap_gain = fuel_burn + track_gain
    
    # Late-Stage Cliff Penalty
    cliff_penalty = 0.0
    if tire == 0 and lap > 15: # Soft
        cliff_penalty = 0.5 * (lap - 15)
    elif tire == 1 and lap > 22: # Medium
        cliff_penalty = 0.4 * (lap - 22)
    elif tire == 2 and lap > 30: # Hard
        cliff_penalty = 0.3 * (lap - 30)
        
    # Actual lap time using non-linear model + cliff
    lap_time = base_time - total_lap_gain + (a_factor * lap) + (total_b * (lap ** 2)) + cliff_penalty
    
    if add_noise:
        noise = np.random.normal(0, 0.05)
        lap_time += noise
        
    return lap_time

def generate_tire_data(num_samples=5000):
    np.random.seed(42)
    
    # 0: Soft, 1: Medium, 2: Hard
    tire_types = np.random.choice([0, 1, 2], size=num_samples)
    
    # Laps constraint based on tire
    laps = []
    for t in tire_types:
        if t == 0:
            laps.append(np.random.randint(1, 28)) # Soft
        elif t == 1:
            laps.append(np.random.randint(1, 38)) # Medium
        else:
            laps.append(np.random.randint(1, 48)) # Hard
    laps = np.array(laps)
    
    # Track Temp (e.g. 20 to 45 degree Celsius)
    track_temps = np.random.uniform(20.0, 45.0, size=num_samples)
    
    # Track Type (0: Silverstone, 1: Monza, 2: Monaco, 3: Singapore, 4: Interlagos)
    track_types = np.random.choice([0, 1, 2, 3, 4], size=num_samples)
    
    lap_times = []
    for i in range(num_samples):
        tire = tire_types[i]
        lap = laps[i]
        temp = track_temps[i]
        track = track_types[i]
        
        actual_lap_time = simulate_lap_time(lap, tire, temp, track, add_noise=True)
        lap_times.append(actual_lap_time)
        
    df = pd.DataFrame({
        'Lap_Number': laps,
        'Tire_Type': tire_types,
        'Track_Temp': track_temps,
        'Track_Type': track_types,
        'Lap_Time': lap_times
    })
    
    # Ensure correct data types
    df['Lap_Number'] = df['Lap_Number'].astype(int)
    df['Tire_Type'] = df['Tire_Type'].astype(int)
    df['Track_Temp'] = df['Track_Temp'].round(2)
    df['Track_Type'] = df['Track_Type'].astype(int)
    df['Lap_Time'] = df['Lap_Time'].round(3)
    
    df.to_csv('tire_wear_data.csv', index=False)
    print(f"Generated {num_samples} rows of AI tire wear telemetry data into tire_wear_data.csv.")

if __name__ == '__main__':
    generate_tire_data()
