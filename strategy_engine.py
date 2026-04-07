import pandas as pd
import numpy as np

def get_base_lap_time(tire_type):
    if tire_type == 0:
        return 90.0
    elif tire_type == 1:
        return 90.8
    else:  # tire_type == 2
        return 92.0

def recommend_pit(current_lap, tire_type, predicted_laps, base_lap_time):
    # Fuel & Track evolution total gain
    TOTAL_GAIN_PER_LAP = 0.05 
    
    if tire_type == 0:  # Soft
        SPIKE_THRESHOLD = 1.0  
        TOTAL_THRESHOLD = 3.0  
        projected_cliff = 15
    elif tire_type == 1:  # Medium
        SPIKE_THRESHOLD = 0.85 
        TOTAL_THRESHOLD = 4.0  
        projected_cliff = 22
    else:  # Hard
        SPIKE_THRESHOLD = 0.65 
        TOTAL_THRESHOLD = 5.0  
        projected_cliff = 30
        
    decision = "STAY OUT"
    pit_lap = None
    true_drop_at_pit = 0.0
    
    prev_time = None
    accumulated_loss = 0.0
    drops = []

    for i, pred_time in enumerate(predicted_laps):
        total_race_lap = current_lap + i
        
        ideal_time = base_lap_time - (total_race_lap * TOTAL_GAIN_PER_LAP)
        true_drop = pred_time - ideal_time
        drops.append(true_drop)
        accumulated_loss += max(0.0, true_drop)
        
        spike = 0.0
        if prev_time is not None:
            spike = pred_time - prev_time
            
        if decision == "STAY OUT" and (true_drop >= TOTAL_THRESHOLD or spike >= SPIKE_THRESHOLD):
            decision = "PIT NOW"
            pit_lap = total_race_lap
            true_drop_at_pit = true_drop
            
        prev_time = pred_time
        
    avg_degradation_rate = (drops[-1] - drops[0]) / len(drops) if len(drops) > 1 else 0.0
    
    # Phase Detection
    if current_lap <= 5:
        phase = "Stable"
    elif current_lap >= projected_cliff:
        phase = "Cliff - Severe Degradation"
    else:
        phase = "Linear Degradation"
    
    reason = f"Degradation rate = {avg_degradation_rate:+.2f}s/lap. "
    if decision == "PIT NOW":
        reason += f"Threshold breached. "
    else:
        reason += "Tires within optimal window. "
        
    strategy_info = {
        "Decision": decision,
        "Reason": reason,
        "Projected_Cliff": projected_cliff,
        "Phase": phase,
        "Avg_Degradation": avg_degradation_rate,
        "Total_Time_Loss": accumulated_loss,
        "Drop": true_drop_at_pit,
        "Pit_Lap": pit_lap
    }
    
    return strategy_info

def evaluate_undercut_advantage(current_lap, current_tire, track_temp, track_type, model, predicted_laps_current):
    # Simulate fresh mediums over the next 3 laps vs current over next 3 laps
    if len(predicted_laps_current) < 3:
        return 0, "Low"
        
    current_3_laps = sum(predicted_laps_current[:3])
    
    X_alt = pd.DataFrame({
        'Lap_Number': [1, 2, 3],
        'Tire_Type': [1, 1, 1], # Medium
        'Track_Temp': [track_temp] * 3,
        'Track_Type': [track_type] * 3
    })
    fresh_3_laps = sum(model.predict(X_alt))
    
    # The gain is how much faster fresh tires are over the next 3 laps
    undercut_gain = current_3_laps - fresh_3_laps
    
    if undercut_gain > 4.5:
        risk = "High (Very Powerful)"
    elif undercut_gain > 2.0:
        risk = "Medium"
    else:
        risk = "Low"
        
    return undercut_gain, risk

def evaluate_alternative_strategy(current_lap, track_temp, track_type, model):
    # Simulate pitting now to Mediums vs staying out
    laps_ahead = list(range(1, 21)) 
    race_laps = list(range(current_lap, current_lap + 20))
    X_alt = pd.DataFrame({
        'Lap_Number': laps_ahead,
        'Tire_Type': [1] * 20, # Medium
        'Track_Temp': [track_temp] * 20,
        'Track_Type': [track_type] * 20
    })
    
    alt_preds = model.predict(X_alt)
    base_m = get_base_lap_time(1)
    alt_loss = 0.0
    for i, pred in enumerate(alt_preds):
        ideal = base_m - (race_laps[i] * 0.05) # Total gain per lap is 0.05 now
        alt_loss += max(0, pred - ideal)
        
    alt_loss += 22.0 # Pit stop penalty in seconds
    return alt_loss, alt_preds

def calculate_pit_window(current_tire, current_lap, track_temp, track_type, model):
    laps_ahead = list(range(current_lap, current_lap + 20))
    X_future = pd.DataFrame({
        'Lap_Number': laps_ahead,
        'Tire_Type': [current_tire] * 20,
        'Track_Temp': [track_temp] * 20,
        'Track_Type': [track_type] * 20
    })
    
    predicted_laps = model.predict(X_future)
    base_lap_time = get_base_lap_time(current_tire)
    
    # Strategy
    strategy_info = recommend_pit(current_lap, current_tire, predicted_laps, base_lap_time)
    
    # Alternative Strategy / Undercut
    alt_loss, alt_preds = evaluate_alternative_strategy(current_lap, track_temp, track_type, model)
    strategy_info["Alt_Loss"] = alt_loss
    
    undercut_gain, undercut_risk = evaluate_undercut_advantage(current_lap, current_tire, track_temp, track_type, model, predicted_laps)
    strategy_info["Undercut_Gain"] = undercut_gain
    strategy_info["Undercut_Risk"] = undercut_risk
    
    # Confidence Score using Variance of RF Trees
    all_tree_preds = np.array([tree.predict(X_future.values) for tree in model.estimators_])
    std_devs = np.std(all_tree_preds, axis=0)
    avg_std = np.mean(std_devs)
    
    # Max practical variance is usually ~1-2s in this model
    confidence_score = max(0, min(100, 100 - (avg_std * 50)))
    strategy_info["Confidence"] = confidence_score
    strategy_info["Std_Dev"] = avg_std
    
    return strategy_info, predicted_laps, alt_preds

def simulate_full_race_strategy(strategy, track_temp, track_type, model):
    """
    Simulates consecutive tire stints, merging them into a single continuous lap time array.
    strategy parameter expects: [("Soft", 20), ("Medium", 30)] or [(0, 20), (1, 30)]
    """
    tire_map_str = {"Soft": 0, "Medium": 1, "Hard": 2}
    all_lap_times = []
    
    current_race_lap = 1
    
    for item in strategy:
        tire_name, stint_length = item
        tire_type = tire_name if isinstance(tire_name, int) else tire_map_str.get(tire_name, 1)
        
        tire_ages = list(range(1, stint_length + 1))
        X_stint = pd.DataFrame({
            'Lap_Number': tire_ages,
            'Tire_Type': [tire_type] * stint_length,
            'Track_Temp': [track_temp] * stint_length,
            'Track_Type': [track_type] * stint_length
        })
        
        preds = model.predict(X_stint)
        
        for i, pred in enumerate(preds):
            actual_race_lap = current_race_lap + i
            tire_age_model = tire_ages[i]
            
            # Adjust the model's heavy-fuel assumption since we pivot to older laps
            time_offset = (actual_race_lap - tire_age_model) * 0.05
            adjusted_pred = pred - time_offset
            
            # Graphically represent the 22-second pit stop loss immediately on the first lap of subsequent stints
            if current_race_lap > 1 and i == 0:
                adjusted_pred += 22.0
                
            all_lap_times.append(adjusted_pred)
            
        current_race_lap += stint_length
        
    return all_lap_times
