# 🏎️ TyreIQ: AI Tire Strategy Engine

TyreIQ is a professional-grade Formula 1 Strategy Engine and Race Simulator. It combines predictive machine learning techniques (`RandomForestRegressor`) with multi-variable deterministic physics (fuel weight reduction, dynamic track evolution) to produce high-fidelity tire degradation curves and optimal pit-stop mathematical comparisons.

## 🧠 System Architecture & Workflow

The architecture is split into four primary micro-services that handle telemetry pipeline simulation, AI training, computational strategy execution, and interactive visualization. 

### 1. Telemetry Generation Pipeline (`generate_data.py`)
Because raw access to actual F1 live telemetry is restricted, the engine utilizes a custom physics generator mapped to compound properties:
- **Baseline Physics**: Computes the foundational lap speeds based on the tire compound (Soft, Medium, Hard).
- **Compound Degradation Mathematics**: Maps quadratic deterioration physics across three distinct phases: `Stable` -> `Linear Loss` -> `Severe Cliff`. It assigns drastic penalty multipliers out of standard deviation bounds when tires surpass severe cliff thresholds mapped uniquely per compound.
- **Circuit Modifiers**: Encodes circuit friction ratings (e.g., Silverstone's abrasive surface vs. Monaco's smooth surface) multiplying deterioration severity.

### 2. Predictive Model (`model_trainer.py`)
- **Random Forest Framework**: Models nonlinear boundaries to capture extreme drop-offs in performance cliffs across 5,000 continuous samples.
- **Features Trained**: `[Lap_Number, Tire_Type, Track_Temp, Track_Type]` -> `Target: Lap_Time`
- **Variance Evaluation**: Predicts the confidence of an undercut explicitly by pulling the Standard Deviation (`std`) across the individual `estimators_` branches in the tree network. Higher node volatility outputs wider prediction uncertainty.

### 3. The Strategy Compute Engine (`strategy_engine.py`)
- **Mid-Race Tactical Engine**: Operates as a dynamic undercut analyzer. Instantly builds a "Third Curve" mapping out a 22-second pitstop loss and calculates the next immediate 3 laps of fresh compound speed vs current degraded speed to generate the `Undercut Advantage` delta.
- **Pre-Race Planner Engine**: Implements the "Time Machine Fuel Engine". Extrapolates compound speed deeper into a 70-lap simulation. It rectifies the AI's naive "heavy fuel" pacing assumption for lap 45 via mathematically factoring a `race_lap * 0.05s / lap` track evolution deduction, generating scientifically accurate late-race lightning-fast stint times.

### 4. Interactive Dashboard (`app.py`)
Built on `Streamlit` natively interfaced with custom `plotly` templates handling multi-stint charting.
- Fully dynamic layout overriding core Streamlit CSS logic for Light/Dark native switching.
- Generates "Degradation Deltas" charting the variance area between optimal physics bounding lines and AI tracked degradation.

## ⚙️ Installation & Usage

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Or `.\.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Initialize the AI Model
Trigger the synthetic sampling script and synthesize the `.joblib` model:
```bash
python model_trainer.py
```

### 3. Run the Analytics Dashboard
Launch the interface locally via port 8501:
```bash
streamlit run app.py
```
