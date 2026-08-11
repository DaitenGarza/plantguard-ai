# PlantGuard AI — Predictive Maintenance Dashboard

A machine-learning dashboard that watches simulated plant sensor data
(bearing temperature, vibration, discharge pressure) for a centrifugal
pump and flags equipment failure risk *before* it trips — the core idea
behind predictive maintenance programs used across petrochemical and
manufacturing plants in Southeast Texas (Formosa Plastics, Invista,
ExxonMobil, DuPont, and similar facilities all run some version of this).

## Why this project

Unplanned equipment failure is one of the most expensive problems a
process plant has — a tripped pump can shut down a whole unit. Most
plants still run heavily on scheduled ("time-based") maintenance or
run-to-failure. Predictive maintenance uses sensor trends to catch
degradation early enough to schedule a repair on your terms instead of
the equipment's.

## What it does

1. **`plantguard/data_sim.py`** — generates realistic synthetic sensor
   data: steady healthy operation, then a slow degradation trend (e.g.
   bearing wear) that accelerates into a spike right before failure.
2. **`plantguard/model.py`** — engineers rolling-window features
   (mean/std/slope) that capture *trend*, not just instantaneous
   readings, and trains two models:
   - an **IsolationForest** (unsupervised) — flags statistically unusual
     readings with no labeled failure data required, which is the
     realistic starting point for most plants.
   - a **RandomForestClassifier** (supervised) — once you have labeled
     failure history, gives a calibrated 0–100% risk score.
3. **`plantguard/app.py`** — a Streamlit dashboard styled like an
   operator console: live sensor trends, a risk score gauge with
   color-coded alerts, and an anomaly log.

## Prerequisites

- **Python 3.10+** (check with `python3 --version`)
- **pip** (comes with Python)
- No paid accounts, API keys, or GPU needed — everything trains in a few
  seconds on CPU using synthetic data generated locally.

## Running it

### Windows (recommended)

Download or clone the repository, open its `plantguard-ai` folder, and
double-click **`start_plantguard.bat`**. The launcher will:

1. Check for Python 3.10 or newer.
2. Install Python 3.12 with Windows Package Manager if Python is missing.
3. Create a project-only virtual environment.
4. Install or update all required packages.
5. Start PlantGuard AI and open `http://localhost:8501` in your browser.

Keep the launcher window open while using the dashboard. Press **Ctrl+C**
in that window when you are finished.

### Manual setup (Windows, macOS, or Linux)

Run these commands from inside the downloaded/cloned `plantguard-ai`
folder:

```bash
python -m pip install -r requirements.txt
cd plantguard
python -m streamlit run app.py
```

On macOS or Linux, use `python3` instead of `python` if needed.

Streamlit will print a local URL (usually `http://localhost:8501`) — open
it in your browser. First run may take a few extra seconds while
scikit-learn trains the models; they're cached after that.

**If `pip install` fails** because of permissions, use:
`pip install -r requirements.txt --user`
or, better, create a virtual environment first:
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).
Drag the **playhead slider** in the sidebar to replay the run and watch
the risk score climb as the simulated bearing degrades.

To just see the model numbers without the UI:

```bash
cd plantguard
python model.py
```

## Skills demonstrated

- Python, pandas, numpy for data engineering and time-series feature work
- scikit-learn: unsupervised (IsolationForest) and supervised
  (RandomForestClassifier) modeling, train/test evaluation, ROC AUC
- Streamlit for a real-time-feeling interactive dashboard/UI
- Domain framing: translating a plant-operations problem (predictive
  maintenance) into a concrete ML pipeline

## Extending this project

- Swap the synthetic generator for a real public dataset (e.g. NASA
  bearing/turbofan degradation datasets) to validate against real physics.
- Add multiple assets/pumps side-by-side on one dashboard.
- Persist alerts to a small SQLite log and add an "acknowledge alert"
  workflow, closer to a real CMMS (computerized maintenance management
  system) integration.
- Replace RandomForest with a simple LSTM to compare sequence models
  against the rolling-feature approach.
