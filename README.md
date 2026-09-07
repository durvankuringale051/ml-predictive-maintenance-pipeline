# ML Predictive Maintenance Pipeline

A real-time, end-to-end predictive maintenance pipeline built around a **simulated industrial digital twin**. Bearing vibration data is generated in Siemens SIMIT, run through PLCSIM Advanced, streamed over OPC UA, scored by two machine learning models, and visualized live in Grafana.

This project bridges an industrial controls background (Siemens PLCs, OPC UA, PROFINET) with applied machine learning — the goal is a pipeline that could plausibly sit on a real production line, not just a notebook demo.

---

## Architecture

```
SIMIT (digital twin) → PLCSIM Advanced → OPC UA Server
                                              │
                                              ▼
                                     opcua_client.py
                                    (subscribes to signals)
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
                 Anomaly_model (per machine)              InfluxDB (time-series storage)
              ┌─────────────────────────┐                        │
              │ 1. Vibration regressor  │                        ▼
              │ 2. Degradation classify │                    Grafana
              └─────────────────────────┘                (live dashboards)
```

Each machine gets its own `Anomaly_model` instance, so rolling state (RPM history, exceedance history, anomaly counts) never mixes between machines running in the same pipeline.

---

## What it does

**Model 1 — Healthy vibration prediction**
A Random Forest regressor trained on healthy-only (0% wear) data predicts the *expected* vibration for the current RPM and Load. The model is RPM-conditioned to account for a resonance peak around ~1750 RPM in the RPM–vibration characteristic.

The residual between predicted and actual vibration is compared against a **per-RPM-bin threshold** (95th percentile of healthy-data residuals, per bin) rather than a single global threshold — this accounts for uneven residual variance across the operating range. An anomaly is only flagged once **5 of the last 8 samples** exceed threshold, to avoid triggering on single noisy readings.

**Model 2 — Degradation state classification**
A second Random Forest classifier takes the running anomaly count from Model 1 and classifies the machine into a discrete degradation state (e.g. Healthy / Degrading / Severe).

**Not yet included: RUL regression.** An earlier version attempted full-range (0–100%) Remaining Useful Life regression and found it inaccurate across the healthy operating range. This was descoped rather than shipped in a broken state — see [Roadmap](#roadmap).

---

## Repository structure

```
ML Pipeline/
├── Config/
│   ├── opcua_config.json      # endpoint, subscription settings, plant/machine/signal config, model paths
│   └── .env                   # InfluxDB credentials (gitignored — see .env.example)
├── data_pipeline/
│   └── influxdb_dp.py         # InfluxDB write client
├── ML_model/
│   ├── model.py                # Anomaly_model — both ML models + feature logic
│   ├── rf_model.pkl            # trained vibration regressor
│   └── state_rf_model.pkl      # trained degradation-state classifier
├── OPC client/
│   └── opcua_client.py         # main entrypoint — OPC UA subscription + orchestration
├── requirements.txt
├── .env.example
└── .pylintrc
```

---

## Setup

1. **Clone the repo**
   ```
   git clone https://github.com/durvankuringale051/ml-predictive-maintenance-pipeline.git
   cd ml-predictive-maintenance-pipeline
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Configure InfluxDB credentials**
   Copy `.env.example` to `Config/.env` and fill in your real values:
   ```
   INFLUXDB_URL=http://localhost:8086
   INFLUXDB_BUCKET=your_bucket_name
   INFLUXDB_ORG=your_org_name
   INFLUXDB_TOKEN=your_influxdb_api_token
   ```
   `Config/.env` is gitignored — never commit real credentials.

4. **Configure the OPC UA endpoint**
   Edit `Config/opcua_config.json` to point at your OPC UA server (SIMIT/PLCSIM Advanced, or a real PLC) and confirm the machine/signal definitions match your setup.

5. **Run**
   ```
   python "OPC client/opcua_client.py"
   ```
   The client subscribes to configured signals, scores each machine independently, writes results to InfluxDB, and reconnects automatically (with exponential backoff) if the OPC UA connection drops.

---

## Engineering notes

A few real production issues found and fixed during development, kept here because the debugging is arguably the more interesting part of this project than the final dashboard:

- **Per-machine model isolation** — an early version shared a single `Anomaly_model` instance across all machines, silently mixing one machine's rolling RPM/anomaly history into another's predictions. Fixed by keying model instances per machine.
- **Non-blocking inference** — sklearn `.predict()` calls are synchronous and were blocking the asyncio event loop during live subscription handling; moved to `loop.run_in_executor()`.
- **Logging collision** — multiple modules calling `logging.basicConfig()` meant only the *first* call (from an imported module, not the entrypoint) took effect, silently dropping the intended rotating file handler.
- **RPM-conditioned thresholding** — a single global anomaly threshold produced false positives clustered around a resonance peak (~1750 RPM); replaced with per-RPM-bin thresholds tuned from k=3 → k=5 (median + k·MAD), cutting false positives at 0–20% wear from ~18% to ~10% while retaining ~98% detection at 80–100% wear.
- **RUL scope correction** — full-range RUL regression was found to be unreliable in the healthy operating range; rather than ship an inaccurate model, RUL prediction was descoped pending a redesign (see Roadmap).

---

## Results

*Formal evaluation metrics (accuracy, F1, confusion matrix) for the degradation-state classifier are in progress — to be added here.*

---

## Roadmap

- [ ] Formal evaluation metrics for the degradation-state classifier (accuracy, precision/recall, confusion matrix)
- [ ] Feature-importance / explainability output for both models
- [ ] RUL regressor, redesigned to activate only after the anomaly detector triggers (empirically ~60% remaining life), using a CNN or RNN architecture on windowed sensor data
- [ ] Cost-sensitive evaluation (false negatives — missed failures — weighted more heavily than false positives)

---

## Tech stack

Python · asyncua (OPC UA) · scikit-learn · pandas · InfluxDB · Grafana · Siemens SIMIT / PLCSIM Advanced
