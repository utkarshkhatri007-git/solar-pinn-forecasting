# Physics-Informed Neural Network for Short-Term Solar Power Generation Forecasting

**Domain:** Deep Learning  
**Project owner:** Utkarsh  
**Forecast horizon:** One hour ahead  
**Target:** Next-hour AC solar generation (kW)  
**Main model:** Physics-informed feed-forward neural network (PINN-style MLP)

> **Important before submission:** The bundled `outputs/` folder is an offline software demonstration generated from the clearly labelled synthetic dataset in `data/demo_synthetic_site_a.csv`. It is provided so the program, plots and saved model can be inspected without internet access. These numbers are **not real-world experimental results** and must not be reported as measured solar-farm performance. Run the real-data command below and replace the demo outputs before submitting the final results section.

## 1. Project summary

This project forecasts the AC power of a 50 kW solar photovoltaic installation one hour ahead. It combines recent inverter output and weather history with future solar geometry, which can be calculated from the timestamp and location. The project is implemented as a reproducible end-to-end pipeline, not just a neural-network definition.

The pipeline includes:

- Download and validation of a public PV generation/weather dataset.
- Hourly data cleaning, time alignment and leakage-safe feature engineering.
- A PINN-style neural network with physics-based regularization.
- A same-architecture DNN ablation without physics regularization.
- Persistence, physics-only and HistGradientBoosting comparison models.
- Chronological training/validation/test split (70% / 15% / 15%).
- Overall and daylight-only evaluation metrics.
- Saved model weights, scaler, metadata, predictions and diagnostic plots.
- Automated tests for target alignment, bounded output and leakage protection.
- Technical report draft, presentation and viva preparation questions.

## 2. Dataset and source

The default source is the **Two-Site Solar PV Inverter & Weather Dataset (Romania)**:

- Repository: <https://github.com/upb-nlp/PV-Generation-Dataset>
- Site A hourly CSV: <https://raw.githubusercontent.com/upb-nlp/PV-Generation-Dataset/main/site_a/site_a_hourly_weather.csv>
- Site B hourly CSV: <https://raw.githubusercontent.com/upb-nlp/PV-Generation-Dataset/main/site_b/site_b_hourly_weather.csv>
- Coverage listed by the repository: Site A, 4 January 2025–30 June 2026; Site B, 1 May 2025–31 May 2026.
- Both installations have a nominal 50 kW AC inverter rating.
- The primary measurement is `active_power_kw`; the file also contains weather estimates and solar geometry.
- The repository states that the dataset is released under CC BY 4.0. Retain the attribution in any submitted copy.

The generation values are inverter records. The historical weather and irradiance columns are Open-Meteo reanalysis estimates, not necessarily readings from sensors installed at the solar array. Both installations are in the same southwestern Romanian climate region. The dataset alone therefore does not establish performance in Bengaluru or other climates.

**Dataset citation:** Linca, N.-R., Rebedea, T.-E., Ruseti, S., and Dascalu, M. (2026). *Two-Site Solar PV Inverter & Weather Dataset (Romania).* GitHub repository. <https://github.com/upb-nlp/PV-Generation-Dataset>.

## 3. Run on Windows

Install Python 3.10–3.12. Extract the project ZIP, then double-click `setup_windows.bat`, or run these commands in PowerShell from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the real-data experiment (internet required for first download):

```powershell
python run_project.py --source real --site site_a
```

To run Site B:

```powershell
python run_project.py --source real --site site_b --output outputs/site_b
```

The downloaded CSV is stored under `data/real/`. The script validates the CSV schema and row count before saving it. If automatic download fails, manually download the Site A or Site B CSV from the links in Section 2 and save it as:

- `data/real/site_a_hourly_weather.csv`, or
- `data/real/site_b_hourly_weather.csv`.

Then re-run the real-data command. You can also pass a custom compatible file:

```powershell
python run_project.py --data "C:\path\to\your_hourly_data.csv" --output outputs/custom
```

Run the offline demonstration (synthetic data, not empirical findings):

```powershell
python run_project.py --source demo
```

## 4. Tests

Run the automated data-pipeline tests:

```powershell
python -m unittest discover -s tests -v
```

The tests verify that capacity factor stays in [0, 1], that the supervised target is aligned one hour ahead, that target/future measurement columns are not used as features, and that mutating the next-hour power label does not change the current forecast-origin feature vector.

## 5. Forecasting protocol and leakage control

At forecast origin `t`, the model predicts `P(t+1)`. Its measurement inputs use records through `t`. The model does **not** use the measured power, irradiance or weather at `t+1`. Future solar elevation, solar azimuth and calendar encodings are permitted because they are known from time and location. This is a rolling one-step-ahead experiment: a new forecast is made each hour after the latest observation becomes available. It is not a one-shot 24-hour forecast.

Rows are chronologically divided into 70% training, 15% validation and 15% testing. The feature scaler is fitted only on training data. A random split is intentionally avoided because it can mix future periods into training.

## 6. Physics-informed model

The target is normalized capacity factor:

\[
 y_{t+1}=P_{AC,t+1}/P_{rated},\qquad P_{rated}=50\,\mathrm{kW}.
\]

The neural network returns a sigmoid-bounded forecast:

\[
 \hat y_{t+1}=\sigma(f_\theta(x_t)),\qquad \hat P_{t+1}=50\hat y_{t+1}.
\]

A simplified future irradiance estimate uses future solar elevation and the most recently observed cloud fraction:

\[
 \widehat G_{t+1}=1000\max(0,\sin\alpha_{t+1})(1-0.75c_t).
\]

An approximate module temperature and PV prior are then formed:

\[
 \widehat T_{cell,t+1}=T_{air,t}+0.03\widehat G_{t+1},
\]

\[
 y^{phys}_{t+1}=\mathrm{clip}\left[0.82\frac{\widehat G_{t+1}}{1000}\left(1-0.004(\widehat T_{cell,t+1}-25)\right),0,1\right].
\]

These are simplified engineering assumptions, not a full single-diode PV simulation. The PINN training objective is:

\[
 \mathcal L=\mathcal L_{data}+\lambda_n\mathcal L_{night}+\lambda_e\mathcal L_{envelope}+\lambda_p\mathcal L_{prior}.
\]

The terms penalize supervised prediction error, generation at night, output above a permissive irradiance-based envelope and disagreement with the approximate PV prior during daylight. The sigmoid constrains the output to the rated-capacity range. A night-time gate sets the final estimate to zero when future solar elevation is at or below 0 degrees.

## 7. Models compared

| Model | Purpose |
|---|---|
| PINN-style MLP | Proposed neural model trained with the supervised and physical regularization terms. |
| Data-driven DNN | Same network architecture with physical regularization disabled; isolates the contribution of the loss constraints. |
| HistGradientBoosting | Non-neural nonlinear machine-learning benchmark. |
| Persistence | Predicts next-hour generation as the latest measured generation. |
| Physics prior | Estimates output from the simplified irradiance/temperature relationship without machine learning. |

A PINN is not assumed to win. If a baseline produces lower held-out error, report that result and discuss why the current physical approximation or neural architecture may not be sufficient.

## 8. Output files

After a successful run, inspect:

- `outputs/metrics.csv` — model comparison metrics.
- `outputs/report_results.md` — auto-generated results interpretation.
- `outputs/test_predictions.csv` — actual and predicted held-out power.
- `outputs/plots/forecast_trace.png` — continuous actual-versus-predicted trace.
- `outputs/plots/prediction_scatter.png` — PINN prediction versus measurement.
- `outputs/plots/training_curve.png` — training and validation convergence.
- `outputs/plots/residuals.png` — PINN residual distribution.
- `outputs/plots/error_by_hour.png` — forecast error by target UTC hour.
- `outputs/model/solar_pinn.pt` — trained PINN weights.
- `outputs/model/feature_scaler.joblib` — train-fitted scaler.
- `outputs/model/hist_gradient_boosting.joblib` — fitted tree model.
- `outputs/model/metadata.json` — data source, features, split counts and parameters.

The ZIP also contains `reports/technical_report_draft.docx`, `presentation/solar_pinn_presentation.pptx` and `reports/viva_questions.md`. The report and results slide have explicit placeholders until you complete a real-data run.

## 9. Suggested demonstration

1. Open the source dataset and identify `active_power_kw` as the target.
2. Show the feature engineering and next-hour label alignment.
3. Run the automated tests to demonstrate the leakage checks.
4. Run `python run_project.py --source real --site site_a`.
5. Open `outputs/metrics.csv` and explain the daylight-only metrics.
6. Show the forecast trace, scatter plot and training curve.
7. Explain which model performed best and why the PINN did or did not improve on the baselines.
8. Discuss the reanalysis-weather limitation and the need for cross-climate validation.

## 10. Responsible reporting

Do not report synthetic demo metrics as real performance. Do not claim “100% accuracy,” live forecasting or validation for Indian solar installations. The current method is a rolling one-hour-ahead historical experiment. Operational deployment would need weather forecasts archived as they were available at the forecast issue time, calibrated PV parameters and validation across independent plants and climates.
