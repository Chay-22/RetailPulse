# RetailPulse

RetailPulse is an enterprise-grade retail analytics platform built on Streamlit and Python. It converts raw retail transaction records into business intelligence dashboards, customer segmentation, demand forecasting, churn analytics, inventory recommendations, and exportable reports.

## Overview

RetailPulse enables retail analysts and business users to:

- clean and validate raw retail transaction files,
- explore sales and customer behavior through interactive dashboards,
- segment customers using RFM and clustering,
- forecast demand and revenue,
- identify churn risk and retention opportunities,
- generate inventory recommendations,
- export analytics summaries via PDF.

## What This Project Includes

- `app.py`: Streamlit dashboard entry point with navigation, filters, and page layout.
- `src/`: Core business logic modules for data cleaning, analytics, forecasting, and reporting.
- `data/raw/`: Source dataset storage.
- `data/processed/`: Cleaned and processed dataset output location.
- `requirements.txt`: Python dependency manifest.
- `streamlit/config.toml`: Streamlit configuration.
- `models/`: Saved trained model files.

## Architecture

RetailPulse is structured into the following layers:

1. **Data ingestion and validation**
   - `src/data_cleaning.py` validates required fields, removes invalid transactions, fills missing descriptions, and derives analytics columns such as `TotalPrice`, `Year`, `Month`, and `Hour`.
2. **Dashboard frontend**
   - `app.py` defines the Streamlit user interface, sidebar controls, data upload flow, and navigation between analytics pages.
3. **Analytics modules**
   - `src/segmentation.py`: RFM score computation, segment assignment, and k-means clustering.
   - `src/churn.py`: Customer churn and retention analytics.
   - `src/inventory.py`: Inventory recommendation generation.
   - `src/advanced_analytics.py`: Additional KPI computation and analytics utilities.
   - `src/smart_insights.py`: Automated insight generation based on dataset patterns.
   - `src/recommendations.py`: Product recommendation logic.
4. **Forecasting engine**
   - `src/forecasting/`: Time series model training, evaluation, and prediction using Prophet, LSTM, and ensemble approaches.
5. **Reporting**
   - `src/report.py`: PDF report creation and export.

## Requirements

RetailPulse requires Python 3.10+ and the packages listed in `requirements.txt`. Key dependency groups are:

- Data processing: `pandas`, `numpy`, `scipy`, `pyarrow`
- Machine learning: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- Forecasting: `prophet`, `statsmodels`
- Deep learning: `tensorflow`, `keras`
- Dashboard: `streamlit`, `streamlit-option-menu`, `streamlit-aggrid`, `streamlit-extras`
- Reporting: `reportlab`, `openpyxl`, `xlsxwriter`, `fpdf2`
- Utilities: `pydantic`, `rich`, `tqdm`, `requests`, `python-dateutil`, `cloudpickle`
- Optional AI: `sentence-transformers`, `transformers`, `torch`
- Dev and testing: `pytest`, `pytest-cov`, `black`, `isort`, `flake8`, `mypy`

## Installation

1. Clone the repository and open the project root:

```bash
git clone <repo-url>
cd RetailPulse
```

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install project dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Launch the dashboard from the project root:

```bash
streamlit run app.py
```

If port `8501` is in use, Streamlit will automatically select the next available port. To explicitly force port `8501`:

```bash
streamlit run app.py --server.port 8501
```

Open the local URL shown in the terminal to access RetailPulse.

## Data Requirements

RetailPulse ships with a default dataset at `data/raw/online_retail_II.csv`.

When uploading a custom dataset, the file must contain all of the following columns exactly:

- `Invoice`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `Price`
- `Customer ID`
- `Country`

For CSV uploads, RetailPulse uses `ISO-8859-1` encoding by default.

## Features and Navigation

The Streamlit app contains the following main dashboard pages:

- **Overview**: Business KPIs, revenue trends, orders trends, country revenue contribution, and smart summaries.
- **Sales**: Top products, hourly sales, daily trend, anomaly detection, and recommendations.
- **Customers**: Customer intelligence with RFM segmentation, CLV, churn risk, and segment analytics.
- **Forecast**: Time series forecasting and revenue projections.
- **Churn**: Churn detection and retention-focused insights.
- **Inventory**: Inventory recommendations and demand planning.
- **Cohort**: Cohort analysis for customer behavior over time.

## File Structure

```text
RetailPulse/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_OVERVIEW.md
├── data/
│   ├── processed/
│   │   └── cleaned_data.csv
│   └── raw/
│       └── online_retail_II.csv
├── models/
│   └── lstm_model.pth
├── src/
│   ├── advanced_analytics.py
│   ├── churn.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── gpt_insights.py
│   ├── insights.py
│   ├── inventory.py
│   ├── monitoring.py
│   ├── recommendations.py
│   ├── report.py
│   ├── segmentation.py
│   ├── smart_insights.py
│   ├── forecasting/
│   │   ├── dataset.py
│   │   ├── ensemble.py
│   │   ├── forecasting.py
│   │   ├── lstm_model.py
│   │   ├── metrics.py
│   │   ├── predictor.py
│   │   ├── prophet_model.py
│   │   ├── trainer.py
│   │   ├── utils.py
│   │   └── __init__.py
│   └── __init__.py
└── streamlit/
    └── config.toml
```

## Development Notes

- Always run the app from the repository root so `src` imports resolve correctly.
- `app.py` modifies `sys.path` to ensure the package imports from `src` are available.
- Use `pytest` to validate new analytic code or business logic changes.
- Keep the `data/raw` dataset unchanged and save cleaned outputs to `data/processed`.

## Troubleshooting

- **Import errors from `src`**: verify current working directory is the project root and the virtual environment is active.
- **Port conflict on `8501`**: either stop the existing Streamlit server or start with `--server.port <port>`.
- **Missing required columns**: ensure uploaded files have exactly the column names listed above.
- **Encoding issues on CSV uploads**: use UTF-8 or `ISO-8859-1` encoded files.

## Recommended Workflow

1. Activate the virtual environment.
2. Install dependencies.
3. Run `streamlit run app.py`.
4. Use the sidebar to choose the default dataset or upload a dataset.
5. Apply filters and analyze pages.
6. Export reports after reviewing insights.

## License

This repository does not include a license. Add a `LICENSE` file if you intend to distribute or publish the project.
