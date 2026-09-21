# Project Overview: RetailPulse

## Overview

RetailPulse is a complete retail analytics solution built using Streamlit and Python. It transforms raw retail transaction data into dashboards, business insights, customer segmentation, forecasting, churn analysis, inventory recommendations, and exportable reports.

This project is designed for retail operations, analysts, and business intelligence teams that need a modular analytics platform with fast data exploration and automation.

## Core Objectives

- Provide a production-ready retail analytics dashboard with modern visualization.
- Clean, validate, and standardize retail transaction data.
- Deliver customer-level segmentation and churn intelligence.
- Support forecasting for revenue and inventory planning.
- Allow business users to upload new datasets without changing code.
- Export results and insights in shareable report formats.

## Architecture

RetailPulse follows a modular architecture with distinct responsibilities:

- **Presentation layer**
  - `app.py` implements the Streamlit UI, navigation, sidebar controls, and page workflows.
- **Data layer**
  - `src/data_cleaning.py` loads and sanitizes raw retail data, adds feature columns, and enforces schema rules.
- **Analytics layer**
  - `src/segmentation.py` builds RFM segmentation, calculates customer lifetime value, and applies clustering.
  - `src/churn.py` computes churn risk and retention indicators.
  - `src/inventory.py` generates inventory recommendations from sales data.
  - `src/recommendations.py` provides product pairings and recommendations.
  - `src/smart_insights.py` generates automated narrative summaries of dataset patterns.
  - `src/advanced_analytics.py` contains additional metrics and analytics utilities.
- **Forecasting layer**
  - `src/forecasting/` contains time-series modeling, training, prediction, and evaluation code.
- **Reporting layer**
  - `src/report.py` builds PDF reports from dashboard insights.

## Detailed Module Breakdown

### `app.py`

- Main Streamlit app entry point.
- Configures page layout and caching.
- Loads either the default `online_retail_II.csv` dataset or uploaded data.
- Renders dashboard pages for Overview, Sales, Customers, Forecast, Churn, Inventory, and Cohort.
- Handles user input filters for country and year.

### `src/data_cleaning.py`

- Defines the required retail schema.
- Validates uploaded and default datasets.
- Drops invalid or canceled transactions.
- Fills missing product descriptions.
- Converts dates and numeric values.
- Creates derived fields: `TotalPrice`, `Year`, `Month`, and `Hour`.

### `src/segmentation.py`

- Builds Recency, Frequency, Monetary (RFM) customer data.
- Calculates RFM scores.
- Assigns customer segments and KMeans clusters.
- Supports customer intelligence views in the dashboard.

### `src/churn.py`

- Computes churn-related metrics.
- Identifies customers with high churn risk.
- Integrates churn insights into the Customers and Churn pages.

### `src/inventory.py`

- Produces inventory recommendations from historical sales.
- Supports demand planning and restocking guidance.

### `src/recommendations.py`

- Generates product recommendation mappings.
- Supports cross-sell analytics in the Sales page.

### `src/smart_insights.py`

- Produces narrative insights for business users.
- Summarizes revenue, customer, and trend signals.

### `src/report.py`

- Builds PDF reports from dashboard metrics.
- Enables export and sharing of business insights.

### `src/forecasting/`

- Contains end-to-end forecasting functionality.
- Supports Prophet, ensemble, and LSTM workflows.
- Includes dataset preparation, training, evaluation, and prediction utilities.

## Data Sources

- Default dataset: `data/raw/online_retail_II.csv`
- Cleaned output path: `data/processed/cleaned_data.csv`

### Required input schema for uploads

The upload pipeline requires the following exact column names:

- `Invoice`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `Price`
- `Customer ID`
- `Country`

Confirmed schema enforcement helps avoid runtime failures and ensures the analytics pipeline works consistently.

## Dependencies

This project manages dependencies through `requirements.txt`. The current key dependency groups are:

- **Dashboard**: `streamlit`, `streamlit-option-menu`, `streamlit-aggrid`, `streamlit-extras`
- **Data processing**: `pandas`, `numpy`, `scipy`, `pyarrow`
- **Machine learning**: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Forecasting**: `prophet`, `statsmodels`
- **Deep learning**: `tensorflow`, `keras`
- **Reporting**: `reportlab`, `openpyxl`, `xlsxwriter`, `fpdf2`
- **Utilities**: `pydantic`, `rich`, `tqdm`, `requests`, `python-dateutil`, `cloudpickle`
- **Optional AI**: `sentence-transformers`, `transformers`, `torch`
- **Dev/test**: `pytest`, `pytest-cov`, `black`, `isort`, `flake8`, `mypy`

## Running the Project

1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. Open the URL displayed in the terminal.

## User Workflow

1. Launch the app from the repository root.
2. Choose the default dataset or upload a compatible CSV/Excel file.
3. Use the sidebar to filter by country and year.
4. Navigate through Overview, Sales, Customers, Forecast, Churn, Inventory, and Cohort.
5. Review insights and export reports as needed.

## Troubleshooting

- **Port conflict**: If `8501` is occupied, use `streamlit run app.py --server.port 8501` or another port.
- **Import issues**: Make sure `app.py` runs from the project root and that `src` is importable.
- **Schema errors**: Verify that uploaded files contain all required columns.
- **Encoding errors**: Use UTF-8 or ISO-8859-1 encoding for CSV uploads.

## Notes

- `app.py` inserts the project root into `sys.path` to ensure the `src` package resolves correctly.
- Keep the raw dataset unchanged and generate processed data under `data/processed`.
- The architecture supports modular addition of new analytics pages and model pipelines.
