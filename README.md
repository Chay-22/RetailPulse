# RetailPulse

### AI-Powered Customer Analytics & Demand Forecasting Platform

RetailPulse is an end-to-end retail analytics platform built with Python and Streamlit. It transforms retail transaction data into actionable business insights through data cleaning, exploratory analytics, customer intelligence, demand forecasting, churn analysis, inventory optimization, recommendations, and report generation.

## 🚀 Live Demo

**Streamlit Application:**  
https://retailpulse-fvkdqccxajwbntdqb5ybdj.streamlit.app/

**GitHub Repository:**  
https://github.com/Chay-22/RetailPulse

---

## 📌 Project Overview

Retail businesses generate large volumes of transaction data, but raw transactional records are difficult to use directly for decision-making.

RetailPulse provides a unified analytics workflow that allows users to:

- Upload their own retail CSV or Excel dataset.
- Automatically clean and validate the data.
- Explore sales and customer behavior.
- Analyze revenue and product performance.
- Segment customers using RFM analysis and clustering.
- Analyze customer churn and retention risk.
- Forecast future demand.
- Generate inventory recommendations.
- Explore advanced retail analytics.
- Generate downloadable reports.

The application also includes a deployment-safe sample dataset so that the public demo can be used without requiring users to upload data first.

---

## 🎯 Business Objectives

RetailPulse is designed to support retail decision-making in four major areas:

### 1. Demand Planning
Use historical sales patterns to estimate future demand and support planning decisions.

### 2. Customer Intelligence
Understand customer purchasing behavior using RFM analysis, customer value metrics, segmentation, and churn analysis.

### 3. Inventory Planning
Use demand forecasts and product-level analytics to support inventory and reorder decisions.

### 4. Business Reporting
Convert analytical results into interactive dashboards and downloadable reports.

---

# ✨ Key Features

| Feature | Description |
|---|---|
| Data Upload | Upload CSV or Excel retail datasets |
| Data Cleaning | Validation, missing-value handling, transaction filtering, and feature preparation |
| Exploratory Analytics | Revenue, orders, products, countries, trends, and other KPIs |
| Customer Analytics | Customer-level purchasing and value analysis |
| RFM Segmentation | Recency, Frequency, and Monetary-value based customer analysis |
| Customer Clustering | K-Means based customer segmentation |
| Churn Analysis | Identification and analysis of customer churn risk |
| Demand Forecasting | Time-series based demand forecasting |
| Inventory Optimization | Forecast-driven inventory and reorder recommendations |
| Product Analytics | Product performance and sales analysis |
| Recommendations | Data-driven retail recommendations |
| Smart Insights | Automated business insights derived from available analytics |
| Cohort Analysis | Customer behavior and retention analysis over time |
| Report Generation | Exportable PDF and spreadsheet-oriented outputs |
| Interactive Dashboard | Streamlit dashboard with navigation, filters, and visualizations |

---

# 🏗️ Application Architecture

RetailPulse follows a modular analytics architecture:

```text
                    ┌─────────────────────────┐
                    │       RetailPulse       │
                    │   Streamlit Dashboard   │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
          Default Sample                    User Dataset
             Dataset                       CSV / Excel
                 │                               │
                 └───────────────┬───────────────┘
                                 ↓
                       Data Validation
                                 ↓
                         Data Cleaning
                                 ↓
                      Feature Engineering
                                 ↓
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ↓                  ↓                  ↓
             EDA          Customer Analytics    Forecasting
              │                  │                  │
              │          ┌───────┴────────┐         ↓
              │          ↓                ↓    Inventory
              │        RFM          Segmentation Optimization
              │          │                │
              │          └───────┬────────┘
              │                  ↓
              │              Churn
              │
              └──────────────────┬──────────────────┐
                                 ↓
                       Interactive Dashboard
                                 ↓
                       Insights & Recommendations
                                 ↓
                         Reports / Downloads