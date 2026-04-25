import pandas as pd
import numpy as np

# ---------------- SEGMENT TRACKING ----------------
def segment_trend(df, rfm):
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M')
    merged = df.merge(rfm[['Segment']], left_on='Customer ID', right_index=True)
    trend = merged.groupby(['MonthYear','Segment'])['TotalPrice'].sum().reset_index()
    trend['MonthYear'] = trend['MonthYear'].astype(str)
    return trend

# ---------------- COHORT ANALYSIS ----------------
def cohort_analysis(df):
    df['OrderMonth'] = df['InvoiceDate'].dt.to_period('M')
    df['Cohort'] = df.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('M')

    cohort_data = df.groupby(['Cohort','OrderMonth'])['Customer ID'].nunique().reset_index()

    cohort_pivot = cohort_data.pivot(index='Cohort', columns='OrderMonth', values='Customer ID')

    cohort_pivot.index = cohort_pivot.index.astype(str)
    cohort_pivot.columns = cohort_pivot.columns.astype(str)
    return cohort_pivot

# ---------------- RETENTION ----------------
def retention_rate(cohort):
    retention = cohort.divide(cohort.iloc[:,0], axis=0)
    return retention

# ---------------- CLV ADVANCED ----------------
def advanced_clv(rfm):
    rfm['CLV'] = rfm['Monetary'] * rfm['Frequency'] * (1 / (rfm['Recency'] + 1))
    return rfm

# ---------------- RECOMMENDATION ENGINE ----------------
def product_recommendations(df):
    basket = df.groupby(['Invoice','Description'])['Quantity'].sum().unstack().fillna(0)
    corr = basket.corr()

    recs = {}
    for item in corr.columns[:5]:
        recs[item] = list(corr[item].sort_values(ascending=False).iloc[1:4].index)

    return recs

# ---------------- ANOMALY DETECTION ----------------
def detect_anomalies(df):
    daily = df.groupby('InvoiceDate')['TotalPrice'].sum()
    mean = daily.mean()
    std = daily.std()

    anomalies = daily[(daily > mean + 2*std) | (daily < mean - 2*std)]
    return anomalies

# ---------------- SMART INSIGHTS ----------------
def smart_summary(df):
    revenue = df['TotalPrice'].sum()
    avg = df['TotalPrice'].mean()

    if revenue > avg * len(df):
        return "Business performance is strong with consistent revenue generation."
    else:
        return "Revenue is moderate; optimization opportunities exist."