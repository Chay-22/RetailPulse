import os
import sys

import streamlit as st
import plotly.express as px
import pandas as pd
from io import BytesIO

# Ensure the app root is on sys.path so src package imports resolve correctly.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_cleaning import REQUIRED_COLUMNS, clean_retail_data, load_and_clean_data
from src.segmentation import (
    create_rfm,
    apply_kmeans,
    calculate_rfm_scores,
    assign_segments,
    segment_actions
)
from src.forecasting.forecasting import forecast
from src.churn import churn
from src.inventory import inventory_recommendation
from src.advanced_analytics import *
from src.smart_insights import *
from src.recommendations import *
from src.report import generate_pdf

st.set_page_config(page_title="RetailPulse", layout="wide")

@st.cache_data
def load_data():
    return load_and_clean_data("data/raw/online_retail_II.csv")


@st.cache_data
def load_uploaded_data(file_bytes, file_name):
    if file_name.lower().endswith(".csv"):
        raw_df = pd.read_csv(BytesIO(file_bytes), encoding="ISO-8859-1")
    elif file_name.lower().endswith((".xlsx", ".xls")):
        raw_df = pd.read_excel(BytesIO(file_bytes))
    else:
        raise ValueError("Upload a CSV or Excel file.")

    return clean_retail_data(raw_df)

# Sidebar
st.sidebar.title("RetailPulse")

st.sidebar.markdown("### Data Source")
data_source = st.sidebar.radio(
    "Choose dataset",
    ["Use default dataset", "Upload new dataset"],
)

if data_source == "Upload new dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload retail data",
        type=["csv", "xlsx", "xls"],
    )

    with st.sidebar.expander("Required columns"):
        st.write(", ".join(REQUIRED_COLUMNS))

    if uploaded_file is None:
        st.title("RetailPulse")
        st.info("Upload a CSV or Excel file from the sidebar to start analysis.")
        st.stop()

    try:
        df = load_uploaded_data(
            uploaded_file.getvalue(),
            uploaded_file.name,
        )
        st.sidebar.success(f"Cleaned uploaded dataset: {len(df):,} rows ready.")
    except Exception as exc:
        st.error(f"Could not clean uploaded dataset: {exc}")
        st.stop()
else:
    df = load_data()
    st.sidebar.caption(f"Using default dataset: {len(df):,} cleaned rows")

menu = st.sidebar.radio("Navigation", [
    "Overview","Sales","Customers","Forecast","Churn","Inventory","Cohort"
])

# Filters
st.sidebar.markdown("### Filters")

country = st.sidebar.multiselect("Country", df['Country'].unique())
year = st.sidebar.multiselect("Year", df['Year'].unique())

if country:
    df = df[df['Country'].isin(country)]

if year:
    df = df[df['Year'].isin(year)]

# KPI
def show_kpis(df):
    c1,c2,c3 = st.columns(3)
    c1.metric("Revenue", int(df['TotalPrice'].sum()))
    c2.metric("Orders", df['Invoice'].nunique())
    c3.metric("Customers", df['Customer ID'].nunique())

def normalize_forecast_metrics(metrics):
    normalized = dict(metrics)

    for key in ("R2", "R\u00b2", "R\u00c2\u00b2", "R\u0100\u00b2"):
        if key in normalized:
            normalized["R2"] = normalized[key]
            break

    normalized.setdefault("R2", 0)
    return normalized

# PDF
report_text = f"""
Revenue: {int(df['TotalPrice'].sum())}
Customers: {df['Customer ID'].nunique()}
Insights:
{smart_summary(df)}
"""

# ---------------- OVERVIEW ----------------
if menu == "Overview":
    st.title("Business Overview")

    show_kpis(df)

    # Revenue Trend
    monthly = df.groupby(['Year','Month'])['TotalPrice'].sum().reset_index()

    st.subheader("Revenue Trend")
    st.plotly_chart(px.line(monthly, x="Month", y="TotalPrice", color="Year"))

    # Orders Trend
    orders = df.groupby(['Year','Month'])['Invoice'].nunique().reset_index()

    st.subheader("Orders Trend")
    st.plotly_chart(px.line(orders, x="Month", y="Invoice", color="Year"))

    # Country Contribution
    st.subheader("Revenue by Country")
    country = df.groupby('Country')['TotalPrice'].sum().nlargest(10).reset_index()
    st.plotly_chart(px.bar(country, x="Country", y="TotalPrice"))

    # Alerts
    st.subheader("Alerts")
    if df['TotalPrice'].mean() < 100:
        st.error("Low revenue trend detected")
    else:
        st.success("Revenue stable")

    #Smart Summary
    st.subheader("Smart Summary")
    st.info(smart_summary(df))    

# ---------------- SALES ----------------
elif menu == "Sales":
    st.title("Sales Analytics")

    show_kpis(df)

    # Top Products
    st.subheader("Top Products")
    top = df.groupby('Description')['Quantity'].sum().nlargest(10).reset_index()
    st.plotly_chart(px.bar(top, x="Quantity", y="Description", orientation='h'))

    # Hourly Sales
    st.subheader("Sales by Hour")
    hourly = df.groupby('Hour')['TotalPrice'].sum().reset_index()
    st.plotly_chart(px.line(hourly, x="Hour", y="TotalPrice"))

    # Daily Trend
    st.subheader("Daily Sales Trend")
    daily = df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
    st.plotly_chart(px.line(daily, x="InvoiceDate", y="TotalPrice"))

    # Insights
    st.subheader("Insights")
    st.write("- Identify peak sales hours")
    st.write("- Focus on high-performing products")

    # Anomaly Detection
    st.subheader("Sales Anomalies")

    anomalies = detect_anomalies(df)

    if not anomalies.empty:
        st.error("Anomalies detected!")
        st.write(anomalies)
    else:
        st.success("No anomalies detected")

    #RECOMMENDATION ENGINE
    st.subheader("Product Recommendations")

    recs = product_recommendations(df)

    for k, v in recs.items():
        st.write(f"{k} -> {', '.join(v)}")    

# ---------------- CUSTOMERS ----------------
elif menu == "Customers":
    st.title("Customer Intelligence Dashboard")

    rfm = create_rfm(df)

    if len(rfm) < 2:
        st.warning("Not enough data for analysis. Adjust filters.")
    else:
        # ---------------- RFM + SEGMENTS ----------------
        rfm = calculate_rfm_scores(rfm)
        rfm = assign_segments(rfm)
        rfm = apply_kmeans(rfm)

        # ---------------- CLV (Customer Lifetime Value) ----------------
        rfm['CLV'] = rfm['Monetary'] * rfm['Frequency']

        # ---------------- CHURN ----------------
        churn_df = churn(df)

        # Merge churn into rfm
        rfm = rfm.merge(churn_df[['Churn_Prob']], left_index=True, right_index=True, how='left')

        # ---------------- FILTER ----------------
        st.sidebar.markdown("### Segment Filter")
        selected_segment = st.sidebar.multiselect(
            "Select Segment",
            rfm['Segment'].unique(),
            default=rfm['Segment'].unique()
        )

        rfm_filtered = rfm[rfm['Segment'].isin(selected_segment)]

        # ---------------- KPIs ----------------
        st.subheader("Customer KPIs")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Customers", rfm_filtered.shape[0])
        c2.metric("Avg CLV", int(rfm_filtered['CLV'].mean()))
        c3.metric("Avg Frequency", round(rfm_filtered['Frequency'].mean(), 2))
        c4.metric("Avg Churn Risk", round(rfm_filtered['Churn_Prob'].mean(), 2))

        # ---------------- SEGMENT DISTRIBUTION ----------------
        st.subheader("Segment Distribution")

        seg_count = rfm_filtered['Segment'].value_counts().reset_index()
        seg_count.columns = ['Segment','Count']

        st.plotly_chart(
            px.pie(seg_count, names='Segment', values='Count'),
            use_container_width=True
        )

        # ---------------- REVENUE BY SEGMENT ----------------
        st.subheader("Revenue by Segment")

        revenue_seg = rfm_filtered.groupby('Segment')['Monetary'].sum().reset_index()

        st.plotly_chart(
            px.bar(revenue_seg, x='Segment', y='Monetary'),
            use_container_width=True
        )

        # ---------------- CLV ANALYSIS ----------------
        st.subheader("Customer Lifetime Value (CLV)")

        st.plotly_chart(
            px.box(rfm_filtered, x='Segment', y='CLV'),
            use_container_width=True
        )

        # ---------------- CHURN ANALYSIS ----------------
        st.subheader("Churn Risk by Segment")

        churn_seg = rfm_filtered.groupby('Segment')['Churn_Prob'].mean().reset_index()

        st.plotly_chart(
            px.bar(churn_seg, x='Segment', y='Churn_Prob'),
            use_container_width=True
        )
        #CLV UPGRADE
        rfm = advanced_clv(rfm)

        st.subheader("CLV Distribution")
        st.plotly_chart(px.histogram(rfm, x="CLV"))

        # ---------------- SCATTER ----------------
        st.subheader("Customer Distribution")

        st.plotly_chart(
            px.scatter(
                rfm_filtered,
                x="Recency",
                y="Monetary",
                color="Segment",
                size="CLV"
            ),
            use_container_width=True
        )

        # ---------------- TOP CUSTOMERS ----------------
        st.subheader("Top Customers")

        top_customers = rfm_filtered.sort_values(by="CLV", ascending=False).head(10)
        st.dataframe(top_customers)

        # ---------------- ACTIONS ----------------
        st.subheader("Recommended Actions")

        for seg in selected_segment:
            st.write(f"**{seg}** -> {segment_actions(seg)}")

        # ---------------- PDF EXPORT ----------------

        kpis = {
            "Customers": rfm_filtered.shape[0],
            "Average CLV": round(rfm_filtered["CLV"].mean(), 2),
            "Average Frequency": round(rfm_filtered["Frequency"].mean(), 2),
            "Average Churn Risk": round(rfm_filtered["Churn_Prob"].mean(), 2)
        }

        insights = [
            "Customer segments generated using RFM scoring and KMeans.",
            "Churn risk and customer lifetime value analysis completed."
        ]

        recommendations = [
            "Prioritize retention campaigns for high churn-risk segments.",
            "Use segment-specific actions to improve loyalty and repeat purchases."
        ]

        file = generate_pdf(
            "Customer Report",
            kpis,
            insights,
            recommendations
        )

        with open(file, "rb") as f:
            st.download_button(
                "Download Customer Report",
                f,
                file_name="customer_report.pdf"
            )
    # Segment Trend
    

    if len(rfm) >= 2:
        st.subheader("Segment Performance Over Time")

        trend = segment_trend(df, rfm)

        st.plotly_chart(
            px.line(trend, x='MonthYear', y='TotalPrice', color='Segment'),
            use_container_width=True
        )      
        

# ---------------- FORECAST ----------------
elif menu == "Forecast":

    st.title("AI Demand Forecasting")

    # Production Forecast
    forecast_df, metrics = forecast(
        df,
        horizon=30,
        retrain=False
    )
    metrics = normalize_forecast_metrics(metrics)

    # ---------------- KPIs ----------------

    st.subheader("Forecast Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Forecast Accuracy",
        f"{metrics['Forecast Accuracy']:.2f}%"
    )

    c2.metric(
        "MAPE",
        f"{metrics['MAPE']:.2f}%"
    )

    c3.metric(
        "RMSE",
        f"{metrics['RMSE']:.2f}"
    )

    c4.metric(
        "R2 Score",
        f"{metrics['R2']:.3f}"
    )

    # ---------------- Forecast Chart ----------------

    st.subheader("30-Day Forecast")

    fig = px.line(
        forecast_df,
        x="ds",
        y="Forecast",
        title="Demand Forecast"
    )

    fig.add_scatter(
        x=forecast_df["ds"],
        y=forecast_df["Lower"],
        mode="lines",
        name="Lower CI"
    )

    fig.add_scatter(
        x=forecast_df["ds"],
        y=forecast_df["Upper"],
        mode="lines",
        name="Upper CI"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- Model Comparison ----------------

    st.subheader("Prophet vs LSTM")

    compare = forecast_df[
        ["ds", "Prophet", "LSTM", "Forecast"]
    ]

    st.dataframe(compare)

    # ---------------- Metrics Table ----------------

    st.subheader("Model Metrics")

    metric_df = pd.DataFrame(
        metrics.items(),
        columns=["Metric", "Value"]
    )

    st.dataframe(
        metric_df,
        use_container_width=True
    )

    # ---------------- Insights ----------------

    st.subheader("AI Insights")

    trend = (
        forecast_df["Forecast"].iloc[-1]
        -
        forecast_df["Forecast"].iloc[0]
    )

    if trend > 0:

        st.success(
            "Demand is expected to increase over the next 30 days."
        )

    else:

        st.warning(
            "Demand is expected to decline."
        )

    st.info(
        "Forecast generated using a Prophet + LSTM Ensemble model."
    )

    # ---------------- Recommendations ----------------

    st.subheader("Recommendations")

    st.write(
        "Increase inventory if demand continues upward."
    )

    st.write(
        "Monitor confidence interval for uncertainty."
    )

    st.write(
        "Schedule procurement based on demand trend."
    )

# ---------------- CHURN ----------------
elif menu == "Churn":
    st.title("Churn Analysis")

    churn_df = churn(df)

    # Top Risk Customers
    st.subheader("High Risk Customers")
    top = churn_df.sort_values(by="Churn_Prob", ascending=False).head(10)
    st.dataframe(top)

    # Distribution
    st.subheader("Churn Distribution")
    st.plotly_chart(px.histogram(churn_df, x="Churn_Prob", nbins=50))

    # Segment Risk
    st.subheader("Risk Categories")

    churn_df['Risk'] = pd.cut(
        churn_df['Churn_Prob'],
        bins=[0,0.4,0.7,1],
        labels=["Low","Medium","High"]
    )

    risk_count = churn_df['Risk'].value_counts().reset_index()
    risk_count.columns = ['Risk','Count']

    st.plotly_chart(px.pie(risk_count, names='Risk', values='Count'))

    # Actions
    st.subheader("Recommendations")
    st.write("- Target high-risk customers")
    st.write("- Offer retention incentives")

# ---------------- INVENTORY ----------------
elif menu == "Inventory":

    st.title("AI Inventory Optimization Dashboard")

    # ----------------------------------------------------
    # Forecast
    # ----------------------------------------------------

    forecast_df, metrics = forecast(
        df,
        horizon=30,
        retrain=False
    )

    inv = inventory_recommendation(
        forecast_df,
        current_stock=1000
    )

    # ----------------------------------------------------
    # Inventory KPIs
    # ----------------------------------------------------

    st.subheader("Inventory KPIs")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "EOQ",
        int(inv["EOQ"].iloc[0])
    )

    c2.metric(
        "Safety Stock",
        int(inv["SafetyStock"].iloc[0])
    )

    c3.metric(
        "Reorder Point",
        int(inv["ReorderPoint"].iloc[0])
    )

    c4.metric(
        "Inventory Turnover",
        round(inv["InventoryTurnover"].iloc[0], 2)
    )

    st.divider()

    # ----------------------------------------------------
    # Forecast Trend
    # ----------------------------------------------------

    st.subheader("Forecast Demand")

    fig = px.line(
        inv,
        x="ds",
        y="Forecast",
        title="30-Day Demand Forecast"
    )

    fig.update_traces(line_width=3)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------------------------------
    # ABC Analysis
    # ----------------------------------------------------

    st.subheader("ABC Inventory Classification")

    abc = inv["ABC"].value_counts().reset_index()

    abc.columns = ["Class", "Count"]

    st.plotly_chart(

        px.pie(

            abc,

            names="Class",

            values="Count",

            hole=0.45

        ),

        use_container_width=True

    )

    # ----------------------------------------------------
    # Forecast Distribution
    # ----------------------------------------------------

    st.subheader("Forecast Distribution")

    st.plotly_chart(

        px.histogram(

            inv,

            x="Forecast",

            nbins=20

        ),

        use_container_width=True

    )

    # ----------------------------------------------------
    # Recommendation Table
    # ----------------------------------------------------

    st.subheader("Inventory Recommendation Table")

    st.dataframe(

        inv[
            [
                "ds",
                "Forecast",
                "ABC",
                "CurrentStock",
                "EOQ",
                "SafetyStock",
                "ReorderPoint",
                "Reorder",
                "Recommendation"
            ]
        ],

        use_container_width=True

    )

    # ----------------------------------------------------
    # Restock Alerts
    # ----------------------------------------------------

    st.subheader("Restock Alerts")

    alerts = inv[

        inv["Reorder"] > 0

    ]

    if len(alerts):

        st.error(

            f"{len(alerts)} products require immediate restocking."

        )

        st.dataframe(

            alerts[
                [
                    "ds",
                    "Forecast",
                    "Reorder",
                    "Recommendation"
                ]
            ],

            use_container_width=True

        )

    else:

        st.success(

            "No products require immediate replenishment."

        )

    # ----------------------------------------------------
    # Inventory Statistics
    # ----------------------------------------------------

    st.subheader("Inventory Statistics")

    s1, s2, s3 = st.columns(3)

    s1.metric(

        "Average Forecast",

        round(inv["Forecast"].mean(), 2)

    )

    s2.metric(

        "Maximum Forecast",

        round(inv["Forecast"].max(), 2)

    )

    s3.metric(

        "Minimum Forecast",

        round(inv["Forecast"].min(), 2)

    )

    # ----------------------------------------------------
    # Smart Recommendations
    # ----------------------------------------------------

    st.subheader("AI Inventory Recommendations")

    if inv["ABC"].eq("A").any():

        st.success(
            "Prioritize Category A inventory. These products contribute the highest business value."
        )

    if inv["Reorder"].max() > 0:

        st.warning(
            "Purchase orders should be initiated immediately to avoid stock-outs."
        )

    if inv["InventoryTurnover"].iloc[0] > 12:

        st.info(
            "Inventory turnover is healthy. Maintain the current replenishment cycle."
        )

    else:

        st.info(
            "Inventory turnover is relatively low. Consider reducing excess stock."
        )

    # ----------------------------------------------------
    # PDF Report
    # ----------------------------------------------------

    st.subheader("Inventory Report")

    kpis = {

        "EOQ": int(inv["EOQ"].iloc[0]),

        "Safety Stock": int(inv["SafetyStock"].iloc[0]),

        "Reorder Point": int(inv["ReorderPoint"].iloc[0]),

        "Inventory Turnover": round(
            inv["InventoryTurnover"].iloc[0],
            2
        )

    }

    insights = [

        "Inventory optimized using AI demand forecasting.",

        "ABC classification completed.",

        "EOQ and Safety Stock calculated."

    ]

    recommendations = [

        "Maintain adequate Safety Stock.",

        "Review Category A inventory daily.",

        "Trigger replenishment when stock reaches Reorder Point."

    ]

    file = generate_pdf(

        "Inventory Optimization Report",

        kpis,

        insights,

        recommendations

    )

    with open(file, "rb") as f:

        st.download_button(

            "Download Inventory Report",

            f,

            file_name="inventory_report.pdf"

        )

# ---------------- COHORT ----------------

elif menu == "Cohort":
    st.title("Cohort Analysis & Retention")

    cohort = cohort_analysis(df)
    retention = retention_rate(cohort)

    st.subheader("Cohort Table")
    st.dataframe(cohort)

    st.subheader("Retention Rate")
    st.dataframe(retention.style.background_gradient(cmap='viridis'))    




