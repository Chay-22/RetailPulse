import streamlit as st
import plotly.express as px
import pandas as pd

from src.data_cleaning import load_and_clean_data
from src.segmentation import (
    create_rfm,
    apply_kmeans,
    calculate_rfm_scores,
    assign_segments,
    segment_actions
)
from src.forecasting import forecast
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

df = load_data()

# Sidebar
st.sidebar.title("📊 RetailPulse")

menu = st.sidebar.radio("Navigation", [
    "Overview","Sales","Customers","Forecast","Churn","Inventory","Cohort"
])

# Filters
st.sidebar.markdown("### 🔍 Filters")

country = st.sidebar.multiselect("Country", df['Country'].unique())
year = st.sidebar.multiselect("Year", df['Year'].unique())

if country:
    df = df[df['Country'].isin(country)]

if year:
    df = df[df['Year'].isin(year)]

# KPI
def show_kpis(df):
    c1,c2,c3 = st.columns(3)
    c1.metric("💰 Revenue", int(df['TotalPrice'].sum()))
    c2.metric("🧾 Orders", df['Invoice'].nunique())
    c3.metric("👥 Customers", df['Customer ID'].nunique())

# PDF
report_text = f"""
Revenue: {int(df['TotalPrice'].sum())}
Customers: {df['Customer ID'].nunique()}
Insights:
{smart_summary(df)}
"""

# ---------------- OVERVIEW ----------------
if menu == "Overview":
    st.title("📊 Business Overview")

    show_kpis(df)

    # Revenue Trend
    monthly = df.groupby(['Year','Month'])['TotalPrice'].sum().reset_index()

    st.subheader("📈 Revenue Trend")
    st.plotly_chart(px.line(monthly, x="Month", y="TotalPrice", color="Year"))

    # Orders Trend
    orders = df.groupby(['Year','Month'])['Invoice'].nunique().reset_index()

    st.subheader("🧾 Orders Trend")
    st.plotly_chart(px.line(orders, x="Month", y="Invoice", color="Year"))

    # Country Contribution
    st.subheader("🌍 Revenue by Country")
    country = df.groupby('Country')['TotalPrice'].sum().nlargest(10).reset_index()
    st.plotly_chart(px.bar(country, x="Country", y="TotalPrice"))

    # Alerts
    st.subheader("🚨 Alerts")
    if df['TotalPrice'].mean() < 100:
        st.error("Low revenue trend detected")
    else:
        st.success("Revenue stable")

    #Smart Summary
    st.subheader("🧠 Smart Summary")
    st.info(smart_summary(df))    

# ---------------- SALES ----------------
elif menu == "Sales":
    st.title("📊 Sales Analytics")

    show_kpis(df)

    # Top Products
    st.subheader("🏆 Top Products")
    top = df.groupby('Description')['Quantity'].sum().nlargest(10).reset_index()
    st.plotly_chart(px.bar(top, x="Quantity", y="Description", orientation='h'))

    # Hourly Sales
    st.subheader("⏰ Sales by Hour")
    hourly = df.groupby('Hour')['TotalPrice'].sum().reset_index()
    st.plotly_chart(px.line(hourly, x="Hour", y="TotalPrice"))

    # Daily Trend
    st.subheader("📅 Daily Sales Trend")
    daily = df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
    st.plotly_chart(px.line(daily, x="InvoiceDate", y="TotalPrice"))

    # Insights
    st.subheader("💡 Insights")
    st.write("- Identify peak sales hours")
    st.write("- Focus on high-performing products")

    # Anomaly Detection
    st.subheader("🚨 Sales Anomalies")

    anomalies = detect_anomalies(df)

    if not anomalies.empty:
        st.error("Anomalies detected!")
        st.write(anomalies)
    else:
        st.success("No anomalies detected")

    #RECOMMENDATION ENGINE
    st.subheader("🛒 Product Recommendations")

    recs = product_recommendations(df)

    for k, v in recs.items():
        st.write(f"{k} → {', '.join(v)}")    

# ---------------- CUSTOMERS ----------------
elif menu == "Customers":
    st.title("👥 Customer Intelligence Dashboard")

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
        st.sidebar.markdown("### 🎯 Segment Filter")
        selected_segment = st.sidebar.multiselect(
            "Select Segment",
            rfm['Segment'].unique(),
            default=rfm['Segment'].unique()
        )

        rfm_filtered = rfm[rfm['Segment'].isin(selected_segment)]

        # ---------------- KPIs ----------------
        st.subheader("📊 Customer KPIs")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("👥 Customers", rfm_filtered.shape[0])
        c2.metric("💰 Avg CLV", int(rfm_filtered['CLV'].mean()))
        c3.metric("🔁 Avg Frequency", round(rfm_filtered['Frequency'].mean(), 2))
        c4.metric("⚠️ Avg Churn Risk", round(rfm_filtered['Churn_Prob'].mean(), 2))

        # ---------------- SEGMENT DISTRIBUTION ----------------
        st.subheader("📊 Segment Distribution")

        seg_count = rfm_filtered['Segment'].value_counts().reset_index()
        seg_count.columns = ['Segment','Count']

        st.plotly_chart(
            px.pie(seg_count, names='Segment', values='Count'),
            use_container_width=True
        )

        # ---------------- REVENUE BY SEGMENT ----------------
        st.subheader("💰 Revenue by Segment")

        revenue_seg = rfm_filtered.groupby('Segment')['Monetary'].sum().reset_index()

        st.plotly_chart(
            px.bar(revenue_seg, x='Segment', y='Monetary'),
            use_container_width=True
        )

        # ---------------- CLV ANALYSIS ----------------
        st.subheader("💎 Customer Lifetime Value (CLV)")

        st.plotly_chart(
            px.box(rfm_filtered, x='Segment', y='CLV'),
            use_container_width=True
        )

        # ---------------- CHURN ANALYSIS ----------------
        st.subheader("⚠️ Churn Risk by Segment")

        churn_seg = rfm_filtered.groupby('Segment')['Churn_Prob'].mean().reset_index()

        st.plotly_chart(
            px.bar(churn_seg, x='Segment', y='Churn_Prob'),
            use_container_width=True
        )
        #CLV UPGRADE
        rfm = advanced_clv(rfm)

        st.subheader("💰 CLV Distribution")
        st.plotly_chart(px.histogram(rfm, x="CLV"))

        # ---------------- SCATTER ----------------
        st.subheader("📈 Customer Distribution")

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
        st.subheader("🏆 Top Customers")

        top_customers = rfm_filtered.sort_values(by="CLV", ascending=False).head(10)
        st.dataframe(top_customers)

        # ---------------- ACTIONS ----------------
        st.subheader("💡 Recommended Actions")

        for seg in selected_segment:
            st.write(f"**{seg}** → {segment_actions(seg)}")

        # ---------------- PDF EXPORT ----------------
        kpis = {
            "Customers": rfm_filtered.shape[0],
            "Avg CLV": int(rfm_filtered['CLV'].mean()),
            "Avg Churn Risk": round(rfm_filtered['Churn_Prob'].mean(), 2)
        }

        insights = [
            f"{rfm_filtered.shape[0]} customers analyzed",
            "Customer segmentation and churn risk evaluated"
        ]

        recommendations = [
            segment_actions(seg) for seg in selected_segment
        ]

        file = generate_pdf(
            "Customer Intelligence Report",
            kpis,
            insights,
            recommendations
        )

        with open(file, "rb") as f:
            st.download_button(
                "📄 Download Customer Report",
                f,
                file_name="customer_report.pdf"
            )
    # Segment Trend
    

    if len(rfm) >= 2:
        st.subheader("📈 Segment Performance Over Time")

        trend = segment_trend(df, rfm)

        st.plotly_chart(
            px.line(trend, x='MonthYear', y='TotalPrice', color='Segment'),
            use_container_width=True
        )      
        

# ---------------- FORECAST ----------------
elif menu == "Forecast":
    st.title("🔮 Demand Forecasting")

    fc = forecast(df)

    st.plotly_chart(px.line(fc, x="ds", y="yhat"))

    # Trend
    trend = fc['yhat'].iloc[-1] - fc['yhat'].iloc[0]

    st.subheader("📊 Forecast Insights")

    if trend > 0:
        st.success("Demand expected to increase")
    else:
        st.warning("Demand may decline")

    # Variability
    st.subheader("📉 Demand Variability")
    st.metric("Std Dev", round(fc['yhat'].std(), 2))

    # Action
    st.subheader("💡 Recommendation")
    st.write("Adjust inventory based on forecast trend")

# ---------------- CHURN ----------------
elif menu == "Churn":
    st.title("⚠️ Churn Analysis")

    churn_df = churn(df)

    # Top Risk Customers
    st.subheader("🚨 High Risk Customers")
    top = churn_df.sort_values(by="Churn_Prob", ascending=False).head(10)
    st.dataframe(top)

    # Distribution
    st.subheader("📊 Churn Distribution")
    st.plotly_chart(px.histogram(churn_df, x="Churn_Prob", nbins=50))

    # Segment Risk
    st.subheader("📊 Risk Categories")

    churn_df['Risk'] = pd.cut(
        churn_df['Churn_Prob'],
        bins=[0,0.4,0.7,1],
        labels=["Low","Medium","High"]
    )

    risk_count = churn_df['Risk'].value_counts().reset_index()
    risk_count.columns = ['Risk','Count']

    st.plotly_chart(px.pie(risk_count, names='Risk', values='Count'))

    # Actions
    st.subheader("💡 Recommendations")
    st.write("- Target high-risk customers")
    st.write("- Offer retention incentives")

# ---------------- INVENTORY ----------------
elif menu == "Inventory":
    st.title("📦 Inventory Optimization")

    inv = inventory_recommendation(forecast(df))

    # Trend
    st.subheader("📈 Reorder Trend")
    st.plotly_chart(px.line(inv, x="ds", y="Reorder"))

    # Alerts
    st.subheader("🚨 Restock Alerts")
    alerts = inv[inv['Reorder'] > 0]

    if not alerts.empty:
        st.error(f"{len(alerts)} items need restocking")
        st.dataframe(alerts.head(10))
    else:
        st.success("Inventory is balanced")

    # Stats
    st.subheader("📊 Inventory Stats")
    st.metric("Avg Reorder", int(inv['Reorder'].mean()))

    # Recommendation
    st.subheader("💡 Recommendations")
    st.write("- Increase stock for high demand")
    st.write("- Avoid overstocking")

# ---------------- COHORT ----------------

elif menu == "Cohort":
    st.title("📊 Cohort Analysis & Retention")

    cohort = cohort_analysis(df)
    retention = retention_rate(cohort)

    st.subheader("📊 Cohort Table")
    st.dataframe(cohort)

    st.subheader("🔁 Retention Rate")
    st.dataframe(retention.style.background_gradient(cmap='viridis'))    