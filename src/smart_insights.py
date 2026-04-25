def smart_sales_insights(df):
    revenue = df['TotalPrice'].sum()
    peak_hour = df.groupby('Hour')['TotalPrice'].sum().idxmax()

    return [
        f"Total revenue is {int(revenue)}.",
        f"Peak sales hour is {peak_hour}."
    ]

def smart_customer_insights(rfm):
    loyal = (rfm['Frequency'] > rfm['Frequency'].median()).sum()

    return [
        f"{loyal} customers are highly loyal."
    ]

def smart_forecast_insights(fc):
    trend = fc['yhat'].iloc[-1] - fc['yhat'].iloc[0]
    return ["Demand increasing" if trend > 0 else "Demand decreasing"]

def smart_churn_insights(churn_df):
    high_risk = (churn_df['Churn_Prob'] > 0.7).sum()

    return [f"{high_risk} customers at high risk"]

def smart_inventory_insights(inv):
    restock = (inv['Reorder'] > 0).sum()

    return [f"{restock} items need restocking"]