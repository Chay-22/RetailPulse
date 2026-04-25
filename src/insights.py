def sales_insights(df):
    return [
        f"Total revenue: {int(df['TotalPrice'].sum())}",
        f"Top product: {df.groupby('Description')['Quantity'].sum().idxmax()}",
        f"Peak hour: {df.groupby('Hour')['TotalPrice'].sum().idxmax()}"
    ]

def customer_insights(rfm):
    return [
        f"High-value customers: {rfm[rfm['Monetary'] > rfm['Monetary'].median()].shape[0]}",
        f"Average recency: {int(rfm['Recency'].mean())} days"
    ]

def forecast_insights(fc):
    trend = fc['yhat'].iloc[-1] - fc['yhat'].iloc[0]
    return ["Demand increasing" if trend > 0 else "Demand decreasing"]

def churn_insights(churn_df):
    return [
        f"{churn_df[churn_df['Churn_Prob'] > 0.7].shape[0]} customers at high risk"
    ]

def inventory_insights(inv):
    return [
        f"{inv[inv['Reorder'] > 0].shape[0]} items need restocking"
    ]