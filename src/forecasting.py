from prophet import Prophet

def forecast(df):
    data = df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
    data.columns = ['ds','y']

    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast