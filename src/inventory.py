def inventory_recommendation(forecast, current_stock=1000):
    forecast['Reorder'] = forecast['yhat'] - current_stock
    return forecast[['ds','yhat','Reorder']]