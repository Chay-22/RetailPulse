import pandas as pd

def monthly_sales(df):
    return df.groupby(['Year','Month'])['TotalPrice'].sum().reset_index()

def top_products(df):
    return df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

def country_sales(df):
    return df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)