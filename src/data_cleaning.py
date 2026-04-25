import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path, encoding='ISO-8859-1')

    df = df.dropna(subset=['Customer ID'])
    df['Description'] = df['Description'].fillna('Unknown')
    df = df.drop_duplicates()

    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]

    df = df[~df['Invoice'].astype(str).str.startswith('C')]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Customer ID'] = df['Customer ID'].astype(int)

    df['TotalPrice'] = df['Quantity'] * df['Price']

    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Hour'] = df['InvoiceDate'].dt.hour

    return df