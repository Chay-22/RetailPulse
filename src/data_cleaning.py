import pandas as pd

__all__ = ["REQUIRED_COLUMNS", "clean_retail_data", "load_and_clean_data"]

REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def clean_retail_data(df):
    df = df.copy()

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

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


def load_and_clean_data(path):
    df = pd.read_csv(path, encoding='ISO-8859-1')
    return clean_retail_data(df)
