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
    """Clean retail transaction data with reduced memory overhead."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    # Remove invalid transactions before creating derived columns.
    df.dropna(subset=["Customer ID"], inplace=True)
    df["Description"] = df["Description"].fillna("Unknown")
    df.drop_duplicates(inplace=True)

    df.drop(df.index[df["Quantity"] <= 0], inplace=True)
    df.drop(df.index[df["Price"] <= 0], inplace=True)

    invoice_text = df["Invoice"].astype(str)
    df.drop(df.index[invoice_text.str.startswith("C")], inplace=True)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["Customer ID"] = pd.to_numeric(
        df["Customer ID"],
        errors="coerce",
        downcast="integer",
    )
    df.dropna(subset=["Customer ID"], inplace=True)
    df["Customer ID"] = df["Customer ID"].astype("int32")

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce",
        downcast="integer",
    )

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce",
        downcast="float",
    )

    # Repeated text values are memory-heavy for large transaction datasets.
    for column in ("StockCode", "Description", "Country"):
        df[column] = df[column].astype("category")

    df["TotalPrice"] = df["Quantity"] * df["Price"]

    df["Year"] = df["InvoiceDate"].dt.year.astype("int16")
    df["Month"] = df["InvoiceDate"].dt.month.astype("int8")
    df["Hour"] = df["InvoiceDate"].dt.hour.astype("int8")

    return df


def load_and_clean_data(path):
    df = pd.read_csv(path, encoding='ISO-8859-1')
    return clean_retail_data(df)
