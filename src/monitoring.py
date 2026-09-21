def data_quality_check(df):
    return {
        "rows": len(df),
        "missing_values": int(df.isnull().sum().sum()),
        "columns": len(df.columns)
    }