"""
advanced_analytics.py
----------------------------------------
Advanced Analytics Module for RetailPulse.

Provides:
- Segment trend analysis
- Cohort analysis
- Retention analysis
- Advanced CLV
- Memory-safe product recommendations
- Anomaly detection
- Smart business summaries

The module is designed to work with large retail transaction
datasets while minimizing unnecessary DataFrame copies and
large dense intermediate matrices.
"""

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd


# ============================================================
# SEGMENT TRACKING
# ============================================================

def segment_trend(df, rfm):
    """
    Calculate monthly revenue trends by customer segment.

    Parameters
    ----------
    df : pandas.DataFrame
        Retail transaction dataset.

    rfm : pandas.DataFrame
        RFM dataset indexed by Customer ID and containing
        a 'Segment' column.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - MonthYear
        - Segment
        - TotalPrice
    """

    required_df_columns = {
        "InvoiceDate",
        "Customer ID",
        "TotalPrice",
    }

    missing = required_df_columns - set(df.columns)
    if missing:
        raise ValueError(
            "segment_trend missing required columns: "
            + ", ".join(sorted(missing))
        )

    if "Segment" not in rfm.columns:
        raise ValueError(
            "RFM data must contain a 'Segment' column."
        )

    if df.empty or rfm.empty:
        return pd.DataFrame(
            columns=[
                "MonthYear",
                "Segment",
                "TotalPrice",
            ]
        )

    # Work only with columns required by this analysis.
    data = df[
        [
            "InvoiceDate",
            "Customer ID",
            "TotalPrice",
        ]
    ].copy()

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"],
        errors="coerce",
    )

    data["TotalPrice"] = pd.to_numeric(
        data["TotalPrice"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "InvoiceDate",
            "Customer ID",
            "TotalPrice",
        ]
    )

    if data.empty:
        return pd.DataFrame(
            columns=[
                "MonthYear",
                "Segment",
                "TotalPrice",
            ]
        )

    data["MonthYear"] = data["InvoiceDate"].dt.to_period("M")

    # Only merge the required RFM column.
    segment_data = rfm[["Segment"]].copy()

    merged = data.merge(
        segment_data,
        left_on="Customer ID",
        right_index=True,
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(
            columns=[
                "MonthYear",
                "Segment",
                "TotalPrice",
            ]
        )

    trend = (
        merged.groupby(
            ["MonthYear", "Segment"],
            observed=True,
            sort=True,
        )["TotalPrice"]
        .sum()
        .reset_index()
    )

    trend["MonthYear"] = trend["MonthYear"].astype(str)

    return trend


# ============================================================
# COHORT ANALYSIS
# ============================================================

def cohort_analysis(df):
    """
    Perform monthly customer cohort analysis.

    Customers are assigned to the month of their first purchase.

    Parameters
    ----------
    df : pandas.DataFrame
        Retail transaction dataset.

    Returns
    -------
    pandas.DataFrame
        Customer-count cohort matrix.
    """

    required_columns = {
        "InvoiceDate",
        "Customer ID",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "cohort_analysis missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        return pd.DataFrame()

    data = df[
        [
            "InvoiceDate",
            "Customer ID",
        ]
    ].copy()

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "InvoiceDate",
            "Customer ID",
        ]
    )

    if data.empty:
        return pd.DataFrame()

    data["OrderMonth"] = (
        data["InvoiceDate"]
        .dt.to_period("M")
    )

    data["Cohort"] = (
        data.groupby(
            "Customer ID",
            observed=True,
        )["InvoiceDate"]
        .transform("min")
        .dt.to_period("M")
    )

    cohort_data = (
        data.groupby(
            ["Cohort", "OrderMonth"],
            observed=True,
            sort=True,
        )["Customer ID"]
        .nunique()
        .reset_index()
    )

    if cohort_data.empty:
        return pd.DataFrame()

    cohort_pivot = cohort_data.pivot(
        index="Cohort",
        columns="OrderMonth",
        values="Customer ID",
    )

    cohort_pivot.index = (
        cohort_pivot.index.astype(str)
    )

    cohort_pivot.columns = (
        cohort_pivot.columns.astype(str)
    )

    return cohort_pivot


# ============================================================
# RETENTION
# ============================================================

def retention_rate(cohort):
    """
    Calculate cohort retention percentages.

    The first observed month of each cohort is treated as
    100% retention.

    Parameters
    ----------
    cohort : pandas.DataFrame
        Cohort customer-count matrix.

    Returns
    -------
    pandas.DataFrame
        Retention percentages.
    """

    if cohort is None or cohort.empty:
        return pd.DataFrame(
            index=getattr(cohort, "index", None),
            columns=getattr(cohort, "columns", None),
        )

    first_period = cohort.iloc[:, 0].replace(
        0,
        np.nan,
    )

    retention = cohort.divide(
        first_period,
        axis=0,
    )

    return retention.fillna(0)


# ============================================================
# ADVANCED CLV
# ============================================================

def advanced_clv(rfm):
    """
    Calculate an advanced CLV-style score.

    Formula
    -------
    CLV = Monetary × Frequency × 1 / (Recency + 1)

    Parameters
    ----------
    rfm : pandas.DataFrame
        RFM dataset containing Monetary, Frequency and Recency.

    Returns
    -------
    pandas.DataFrame
        Updated RFM DataFrame containing CLV.
    """

    required_columns = {
        "Monetary",
        "Frequency",
        "Recency",
    }

    missing = required_columns - set(rfm.columns)
    if missing:
        raise ValueError(
            "advanced_clv missing required columns: "
            + ", ".join(sorted(missing))
        )

    if rfm.empty:
        result = rfm.copy()
        result["CLV"] = pd.Series(
            dtype="float64",
            index=result.index,
        )
        return result

    result = rfm.copy()

    monetary = pd.to_numeric(
        result["Monetary"],
        errors="coerce",
    ).fillna(0)

    frequency = pd.to_numeric(
        result["Frequency"],
        errors="coerce",
    ).fillna(0)

    recency = pd.to_numeric(
        result["Recency"],
        errors="coerce",
    ).fillna(0)

    result["CLV"] = (
        monetary
        * frequency
        * (1.0 / (recency + 1.0))
    )

    return result


# ============================================================
# MEMORY-SAFE PRODUCT RECOMMENDATION ENGINE
# ============================================================

def product_recommendations(
    df,
    max_products=100,
    recommendation_count=3,
    max_items_per_invoice=50,
):
    """
    Generate product recommendations using transaction
    co-occurrence instead of constructing a huge dense
    Invoice × Product matrix.

    This replaces the previous implementation:

        df.groupby(
            ['Invoice', 'Description']
        )['Quantity'].sum().unstack()

    The previous approach could allocate hundreds of millions
    of cells for large retail datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        Retail transaction dataset.

    max_products : int, default=100
        Maximum number of frequently purchased products used
        by the recommendation engine.

    recommendation_count : int, default=3
        Number of recommendations returned for each product.

    max_items_per_invoice : int, default=50
        Maximum number of products considered from one invoice.
        This prevents unusually large baskets from generating
        excessive pair combinations.

    Returns
    -------
    dict
        Dictionary:

        {
            "Product A": ["Product B", "Product C", "Product D"],
            ...
        }

    Notes
    -----
    Recommendations are based on normalized co-occurrence:

        score(i, j) =
            co_occurrence(i, j)
            / sqrt(freq(i) * freq(j))

    This is similar in spirit to cosine similarity but avoids
    constructing a massive dense correlation matrix.
    """

    required_columns = {
        "Invoice",
        "Description",
        "Quantity",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "product_recommendations missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        return {}

    if max_products <= 1:
        return {}

    if recommendation_count <= 0:
        return {}

    if max_items_per_invoice <= 1:
        return {}

    # --------------------------------------------------------
    # Keep only columns required for recommendations.
    # --------------------------------------------------------

    data = df[
        [
            "Invoice",
            "Description",
            "Quantity",
        ]
    ].copy()

    # Remove invalid rows early.
    data = data.dropna(
        subset=[
            "Invoice",
            "Description",
        ]
    )

    if data.empty:
        return {}

    # Convert quantity only where needed.
    data["Quantity"] = pd.to_numeric(
        data["Quantity"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["Quantity"]
    )

    # --------------------------------------------------------
    # Determine the most frequently purchased products.
    #
    # Limiting the product universe is important for large
    # retail datasets because recommendation complexity grows
    # rapidly with the number of products.
    # --------------------------------------------------------

    product_frequency = (
        data.groupby(
            "Description",
            observed=True,
            sort=False,
        )["Quantity"]
        .sum()
    )

    product_frequency = (
        product_frequency
        .sort_values(
            ascending=False
        )
        .head(max_products)
    )

    if product_frequency.empty:
        return {}

    selected_products = set(
        product_frequency.index
    )

    # --------------------------------------------------------
    # Keep only transactions containing selected products.
    # --------------------------------------------------------

    data = data[
        data["Description"].isin(
            selected_products
        )
    ]

    if data.empty:
        return {}

    # --------------------------------------------------------
    # Remove duplicate product occurrences within the same
    # invoice.
    #
    # For recommendation purposes, whether a product appears
    # in a basket is more important than how many times it
    # appears.
    # --------------------------------------------------------

    data = data.drop_duplicates(
        subset=[
            "Invoice",
            "Description",
        ]
    )

    # --------------------------------------------------------
    # Build invoice -> products mapping.
    #
    # We intentionally do NOT use unstack(), pivot(), or
    # crosstab() here.
    # --------------------------------------------------------

    invoice_groups = (
        data.groupby(
            "Invoice",
            observed=True,
            sort=False,
        )["Description"]
        .agg(list)
    )

    pair_counts = Counter()
    item_counts = Counter()

    # --------------------------------------------------------
    # Count products and co-occurring product pairs.
    # --------------------------------------------------------

    for products in invoice_groups:
        # Remove duplicates while preserving order.
        unique_products = list(
            dict.fromkeys(products)
        )

        # Prevent very large baskets from creating an
        # excessive number of pair combinations.
        if len(unique_products) > max_items_per_invoice:
            unique_products = unique_products[
                :max_items_per_invoice
            ]

        if not unique_products:
            continue

        for product in unique_products:
            item_counts[product] += 1

        if len(unique_products) < 2:
            continue

        for item_a, item_b in combinations(
            unique_products,
            2,
        ):
            if item_a == item_b:
                continue

            if item_a > item_b:
                item_a, item_b = (
                    item_b,
                    item_a,
                )

            pair_counts[
                (item_a, item_b)
            ] += 1

    if not item_counts:
        return {}

    # --------------------------------------------------------
    # Calculate normalized co-occurrence scores.
    # --------------------------------------------------------

    recommendation_scores = {}

    for (item_a, item_b), count in pair_counts.items():

        frequency_a = item_counts.get(
            item_a,
            0,
        )

        frequency_b = item_counts.get(
            item_b,
            0,
        )

        if (
            frequency_a <= 0
            or frequency_b <= 0
        ):
            continue

        denominator = np.sqrt(
            frequency_a * frequency_b
        )

        if denominator <= 0:
            continue

        score = count / denominator

        recommendation_scores.setdefault(
            item_a,
            []
        ).append(
            (
                item_b,
                score,
                count,
            )
        )

        recommendation_scores.setdefault(
            item_b,
            []
        ).append(
            (
                item_a,
                score,
                count,
            )
        )

    # --------------------------------------------------------
    # Build final recommendation dictionary.
    #
    # Preserve the original behavior of returning
    # recommendations for up to the first five products.
    # --------------------------------------------------------

    recs = {}

    target_products = list(
        product_frequency.index
    )[:5]

    for item in target_products:

        candidates = (
            recommendation_scores
            .get(item, [])
        )

        if not candidates:
            recs[item] = []
            continue

        candidates.sort(
            key=lambda value: (
                value[1],
                value[2],
            ),
            reverse=True,
        )

        recommendations = []

        for candidate, _, _ in candidates:
            if candidate == item:
                continue

            recommendations.append(
                candidate
            )

            if len(recommendations) >= recommendation_count:
                break

        recs[item] = recommendations

    return recs


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomalies(df):
    """
    Detect anomalous daily revenue values using a 2-standard-
    deviation threshold.

    Returns
    -------
    pandas.Series
        Daily revenue values considered anomalous.
    """

    required_columns = {
        "InvoiceDate",
        "TotalPrice",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "detect_anomalies missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        return pd.Series(
            dtype="float64",
            name="TotalPrice",
        )

    data = df[
        [
            "InvoiceDate",
            "TotalPrice",
        ]
    ].copy()

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"],
        errors="coerce",
    )

    data["TotalPrice"] = pd.to_numeric(
        data["TotalPrice"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "InvoiceDate",
            "TotalPrice",
        ]
    )

    if data.empty:
        return pd.Series(
            dtype="float64",
            name="TotalPrice",
        )

    # Use daily aggregation rather than grouping by every
    # timestamp.
    data["Date"] = (
        data["InvoiceDate"].dt.floor("D")
    )

    daily = (
        data.groupby(
            "Date",
            observed=True,
            sort=True,
        )["TotalPrice"]
        .sum()
    )

    if daily.empty:
        return daily

    mean = daily.mean()

    std = daily.std()

    # A single observation has NaN std. In that case there is
    # not enough information to identify anomalies.
    if pd.isna(std) or std == 0:
        return daily.iloc[0:0]

    upper_threshold = mean + 2 * std
    lower_threshold = mean - 2 * std

    anomalies = daily[
        (daily > upper_threshold)
        | (daily < lower_threshold)
    ]

    return anomalies


# ============================================================
# SMART INSIGHTS
# ============================================================

def smart_summary(df):
    """
    Generate a basic business performance summary.

    Parameters
    ----------
    df : pandas.DataFrame
        Retail transaction dataset.

    Returns
    -------
    str
        Human-readable summary.
    """

    if "TotalPrice" not in df.columns:
        raise ValueError(
            "smart_summary requires a 'TotalPrice' column."
        )

    if df.empty:
        return (
            "No transaction data is available "
            "for generating business insights."
        )

    revenue = pd.to_numeric(
        df["TotalPrice"],
        errors="coerce",
    ).sum()

    valid_values = pd.to_numeric(
        df["TotalPrice"],
        errors="coerce",
    ).dropna()

    if valid_values.empty:
        return (
            "No valid revenue data is available "
            "for generating business insights."
        )

    average_transaction_value = (
        valid_values.mean()
    )

    transaction_count = len(
        valid_values
    )

    if revenue > 0:
        return (
            "Business performance is strong with "
            f"total revenue of {revenue:,.2f} across "
            f"{transaction_count:,} transactions and "
            f"an average transaction value of "
            f"{average_transaction_value:,.2f}."
        )

    if revenue == 0:
        return (
            "Revenue is currently zero; review the "
            "transaction data and sales activity."
        )

    return (
        "Revenue is negative overall; review returns, "
        "cancellations, refunds, and transaction quality."
    )