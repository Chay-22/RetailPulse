"""
=========================================================
RetailPulse Production Inventory Optimization
---------------------------------------------------------
Features
--------
✔ EOQ (Economic Order Quantity)
✔ Safety Stock
✔ Reorder Point
✔ Inventory Turnover
✔ ABC Classification
✔ Smart Recommendations
✔ Works with Prophet + LSTM Ensemble
=========================================================
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

Number = Union[int, float, np.number]

logger = logging.getLogger(__name__)

FORECAST_COLUMN = "Forecast"
LEGACY_FORECAST_COLUMN = "yhat"


# -------------------------------------------------------
# EOQ
# -------------------------------------------------------

def calculate_eoq(
    annual_demand: Number,
    ordering_cost: Number = 100,
    holding_cost: Number = 10
) -> float:
    """
    Calculate Economic Order Quantity (EOQ).

    Invalid, missing, or non-positive demand returns 0 to preserve the
    historical API behavior while avoiding runtime failures.
    """
    annual_demand = _safe_float(annual_demand, "annual_demand", minimum=0.0)
    ordering_cost = _safe_float(ordering_cost, "ordering_cost", minimum=0.0)
    holding_cost = _safe_float(holding_cost, "holding_cost", minimum=0.0)

    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost <= 0:
        return 0.0

    return float(np.sqrt(
        (2 * annual_demand * ordering_cost)
        /
        holding_cost
    ))


# -------------------------------------------------------
# Safety Stock
# -------------------------------------------------------

def calculate_safety_stock(
    demand_std: Number,
    lead_time: Number = 7,
    service_factor: Number = 1.65
) -> float:
    """
    Calculate safety stock from demand volatility and lead time.

    Missing or invalid inputs are treated as 0 because a zero-variance
    forecast should not require buffer stock.
    """
    demand_std = _safe_float(demand_std, "demand_std", minimum=0.0)
    lead_time = _safe_float(lead_time, "lead_time", minimum=0.0)
    service_factor = _safe_float(service_factor, "service_factor", minimum=0.0)

    if demand_std <= 0 or lead_time <= 0 or service_factor <= 0:
        return 0.0

    return float(service_factor * demand_std * np.sqrt(lead_time))


# -------------------------------------------------------
# Reorder Point
# -------------------------------------------------------

def calculate_reorder_point(
    average_daily_demand: Number,
    lead_time: Number,
    safety_stock: Number
) -> float:
    """
    Calculate the reorder point for replenishment planning.
    """
    average_daily_demand = _safe_float(
        average_daily_demand,
        "average_daily_demand",
        minimum=0.0
    )
    lead_time = _safe_float(lead_time, "lead_time", minimum=0.0)
    safety_stock = _safe_float(safety_stock, "safety_stock", minimum=0.0)

    return float(
        average_daily_demand * lead_time
        + safety_stock
    )


def _safe_float(
    value: Number,
    name: str,
    minimum: Optional[float] = None
) -> float:
    """
    Convert a scalar value to a finite float with optional lower bound.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using 0.0", name, value)
        return 0.0

    if not np.isfinite(result):
        logger.warning("Non-finite %s=%r; using 0.0", name, value)
        return 0.0

    if minimum is not None and result < minimum:
        logger.warning("%s=%r is below %s; using %s", name, value, minimum, minimum)
        return minimum

    return result


def _validate_forecast_df(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize forecast input without mutating caller data.
    """
    if not isinstance(forecast_df, pd.DataFrame):
        raise TypeError("forecast_df must be a pandas DataFrame")

    df = forecast_df.copy()

    if FORECAST_COLUMN not in df.columns:
        if LEGACY_FORECAST_COLUMN in df.columns:
            logger.info(
                "Using legacy '%s' column as '%s'",
                LEGACY_FORECAST_COLUMN,
                FORECAST_COLUMN
            )
            df[FORECAST_COLUMN] = df[LEGACY_FORECAST_COLUMN]
        else:
            raise ValueError("forecast_df must contain a 'Forecast' column")

    forecast = pd.to_numeric(df[FORECAST_COLUMN], errors="coerce")

    invalid_count = int(forecast.isna().sum())
    if invalid_count:
        logger.warning(
            "Coerced %s invalid Forecast value(s) to 0.0",
            invalid_count
        )

    negative_count = int((forecast < 0).sum())
    if negative_count:
        logger.warning(
            "Clipped %s negative Forecast value(s) to 0.0",
            negative_count
        )

    df[FORECAST_COLUMN] = forecast.fillna(0.0).clip(lower=0.0).astype(float)
    return df


# -------------------------------------------------------
# ABC Classification
# -------------------------------------------------------

def abc_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add contribution, cumulative contribution, and ABC classification columns.

    The output columns remain backward compatible with the existing Streamlit
    app while handling empty and zero-demand inputs safely.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if FORECAST_COLUMN not in df.columns:
        raise ValueError("df must contain a 'Forecast' column")

    temp = df.copy()
    forecast = pd.to_numeric(temp[FORECAST_COLUMN], errors="coerce")
    temp[FORECAST_COLUMN] = forecast.fillna(0.0).clip(lower=0.0).astype(float)

    total_forecast = float(temp[FORECAST_COLUMN].sum())

    if temp.empty:
        temp["Contribution"] = pd.Series(dtype=float)
        temp["CumContribution"] = pd.Series(dtype=float)
        temp["ABC"] = pd.Series(dtype=object)
        logger.info("ABC analysis received an empty forecast DataFrame")
        return temp

    if total_forecast <= 0:
        temp["Contribution"] = 0.0
        temp["CumContribution"] = 0.0
        temp["ABC"] = "C"
        logger.info("ABC analysis completed with zero total forecast demand")
        return temp

    temp["Contribution"] = temp[FORECAST_COLUMN] / total_forecast

    temp = temp.sort_values(
        "Contribution",
        ascending=False
    )

    temp["CumContribution"] = (
        temp["Contribution"].cumsum()
    )

    temp["ABC"] = "C"

    temp.loc[
        temp["CumContribution"] <= 0.80,
        "ABC"
    ] = "A"

    temp.loc[
        (
            temp["CumContribution"] > 0.80
        )
        &
        (
            temp["CumContribution"] <= 0.95
        ),
        "ABC"
    ] = "B"

    return temp


# -------------------------------------------------------
# Main Inventory Engine
# -------------------------------------------------------

def inventory_recommendation(
    forecast_df: pd.DataFrame,
    current_stock: Number = 1000,
    lead_time: Number = 7
) -> pd.DataFrame:
    """
    Generate inventory metrics and replenishment recommendations.

    Parameters
    ----------
    forecast_df:
        DataFrame containing a ``Forecast`` column. All original columns are
        preserved and inventory output columns are appended.
    current_stock:
        Current available stock. Invalid or negative values are treated as 0.
    lead_time:
        Lead time in days. Invalid or negative values are treated as 0.

    Returns
    -------
    pandas.DataFrame
        Input rows plus EOQ, SafetyStock, ReorderPoint, CurrentStock, Reorder,
        InventoryTurnover, Contribution, CumContribution, ABC, and
        Recommendation.
    """

    df = _validate_forecast_df(forecast_df)
    current_stock = _safe_float(current_stock, "current_stock", minimum=0.0)
    lead_time = _safe_float(lead_time, "lead_time", minimum=0.0)

    # -----------------------------
    # Forecast Demand
    # -----------------------------

    avg_daily = float(df[FORECAST_COLUMN].mean()) if not df.empty else 0.0

    annual_demand = avg_daily * 365

    demand_std = float(df[FORECAST_COLUMN].std(ddof=0)) if not df.empty else 0.0

    # -----------------------------
    # Inventory Metrics
    # -----------------------------

    eoq = calculate_eoq(
        annual_demand
    )

    safety_stock = calculate_safety_stock(
        demand_std,
        lead_time
    )

    reorder_point = calculate_reorder_point(
        avg_daily,
        lead_time,
        safety_stock
    )

    inventory_turnover = (
        annual_demand
        /
        max(current_stock, 1.0)
    )

    # -----------------------------
    # Recommendation
    # -----------------------------

    rounded_eoq = int(round(eoq))
    rounded_safety_stock = int(round(safety_stock))
    rounded_reorder_point = int(round(reorder_point))

    df["EOQ"] = rounded_eoq

    df["SafetyStock"] = rounded_safety_stock

    df["ReorderPoint"] = rounded_reorder_point

    df["CurrentStock"] = current_stock

    df["Reorder"] = np.where(

        current_stock <= reorder_point,

        rounded_eoq,

        0

    )

    df["InventoryTurnover"] = round(
        inventory_turnover,
        2
    )

    # -----------------------------
    # ABC
    # -----------------------------

    df = abc_analysis(df)

    # -----------------------------
    # Recommendation Text
    # -----------------------------

    conditions = [
        df["Reorder"] > 0,
        df["ABC"].eq("A"),
        df["ABC"].eq("B")
    ]

    choices = [
        "Restock Immediately",
        "Monitor Frequently",
        "Weekly Review"
    ]

    df["Recommendation"] = np.select(
        conditions,
        choices,
        default="Normal Monitoring"
    )

    logger.info(
        "Generated inventory recommendations for %s row(s): "
        "avg_daily=%.4f annual_demand=%.4f eoq=%.4f safety_stock=%.4f "
        "reorder_point=%.4f current_stock=%.4f lead_time=%.4f",
        len(df),
        avg_daily,
        annual_demand,
        eoq,
        safety_stock,
        reorder_point,
        current_stock,
        lead_time
    )

    return df
