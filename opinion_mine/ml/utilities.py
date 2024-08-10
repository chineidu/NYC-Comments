"""This module contains utility functions for buidling the machine learning pipelines."""

from typing import Union

import numpy as np
import pandas as pd
import polars as pl


def extract_temporal_features(
    data: Union[pl.DataFrame, pd.DataFrame],
    date_column: str,
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> pl.DataFrame:
    """
    Extract temporal features from a date column in a Polars or Pandas DataFrame.

    Parameters
    ----------
    data : Union[pl.DataFrame, pd.DataFrame]
        Input DataFrame containing the date column.
    date_column : str
        Name of the column containing date information.
    date_format : str, optional
        Format of the date string, by default "%Y-%m-%d %H:%M:%S".

    Returns
    -------
    pl.DataFrame
        DataFrame with additional temporal features.
    """
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    # Convert to datetime
    if not data[date_column].dtype == pl.Date:
        data = data.with_columns(pubDate=pl.col(date_column).str.to_date(date_format))
    else:
        data = data

    try:
        data = data.with_columns(
            day=pl.col(date_column).dt.day(),
            # where monday = 1 and sunday = 7
            day_of_week=pl.col(date_column).dt.weekday(),
            week_of_year=pl.col(date_column).dt.week(),
            month=pl.col(date_column).dt.month(),
            year=pl.col(date_column).dt.year(),
            quarter=pl.col(date_column).dt.quarter(),
        )
    except:
        print("Error creating temporal features")  # noqa: T201
        return pl.DataFrame()

    return data


def create_cyclic_features(data: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    """
    Create cyclic temporal features from the input DataFrame.

    Parameters
    ----------
    data : Union[pl.DataFrame, pd.DataFrame]
        Input DataFrame containing temporal features.

    Returns
    -------
    pl.DataFrame
        DataFrame with additional cyclic temporal features.

    Notes
    -----
    This function creates sine and cosine transformations for day, day of week,
    week of year, and month columns.
    """
    day_factor: int = 30
    day_of_week_factor: int = 7
    week_of_year_factor: int = 52
    month_factor: int = 12

    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    try:
        data = data.with_columns(
            day_sin=pl.col("day").map_elements(lambda x: np.sin(2 * np.pi * x / day_factor)),
            day_cos=pl.col("day").map_elements(lambda x: np.cos(2 * np.pi * x / day_factor)),
            # where monday = 1 and sunday = 7
            day_of_week_sin=pl.col("day_of_week").map_elements(
                lambda x: np.sin(2 * np.pi * x / day_of_week_factor)
            ),
            day_of_week_cos=pl.col("day_of_week").map_elements(
                lambda x: np.cos(2 * np.pi * x / day_of_week_factor)
            ),
            week_of_year_sin=pl.col("week_of_year").map_elements(
                lambda x: np.sin(2 * np.pi * x / week_of_year_factor)
            ),
            week_of_year_cos=pl.col("week_of_year").map_elements(
                lambda x: np.cos(2 * np.pi * x / week_of_year_factor),
            ),
            month_sin=pl.col("month").map_elements(lambda x: np.sin(2 * np.pi * x / month_factor)),
            month_cos=pl.col("month").map_elements(lambda x: np.cos(2 * np.pi * x / month_factor)),
        )
    except:
        print("Error creating cyclic temporal features")  # noqa: T201
        return pl.DataFrame()

    return data
