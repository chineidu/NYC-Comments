"""This module contains transformers for buidling the machine learning pipelines."""

import re
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from opinion_mine.ml.utilities import create_cyclic_features, extract_temporal_features


class ExtractTemporalFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer to extract temporal features from a date column.

    Parameters
    ----------
    date_column : str
        Name of the column containing date information.
    date_format : str, optional
        Format of the date string, by default "%Y-%m-%d %H:%M:%S".
    """

    def __init__(self, date_column: str, date_format: str = "%Y-%m-%d %H:%M:%S"):
        self.date_column: str = date_column
        self.date_format: str = date_format

    def fit(self, X: pl.DataFrame, y: Optional[pl.DataFrame] = None) -> "ExtractTemporalFeatures":
        """
        Fit the transformer (no-op).

        Parameters
        ----------
        X : pl.DataFrame
            Input features.
        y : Optional[pl.DataFrame], optional
            Target variable, by default None.

        Returns
        -------
        ExtractTemporalFeatures
            Fitted transformer.
        """
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input DataFrame by extracting temporal features.

        Parameters
        ----------
        X : pl.DataFrame
            Input features.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with additional temporal features.
        """
        return extract_temporal_features(X, self.date_column, self.date_format)


class DropFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer to drop specified features from a DataFrame.

    Parameters
    ----------
    features : list[str]
        List of feature names to be dropped.
    """

    def __init__(self, features: list[str]):
        self.features: list[str] = features

    def fit(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    ) -> "DropFeatures":
        """
        Fit the transformer (no-op).

        Parameters
        ----------
        X : Union[pl.DataFrame, pd.DataFrame]
            Input features.
        y : Optional[Union[pl.DataFrame, pd.DataFrame]], optional
            Target variable, by default None.

        Returns
        -------
        DropFeatures
            Fitted transformer.
        """
        return self

    def transform(self, X: Union[pl.DataFrame, pd.DataFrame]) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Transform the input DataFrame by dropping specified features.

        Parameters
        ----------
        X : Union[pl.DataFrame, pd.DataFrame]
            Input features.

        Returns
        -------
        Union[pl.DataFrame, pd.DataFrame]
            Transformed DataFrame with specified features dropped.

        Raises
        ------
        ValueError
            If the input is not a pandas DataFrame or a polars DataFrame.
        """
        if isinstance(X, pl.DataFrame):
            return X.drop(self.features)
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.features)
        raise ValueError("Input must be a pandas DataFrame or a polars DataFrame")


class CyclicTemporalFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer for creating cyclic temporal features.

    This transformer applies the create_cyclic_features function to the input data.
    """

    def __init__(self) -> None:
        """
        Initialize the CyclicTemporalFeatures transformer.
        """
        ...

    def fit(self, X: pl.DataFrame, y: Optional[pl.DataFrame] = None) -> "CyclicTemporalFeatures":
        """
        Fit the transformer to the data.

        This method is a no-op and returns self.

        Parameters
        ----------
        X : pl.DataFrame, shape (n_samples, n_features)
            Input features.
        y : Optional[pl.DataFrame], default=None
            Target values (ignored).

        Returns
        -------
        CyclicTemporalFeatures
            The fitted transformer.
        """
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input DataFrame by extracting temporal features.

        Parameters
        ----------
        X : pl.DataFrame, shape (n_samples, n_features)
            Input features.

        Returns
        -------
        pl.DataFrame, shape (n_samples, n_features + n_cyclic_features)
            Transformed DataFrame with additional temporal features.
        """
        return create_cyclic_features(X)


class NumericalScaler(BaseEstimator, TransformerMixin):
    """
    Transformer for scaling numerical features.

    This transformer applies either StandardScaler or MinMaxScaler to the specified features.

    Parameters
    ----------
    scaler_type : Literal["standard", "min_max"]
        Type of scaler to use.
    features : list[str] | None, optional
        List of features to scale. If None, all features will be scaled.
    exclude_features : list[str] | None, optional
        List of features to exclude from scaling.

    Attributes
    ----------
    scaler : StandardScaler | MinMaxScaler
        The scaler object used for transformation.
    features : list[str] | None
        List of features to scale.
    exclude_features : list[str]
        List of features to exclude from scaling.
    ignore_columns_ : list[str]
        List of columns to ignore during scaling.
    """

    def __init__(
        self,
        scaler_type: Literal["standard", "min_max"],
        features: list[str] | None = None,
        exclude_features: list[str] | None = None,
    ) -> None:
        self.scaler_type: Literal["standard", "min_max"] = scaler_type
        if features is None and exclude_features is None:
            raise ValueError("`features` and `exclude_features` cannot both be None")
        if features is not None and exclude_features is not None:
            raise ValueError("`features` and `exclude_features` cannot both be not None")

        self.features: list[str] | None = features

        if exclude_features is None:
            self.exclude_features: list[str] = []
        else:
            assert isinstance(exclude_features, list), "`exclude_features` must be of type List"
            self.exclude_features = exclude_features

        if scaler_type not in ["standard", "min_max"]:
            raise ValueError("scaler_type must be either 'standard' or 'min_max'")
        if scaler_type == "standard":
            self.scaler: StandardScaler = StandardScaler()
        else:
            self.scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), clip=True)  # type: ignore

    def fit(self, X: pl.DataFrame, y: Optional[pl.DataFrame] = None) -> "NumericalScaler":
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : pl.DataFrame, shape (n_samples, n_features)
            Input features.
        y : Optional[pl.DataFrame], default=None
            Target values (ignored).

        Returns
        -------
        NumericalScaler
            The fitted transformer.
        """
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)
        if self.features is not None:
            self.ignore_columns_: list[str] = sorted(set(X.columns) - set(self.features))

        elif self.exclude_features and self.features is None:
            self.ignore_columns_ = self.exclude_features
            self.features = sorted(set(X.columns) - set(self.exclude_features))

        self.ignore_columns_ = self.ignore_columns_
        self.scaler.fit(X.select(self.features))
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input DataFrame by scaling numerical features.

        Parameters
        ----------
        X : pl.DataFrame, shape (n_samples, n_features)
            Input features.

        Returns
        -------
        pl.DataFrame, shape (n_samples, n_features)
            Transformed DataFrame with scaled features.
        """
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)
        vector: np.ndarray = self.scaler.transform(X.select(self.features))
        ignore_df: pl.DataFrame = X.select(self.ignore_columns_)
        vector_df: pl.DataFrame = pl.DataFrame(vector, schema=self.features)
        df: pl.DataFrame = pl.concat([ignore_df, vector_df], how="horizontal")
        return df


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    A custom one-hot encoder that works with Polars DataFrames.

    Parameters
    ----------
    features : list[str] | None, optional
        List of column names to encode. If None, all columns will be encoded.

    Attributes
    ----------
    features : list[str] | None
        List of column names to encode.
    encoder : OneHotEncoder
        The underlying scikit-learn OneHotEncoder.
    ignore_columns : list[str]
        List of column names to ignore during encoding.
    ignore_columns_ : list[str]
        List of column names ignored during encoding after fitting.
    """

    def __init__(
        self,
        features: list[str] | None = None,
    ) -> None:
        self.features: list[str] | None = features
        self.encoder: OneHotEncoder = OneHotEncoder(handle_unknown="ignore")
        self.ignore_columns: list[str] = []

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "CustomOneHotEncoder":
        """
        Fit the OneHotEncoder to the input data.

        Parameters
        ----------
        X : pl.DataFrame, shape (n_samples, n_features)
            Input features to fit the encoder.
        y : pl.DataFrame | None, optional
            Ignored. Kept for scikit-learn compatibility.

        Returns
        -------
        CustomOneHotEncoder
            The fitted encoder.
        """
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)
        if self.features is None:
            self.features = X.columns
        self.ignore_columns_: list[str] = sorted(set(X.columns) - set(self.features))
        self.encoder.fit(X.select(self.features))
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input data using the fitted encoder.

        Parameters
        ----------
        X : pl.DataFrame, shape (n_samples, n_features)
            Input features to transform.

        Returns
        -------
        pl.DataFrame, shape (n_samples, n_encoded_features)
            Transformed DataFrame with one-hot encoded features.
        """
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)
        vector: np.ndarray = self.encoder.transform(X.select(self.features)).toarray()
        ignore_df: pl.DataFrame = X.select(self.ignore_columns_)
        vector_df: pl.DataFrame = pl.DataFrame(
            vector, schema=sorted(self.encoder.get_feature_names_out())
        )
        df: pl.DataFrame = pl.concat([ignore_df, vector_df], how="horizontal")
        return df


class TFIDFTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for TF-IDF vectorization of text data.

    Parameters
    ----------
    feature : str
        The name of the feature column containing text data.
    max_df : float, default=1.0
        The maximum document frequency for TfidfVectorizer.
    min_df : int, default=1
        The minimum document frequency for TfidfVectorizer.

    Attributes
    ----------
    tfidf : TfidfVectorizer
        The TF-IDF vectorizer object.
    ignore_columns_ : list[str]
        List of columns to ignore during transformation.
    """

    def __init__(
        self,
        feature: str,
        max_df: float = 1.0,
        min_df: int = 1,
    ) -> None:
        self.feature: str = feature
        self.max_df: float = max_df
        self.min_df: int = min_df
        self.tfidf: TfidfVectorizer = TfidfVectorizer(
            stop_words="english", max_df=max_df, min_df=min_df
        )

    def fit(self, X: pd.DataFrame | pl.DataFrame, y: None = None) -> "TFIDFTransformer":
        """
        Fit the TF-IDF vectorizer to the input data.

        Parameters
        ----------
        X : pd.DataFrame | pl.DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the text feature.
        y : None
            Ignored. Kept for compatibility with scikit-learn API.

        Returns
        -------
        TFIDFTransformer
            The fitted transformer.
        """
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)
        self.ignore_columns_: list[str] = sorted(set(X.columns) - {self.feature})
        self.tfidf.fit(X[self.feature])
        return self

    def transform(self, X: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input data using the fitted TF-IDF vectorizer.

        Parameters
        ----------
        X : pd.DataFrame | pl.DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the text feature.

        Returns
        -------
        pl.DataFrame, shape (n_samples, n_features + n_tfidf_features)
            Transformed DataFrame with TF-IDF features.
        """
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)
        tfidf_matrix: csr_matrix = self.tfidf.transform(X[self.feature])
        schema: list[str] = list(self.tfidf.get_feature_names_out())
        schema = [f"tfidf__{s}" for s in schema]
        tfidf_df: pl.DataFrame = pl.DataFrame(tfidf_matrix.toarray(), schema=schema)
        ignore_df: pl.DataFrame = X.select(self.ignore_columns_)
        df: pl.DataFrame = pl.concat([ignore_df, tfidf_df], how="horizontal")
        return df


class SVDTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for dimensionality reduction using Truncated SVD.

    Parameters
    ----------
    exclude_features : list[str] | None, default=None
        List of feature names to exclude from SVD transformation.
    include_pattern : str | None, default=None
        Regular expression pattern to include features for SVD transformation.
    n_components : int, default=100
        Number of components to keep in the SVD transformation.
    random_state : int, default=42
        Random state for reproducibility.

    Attributes
    ----------
    svd : TruncatedSVD
        The TruncatedSVD object for dimensionality reduction.
    features : list[str]
        List of feature names to include in SVD transformation.
    ignore_columns_ : list[str]
        List of feature names to exclude from SVD transformation.
    """

    def __init__(
        self,
        exclude_features: list[str] | None = None,
        include_pattern: str | None = None,
        n_components: int = 100,
        random_state: int = 42,
    ) -> None:
        if exclude_features is None and include_pattern is None:
            raise ValueError("`exclude_features` and `include_pattern` cannot both be None")
        if exclude_features is not None and include_pattern is not None:
            raise ValueError("`exclude_features` and `include_pattern` cannot both be not None")
        assert (
            isinstance(exclude_features, list) or exclude_features is None
        ), "`exclude_features` must be of type List"
        self.exclude_features: list[str] | None = exclude_features
        if include_pattern is not None:
            assert isinstance(include_pattern, str), "`include_pattern` must be of type str"
            self.include_pattern: str = include_pattern
        self.n_components: int = n_components
        self.random_state: int = random_state
        self.svd: TruncatedSVD = TruncatedSVD(n_components=n_components, random_state=random_state)

    def fit(self, X: pd.DataFrame | pl.DataFrame, y: None = None) -> "SVDTransformer":
        """
        Fit the SVD transformer to the input data.

        Parameters
        ----------
        X : pd.DataFrame | pl.DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the features.
        y : None
            Ignored. Kept for compatibility with scikit-learn API.

        Returns
        -------
        SVDTransformer
            The fitted transformer.
        """
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)
        if self.exclude_features:
            self.features: list[str] = list(set(X.columns) - set(self.exclude_features))
            self.ignore_columns_: list[str] = self.exclude_features
        elif hasattr(self, "include_pattern"):
            self.features = [
                col for col in X.columns if re.match(self.include_pattern, col, flags=re.IGNORECASE)
            ]
            self.ignore_columns_ = sorted(set(X.columns) - set(self.features))
        self.svd.fit(X.select(self.features))
        return self

    def transform(self, X: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input data using the fitted SVD transformer.

        Parameters
        ----------
        X : pd.DataFrame | pl.DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the features.

        Returns
        -------
        pl.DataFrame, shape (n_samples, n_excluded_features + n_components)
            Transformed DataFrame with reduced dimensions.
        """
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)
        svd_matrix: np.ndarray = self.svd.transform(X.select(self.features))
        svd_df: pl.DataFrame = pl.DataFrame(
            svd_matrix,
            schema=list(self.svd.get_feature_names_out()),
        )
        ignore_df: pl.DataFrame = X.select(self.ignore_columns_)
        df: pl.DataFrame = pl.concat([ignore_df, svd_df], how="horizontal")
        return df
