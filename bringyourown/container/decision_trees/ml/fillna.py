# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:02:29 2022

@author: A823118
"""
# =============================================================================
# Import library
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from collections import OrderedDict

# =============================================================================
# Fill Missing Value
# =============================================================================
class FillMissingValue(BaseEstimator, TransformerMixin):
    """
    FillMissingValue

    Parameters
    ----------
    cat_cols : list
        categorical column.
    num_cols : list
        numerical column.

    Returns
    -------
    Data Frame.

    """

    def __init__(
        self,
        cat_cols=None,
        num_cols=None,
        date_cols=None,
        user_defined_fillna=None,
        **kwargs
    ):

        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.date_cols = date_cols
        self.user_defined_fillna = user_defined_fillna

    def fit(self, X, y=None):

        attributes = OrderedDict()

        # mode of column
        if self.cat_cols:
            attributes["cat_cols_mode"] = OrderedDict()
            for col in self.cat_cols:
                mode = X[[col]].mode()
                attributes["cat_cols_mode"][col] = mode

        # mean of column
        if self.num_cols:
            attributes["num_cols_mean"] = OrderedDict()
            for col in self.num_cols:
                mean = X[[col]].mean()
                attributes["num_cols_mean"][col] = mean
        
        # set attributes
        for k, v in attributes.items():
            setattr(self, k, v)
        return self

    def transform(self, X):

        # custom fill missing value
        if self.user_defined_fillna:
            for col, val in self.user_defined_fillna.items():
                X[col] = X[col].fillna(val)

        # fill na mode of column
        if self.cat_cols:
            for col, mode in self.cat_cols_mode.items():
                X[col] = X[col].fillna(mode.values[0][0])

        # fill na mean of column
        if self.num_cols:
            for col, mean in self.num_cols_mean.items():
                X[col] = X[col].fillna(mean[0])

        # fill na for date columns
        if self.date_cols:
            for col, date in self.date_cols.items():
                X[col] = X[col].fillna(date)

        return X


# =============================================================================
# Replace Nan
# =============================================================================


class ReplaceNan(BaseEstimator, TransformerMixin):
    """
    Replace noise character to as Missing Value

    Parameters
    ----------

    Returns
    -------
    Data Frame.

    """

    def __init__(self):
        self.na_value = {
            "None": np.nan,
            "NONE": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "NAN": np.nan,
            "NaT": np.nan,
            "NA": np.nan,
            "na": np.nan,
            "": np.nan,
            " ": np.nan,
            "-": np.nan,
            "_": np.nan,
            "null": np.nan,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # replace na
        X = X.replace(self.na_value)
        return X
