import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SelectByRank(BaseEstimator, TransformerMixin):
    """
    Class implementing feature selection with given rank.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        row (int): row in rank matrix to use to perform feature selection. 
        rank (int): rank of features computed by some feature ranking algorithm.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        row (int): row in rank matrix to use to perform feature selection. 
        rank (int): rank of features computed by some feature ranking algorithm.
    """

    def __init__(self, n_features_to_select=10, row=0, rank=np.arange(1, 10+1)):
        self.n_features_to_select = n_features_to_select
        self.rank = rank
        self.row = row

    def fit(self, data, target):
        """
        Do nothing.

        Args:
            data (numpy.ndarray): matrix of training samples.
            target (numpy.ndarray): array of class values for training samples.

        Returns:
            (object) reference to self.
        """

        return self

    def transform(self, data):
        """
        Perform feature selection.

        Args:
            data (numpy.ndarray): matrix of data samples.

        Returns:
            data (numpy.ndarray): matrix of data samples with selected features.
        """

        # select n_features_to_select best features and return selected features.
        msk = self.rank[self.row, :] <= self.n_features_to_select  # Compute mask.
        return data[:, msk]  # Perform feature selection.


