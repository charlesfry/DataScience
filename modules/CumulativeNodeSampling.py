import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

class CumulativeNodeSampling(BaseEstimator,TransformerMixin) :
    """
    class for Cumulative Node Sampling: a way to drop less-important
    features from a high-dimensional dataset
    """
    def __init__(self,feat_importance:np.array,sample=.9):
        """

        :param feat_importance: a numpy array that contains the information on what features are most important
        :param sample: how much of the explainatory power you want to keep
        """

        self.feat_importance = feat_importance.copy()
        self.sample = sample

        assert sample > 0
        assert sample <= 1

    def fit(self, X, y=None):
        feats = self.feat_importance
        sample = self.sample

        feats /= feats.sum()
        feat_df = pd.DataFrame({
            'feats':X.index,
            'values':feats
        })

        feat_df.sort_values(ascending=False,inplace=True)

        num_cols = 1
        score = 0
        for i in range(len(feats)) :
            num_cols += 1
            score += feat_df.values[i]
            if score > sample : break

        keys = feat_df.feats[:num_cols]

        self.keys = keys

        return self

    def transform(self,X,y=None):





if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris.data,iris.target],
                      columns=iris.feature_names + ['target'])
    df.target = pd.Series(df.target == 1).astype(np.int)

    X,y = load_iris(return_X_y=True)

    from lightgbm import LGBMClassifier

    clf = LGBMClassifier()

    clf.fit(X,y)

    feats = clf.feature_importances_
    print(feats.sum())