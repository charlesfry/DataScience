import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

class CumulativeNodeSampling(BaseEstimator,TransformerMixin) :
    """
    class for Cumulative Node Sampling: a way to drop less-important
    features from a high-dimensional dataset
    """
    def __init__(self,sample=.9,clf=None):
        """

        :param sample: how much of the explainatory power you want to keep
        :param clf: optional classifier to evaluate feature importances
        """

        self.sample = sample
        self.clf = clf

        assert sample > 0
        assert sample <= 1

    def fit(self, X, y=None):
        sample = self.sample

        if clf :
            feats = clf.fit(X,y).feature_importances_
        else :
            feats = pd.DataFrame([X,y]).corr(method='spearman').iloc[-1,:-1]
            feats = np.abs(feats)

        feats = feats / feats.sum()
        feat_df = pd.DataFrame({
            'feats':(-feats).argsort(),
            'values':feats
        })
        return feat_df

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
        pass


butt = np.array([1,2,3,4,5])
print((-butt).argsort())
quit()

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris.data,iris.target],
                      columns=iris.feature_names + ['target'])
    df.target = pd.Series(df.target == 1).astype(np.int)

    X,y = load_iris(return_X_y=True)

    from lightgbm import LGBMClassifier

    clf = LGBMClassifier()

    sample = CumulativeNodeSampling()
    butt = sample.fit(X,y)

    print(butt)