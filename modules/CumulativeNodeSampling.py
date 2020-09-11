import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

class CumulativeNodeSampling(BaseEstimator,TransformerMixin) :
    """
    class for Cumulative Node Sampling: a way to drop less-important
    features from a high-dimensional dataset
    """
    def __init__(self,clf,sample=.9):
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
        clf = self.clf

        clf.fit(X,y)

        feats = clf.feature_importances_
        return feats
        feats = feats / feats.sum()




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

from lightgbm import LGBMClassifier
poop = np.random.randn(5)
anus = np.array([.9,2.1,2.9,4.4,5.22])
butt = np.array([1,2,3,4,5])
X = np.column_stack([poop,anus])
clf = LGBMClassifier()
pog = CumulativeNodeSampling(clf)
poggers = pog.fit(X,butt)
print(poggers)
quit()

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris.data,iris.target],
                      columns=iris.feature_names + ['target'])
    df.target = pd.Series(df.target == 1).astype(np.int)

    X,y = load_iris(return_X_y=True)



    clf = LGBMClassifier()

    sample = CumulativeNodeSampling(clf)
    butt = sample.fit(X,y)

    print(butt)