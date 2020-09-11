import numpy as np

from sklearn.base import BaseEstimator,TransformerMixin

class CumulativeColumnSampling(BaseEstimator, TransformerMixin) :
    """
    class for Cumulative Node Sampling: a way to drop less-important
    features from a high-dimensional dataset
    """
    def __init__(self,clf=None,sample=.5):
        """

        :param sample: how much of the explainatory power you want to keep, either a float amount of the explainatory
                        power, or an int of how many things to keep
        :param clf: classifier that supports feature_importances_
        """

        self.sample = sample
        self.clf = clf

        assert sample > 0

    def fit(self, X, y=None, **fit_params):

        clf = self.clf
        sample = self.sample

        if y is not None :
            clf.fit(X,y,**fit_params)
        else :
            clf.fit(X,**fit_params)

        feats = clf.feature_importances_

        feats = np.abs(feats) / np.abs(feats).sum()

        sort_feats = np.argsort(-feats)

        X_indicies = []
        cumulative_score = 0

        if sample <= 1 :
            for i in range(len(sort_feats)):
                X_indicies.append(sort_feats[i])
                cumulative_score += feats[sort_feats[i]]
                if cumulative_score >= sample: break

        else :
            for i in range(min(sample,X.shape[1])) :
                X_indicies.append(sort_feats[i])
                cumulative_score += feats[sort_feats[i]]

        self.X_indicies = X_indicies
        return self


    def transform(self,X,y=None):
        X_indicies = self.X_indicies
        return X[:,X_indicies]

    def fit_transform(self, X, y=None, clf=None,**fit_params):
        self.fit(X,y,**fit_params)
        self.transform(X)
