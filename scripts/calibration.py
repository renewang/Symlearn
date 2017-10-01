import numpy
import sklearn.calibration as _calibration

from sklearn.preprocessing import label_binarize, LabelBinarizer
from unittest.mock import patch

n_classes_ = 5

class _LabelBinarizer(LabelBinarizer):
    # need to move to global scope

    def fit(self, y):
        # able to fit smaller data set
        self = super(_LabelBinarizer, self).fit(y)
        if numpy.any(self.classes_ != numpy.arange(n_classes_)):
            self.classes_ = numpy.arange(n_classes_)
        return self


class CalibratedClassifier(_calibration._CalibratedClassifier):

    def __init__(self, base_estimator=None, method='sigmoid', classes=None, normalized=True):
        self.method = method
        self.classes = classes
        self.base_estimator = base_estimator
        self.normalized = normalized

    def _preproc(self, X):
        """
        return predicting probability and one hot encoding labels
        """
        n_classes = len(self.classes_)
        if self.base_estimator:
            return super(CalibratedClassifier, self)._preproc(X)
        # X as probas
        if X.ndim == 1:
            df = X[:, numpy.newaxis]
        elif n_classes == 2:
            df = X[:, 1:]
        else:
            df = X

        idx_pos_class = self.label_encoder_.transform(self.classes_)
        return df, idx_pos_class

    def predict_proba(self, X): 
        """make a simple modification from sklearn.CalibratedClassifier.predict_proba
        """
        n_classes = len(self.classes_)
        proba = numpy.zeros((X.shape[0], n_classes))

        df, idx_pos_class = self._preproc(X)

        for k, this_df, calibrator in \
                zip(idx_pos_class, df.T, self.calibrators_):
            if n_classes == 2:
                k += 1
            proba[:, k] = calibrator.predict(this_df)

        # Normalize the probabilities
        if n_classes == 2:
            proba[:, 0] = 1. - proba[:, 1]
        else:
            if self.normalized:
                proba /= numpy.sum(proba, axis=1)[:, numpy.newaxis]

        # XXX : for some reason all probas can be 0
        proba[numpy.isnan(proba)] = 1. / n_classes

        # Deal with cases where the predicted probability minimally exceeds 1.0
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0 

        return proba


class CalibratedClassifierCV(_calibration.CalibratedClassifierCV):
   
    def __init__(self, base_estimator=None, method='sigmoid', cv=3, normalized=False):
        super(CalibratedClassifierCV, self).__init__(base_estimator=base_estimator,
            method=method, cv=cv)
        self.normalized = normalized

    @patch.object(_calibration, '_CalibratedClassifier', new=CalibratedClassifier)
    @patch.object(_calibration, 'LabelBinarizer', new=_LabelBinarizer)
    def fit(self, X, y, sample_weight=None):
        return(super(CalibratedClassifierCV, self).fit(X, y, sample_weight=sample_weight))


    def predict_proba(self, X):
        # align normalized attribute with CV's
        for calibrated_classifier in self.calibrated_classifiers_:
            if calibrated_classifier.normalized != self.normalized:
                calibrated_classifier.normalized = self.normalized
        return super(CalibratedClassifierCV, self).predict_proba(X)