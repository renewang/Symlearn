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


class CalibratedClassifierCV(_calibration.CalibratedClassifierCV):
   
    @patch.object(_calibration, 'LabelBinarizer', new=_LabelBinarizer)
    def fit(self, X, y, sample_weight=None):
        return(super(CalibratedClassifierCV, self).fit(X, y, sample_weight=sample_weight))


class CalibratedClassifier(_calibration._CalibratedClassifier):

    def __init__(self, base_estimator=None, method='sigmoid', classes=None):
        self.method = method
        self.classes = classes
        self.base_estimator = base_estimator

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