import sklearn.metrics as metrics 
import numbers
import numpy

class JointScore(numbers.Number):
    __slots__ = ('target', 'predict')

    def __init__(self, target_score, predict_score=None):
        self.target = target_score
        if predict_score is None:
            self.predict =  self.target
        else:
            self.predict = predict_score

    def __ne__(self, other):
        """

        >>> (JointScore(-0.1, 0.9) != JointScore(-0.1, 0.2)) == (-0.1 != -0.1)
        True
        >>> (JointScore(-0.1, 0.2) != JointScore(-0.3, 0.2)) == (-0.1 != -0.3)
        True
        """
        return(not self == other)

    def __eq__(self, other):
        """

        >>> (JointScore(-0.1, 0.9) == JointScore(-0.1, 0.2)) == (-0.1 == -0.1)
        True
        >>> (JointScore(-0.1, 0.2) == JointScore(-0.3, 0.2)) == (-0.1 == -0.3)
        True
        """
        return(self.target == other.target)

    def __lt__(self, other):
        """

        >>> (JointScore(0.9, 0.9) < JointScore(0.1, 0.2)) == (0.9 < 0.1)
        True
        >>> (JointScore(0.1, 0.9) < JointScore(0.9, 0.2)) == (0.1 < 0.9)
        True
        >>> (JointScore(-0.9, 0.9) < JointScore(-0.1, 0.2)) == (-0.9 < -0.1)
        True
        >>> (JointScore(-0.1, 0.2) < JointScore(-0.9, 0.9)) == (-0.1 < -0.9)
        True
        >>> (JointScore(-0.1, 0.9) < JointScore(-0.1, 0.2)) == (-0.1 < -0.1)
        True
        """
        return(self.target < other.target)

    def __gt__(self, other):
        """

        >>> (JointScore(0.9, 0.9) > JointScore(0.1, 0.2)) == (0.9 > 0.1)
        True
        >>> (JointScore(0.1, 0.9) > JointScore(0.9, 0.2)) == (0.1 > 0.9)
        True
        >>> (JointScore(-0.9, 0.9) > JointScore(-0.1, 0.2)) == (-0.9 > -0.1)
        True
        >>> (JointScore(-0.1, 0.2) > JointScore(-0.9, 0.9)) == (-0.1 > -0.9)
        True
        >>> (JointScore(-0.1, 0.9) > JointScore(-0.1, 0.2)) == (-0.1 > -0.1)
        True
        """
        return(not self <= other)

    def __le__(self, other):
        """

        >>> (JointScore(-0.1, 0.9) <= JointScore(-0.1, 0.2)) == (-0.1 <= -0.1)
        True
        >>> (JointScore(-0.9, 0.9) <= JointScore(-0.1, 0.2)) == (-0.9 <= -0.1)
        True
        """
        return(self == other or self < other)

    def __ge__(self, other):
        """

        >>> (JointScore(-0.1, 0.9) >= JointScore(-0.1, 0.2)) == (-0.1 >= -0.1)
        True
        >>> (JointScore(-0.1, 0.2) >= JointScore(-0.9, 0.9)) == (-0.1 >= -0.9)
        True
        """
        return(self == other or self > other)

    def __hash__(self):
        raise ValueError('JointScorer instance is mutable and cannot be hashed')

    def __add__(self, other):
        """

        >>> (JointScore(-0.1, 0.9) + JointScore(-0.1, 0.2)) == (JointScore(-0.1+-0.1, 0.2+0.9))
        True
        >>> (JointScore(0.1, 0.2) + 0.1) == JointScore(0.1+0.1, 0.2+0.1)
        True
        """
        if isinstance(other, JointScore):
            return(JointScore(self.target + other.target, self.predict +
                other.predict))  
        elif isinstance(other, (float, int)):
            return(JointScore(self.target + other, self.predict +
                other))  

    def __sub__(self, other):
        """

        >>> (JointScore(-0.1, 0.9) - JointScore(-0.1, 0.2))== JointScore(-0.1-(-0.1), 0.9-0.2)
        True
        """
        return(JointScore(self.target - other.target, self.predict -
            other.predict))  

    def __truediv__(self, other):
        """
         
        >>> (JointScore(0.1, 0.9) / JointScore(0.1, 0.9))== JointScore(0.1/0.1, 0.9/0.9)
        True
        >>> (JointScore(0.1, 0.9) / 10)== JointScore(0.1/10, 0.9/10)
        True
        """
        if isinstance(other, JointScore):
            return(JointScore(self.target/other.target,
                self.predict/other.predict))
        elif isinstance(other, (int, float)):
            return(JointScore(self.target/other,
                self.predict/other))
        else:
            return NotImplementedError("cannot take other than JointScore,"
                    "int or float object")

    def __rtruediv__(self, other):
        """

        >>> (1 / JointScore(0.1, 0.5)) == JointScore(1/0.1, 1/0.5)
        True
        """
        if isinstance(other, (int, float)):
            return(JointScore(other/self.target,
                other/self.predict))

    def __mul__(self, other):
        """

        >>> (JointScore(0.1, 0.9) * JointScore(0.9, 0.1)) == JointScore(0.1*0.9, 0.9*0.1)
        True
        >>> ((JointScore(0.1, 0.9) * 0.1))==JointScore(0.1*0.1, 0.1*0.9)
        True
        """
        if isinstance(other, JointScore):
            return(JointScore(self.target * other.target,
                self.predict * other.predict))
        elif isinstance(other, (int, float)):
            return(JointScore(self.target * other,
                self.predict * other))

    def __rmul__(self, other):
        """

        >>> (0.1 * JointScore(0.1, 0.9)) == JointScore(0.1*0.1, 0.1*0.9)
        True
        """
        return(self.__mul__(other))

    def __repr__(self):
        return("{}(target={:.3f}, predict={:.3f})".format(
            self.__class__.__name__, self.target, self.predict))

    def __float__(self):
        """
        will return real representation of target score

        >>> float(JointScore(0.1, 0.9))
        0.1
        >>> print("%.2f" % JointScore(0.9, 0.1))
        0.90
        """
        return(self.target)

    def __int__(self):
        """
        will return percent representation of predict score

        >>> int(JointScore(0.1, 0.8))
        80
        >>> print("%d" % JointScore(0.1, 0.5))
        50
        """
        return(round(self.predict * 100))

    def __iadd__(self, other):
        """
        inplace addition
        >>> s = JointScore(0.1, 0.5) 
        >>> s+=JointScore(0.2, 0.3)
        >>> s == JointScore(0.1+0.2, 0.5+0.3)
        True
        >>> s+=0.1
        >>> s == JointScore(0.1+0.2+0.1, 0.5+0.3+0.1)
        True
        """
        return(self + other)

    def __imul__(self, other):
        """
        inplace multiplication 
        >>> s = JointScore(0.1, 0.5) 
        >>> s *= JointScore(0.2, 0.3)
        >>> s == JointScore(0.1*0.2, 0.5*0.3)
        True
        >>> s *= 0.1
        >>> s == JointScore(0.1*0.2*0.1, 0.5*0.3*0.1)
        True

        """
        return(self*other)

    def __radd__(self, other):
        """
        >>> (0.1 + JointScore(0.1, 0.5)) == JointScore(0.1+0.1, 0.1+0.5)
        True
        """
        return(self.__add__(other))

class JointScorer(metrics.scorer._ProbaScorer):
    """
    TODO: remove JointScorer in hyperopt_ext.hyperopt_search after fully
    tested
    """

    def __init__(self, scoring, greater_is_better=True, **kwargs):
        """
        """
        assert(type(scoring) is str)
        if scoring.endswith('loss'):
            greater_is_better = False
        if greater_is_better == True:
            sign = 1
        else:
            sign = -1 
        super(__class__, self).__init__(
                getattr(metrics, scoring), sign, kwargs)

    def __call__(self, clf, X, y, sample_weight=None):
        if not isinstance(y, numpy.ndarray):
            y = numpy.asarray(y)
        y_proba = clf.predict_proba(X)
        kws = {'sample_weight': sample_weight}
        kws.update(self._kwargs)
        if y.ndim == 2:
            y_true = numpy.argmax(y, axis=1) 
        else:
            assert(y.ndim == 1)
            y_true = y
        y_pred = clf.predict(X)
        predict_score = metrics.zero_one_loss(y_true, y_pred, **kws)
        if self._score_func.__name__ != 'zero_one_loss':
           target_score = self._score_func(y_true, y_proba, **kws)
        else:
           target_score = predict_score # copy from predict_score
        return(JointScore(target_score, predict_score))
