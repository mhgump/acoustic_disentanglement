import numpy as np


def load_normalization(encoding):
    class_name, encoding = encoding.split(';')
    if class_name == 'MeanVarianceNormalization':
        return MeanVarianceNormalization.load(encoding)
    if class_name == 'MaskedMeanVarianceNormalization':
        return MaskedMeanVarianceNormalization.load(encoding)
    raise KeyError('Normalization method {} not found'.format(class_name))


class Normalization:

    def __init__(self):
        raise NotImplementedError    

    def normalize(self, batch):
        raise NotImplementedError
    
    def add(self, sample):
        raise NotImplementedError

    def save(self):
        return '{};{}'.format(self.__class__.__name__, self._save())


class MeanVarianceNormalization(Normalization):

    def __init__(self):
        self._mean = None
        self._variance = None
        self._norm = None
        self.total = None
        self.N = None
        self.sq_total = None

    def add(self, sample):
        if self.N is None:
            self.N = np.array(sample.shape[0]).reshape(-1)
            self.total = np.array(np.sum(sample, axis=0)).reshape(-1)
            self.sq_total = np.array(np.sum(np.square(sample), axis=0)).reshape(-1)
        else:
            self.N += sample.shape[0]
            self.total += np.sum(sample, axis=0)
            self.sq_total += np.sum(np.square(sample), axis=0)

    @property
    def mean(self):
        if self._mean is None:
            if self.N.shape != self.total.shape: # N is a constant factor
                assert self.N.shape == (1,)
                if self.N[0] == 0:
                    self._mean = np.full_like(self.total, np.nan)
                else:
                    self._mean = self.total * 1. / self.N
            else: # N is computed per dimension
                valid_mask = 1 <= self.N 
                self._mean = np.full_like(self.total, np.nan)
                self._mean[valid_mask] = self.total[valid_mask] / self.N[valid_mask]
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            if self.N.shape != self.total.shape: # N is a constant factor
                assert self.N.shape == (1,)
                if self.N[0] <= 1:
                    self._variance = np.full_like(self.total, np.nan)
                else:
                    self._variance = (self.sq_total - np.square(self.total) * (1./self.N)) * (1./(self.N - 1))
            else: # N is computed per dimension
                valid_mask = 2 <= self.N 
                self._variance = np.full_like(self.total, np.nan)
                self._variance[valid_mask] = \
                    (self.sq_total[valid_mask] - \
                        self.total[valid_mask] * self.total[valid_mask] * np.reciprocal(1.*self.N[valid_mask])) \
                    * np.reciprocal(1.*self.N[valid_mask] - 1)
        return self._variance

    @property
    def norm(self):
        if self._norm is None:
            self._norm = np.reciprocal(np.sqrt(self.variance))
        return self._norm

    def normalize(self, batch):
        return (batch - self.mean) * self.norm

    def _save(self):
        mean = ','.join([str(e) for e in self.mean])
        variance = ','.join([str(e) for e in self.variance])
        return '{}:{}'.format(mean, variance)

    @classmethod
    def load(cls, encoding):
        mean, variance = encoding.split(':')
        obj = cls()
        obj._mean = np.array([float(e) for e in mean.split(',')])
        obj._variance = np.array([float(e) for e in variance.split(',')])
        return obj


class MaskedMeanVarianceNormalization(MeanVarianceNormalization):

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def add(self, sample):
        if self.N is None:
            self.N = np.array(np.sum(sample > self.threshold, axis=0)).reshape(-1)
            self.total = np.array(np.sum(sample, axis=0)).reshape(-1)
            self.sq_total = np.array(np.sum(np.square(sample), axis=0)).reshape(-1)
        else:
            self.N += np.sum(sample > self.threshold, axis=0)
            self.total += np.sum(sample, axis=0)
            self.sq_total += np.sum(np.square(sample), axis=0)

    def _save(self):
        mean = ','.join([str(e) for e in self.mean])
        variance = ','.join([str(e) for e in self.variance])
        return '{}:{}:{}'.format(self.threshold, mean, variance)

    @classmethod
    def load(cls, encoding):
        threshold, mean, variance = encoding.split(':')
        obj = cls(threshold)
        obj._mean = np.array([float(e) for e in mean.split(',')])
        obj._variance = np.array([float(e) for e in variance.split(',')])
        return obj
