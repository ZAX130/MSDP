# import math
import collections
import numpy as np


class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False):
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            shape = im.shape[1:dim+1]
            self.sample(*shape)

        if isinstance(img, collections.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

class Seg_norm(Base):
    def __init__(self, ):
        a = None
        self.seg_table = np.array([0, 1, 2, 3, 4])
    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        for i in range(len(self.seg_table)):
            img_out[img == self.seg_table[i]] = i
        return img_out


class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)
