# flake8: noqa
from ..datasets.base import (Dataset, IterableDataset,
                                IndexableDataset)

from ..datasets.hdf5 import H5PYDataset
from ..datasets.adult import Adult
from ..datasets.binarized_mnist import BinarizedMNIST
from ..datasets.cifar10 import CIFAR10
from ..datasets.cifar100 import CIFAR100
from ..datasets.caltech101_silhouettes import CalTech101Silhouettes
from ..datasets.iris import Iris
from ..datasets.mnist import MNIST
from ..datasets.svhn import SVHN
from ..datasets.text import TextFile
from ..datasets.billion import OneBillionWord
