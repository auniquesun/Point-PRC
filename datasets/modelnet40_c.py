import os
import h5py
import numpy as np
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class ModelNet40_C(DatasetBase):
    """ModelNer40-C(orruption).

    This dataset is used for testing only.
    """

    def __init__(self, cfg):

        self.dataset_dir = cfg.DATASET.ROOT

        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        classnames = self.read_classnames(text_file)

        cor_type = cfg.DATASET.CORRUPTION_TYPE
        data_file = f'data_{cor_type}_1.npy'
        test_data, test_label = self.load_data(data_file)

        test = self.read_data(classnames, test_data, test_label)

        super().__init__(train_x=test, val=test, test=test)

    def load_data(self, data_file):
        test_data = np.load(f'{self.dataset_dir}/{data_file}')
        test_label = np.load(f'{self.dataset_dir}/label.npy')

        return test_data, test_label

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                classname = line.strip()
                classnames[i] = classname
        return classnames

    def read_data(self, classnames, datas, labels):
        items = []

        for i, data in enumerate(datas):
            label = int(labels[i])
            classname = classnames[label]

            item = Datum(
                impath=data,
                label=label,
                classname=classname,
                order = i
            )

            items.append(item)

        return items
