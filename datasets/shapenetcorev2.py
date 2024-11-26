import os
import h5py
import numpy as np
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase, subsample_classes


@DATASET_REGISTRY.register()
class ShapeNetCoreV2(DatasetBase):

    def __init__(self, cfg):

        self.dataset_dir = cfg.DATASET.ROOT

        text_file = os.path.join(self.dataset_dir, 'shape_names_with_rowid.txt')
        classnames = self.read_classnames(text_file)

        train_data, train_label = self.load_data(os.path.join(self.dataset_dir, 'train_files.txt'))
        val_data, val_label = self.load_data(os.path.join(self.dataset_dir, 'val_files.txt'))
        test_data, test_label = self.load_data(os.path.join(self.dataset_dir, 'test_files.txt'))

        train = self.read_data(classnames, train_data, train_label)
        val = self.read_data(classnames, val_data, val_label)
        test = self.read_data(classnames, test_data, test_label)

        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # NOTE this argument is updated by the command-line parameter
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def load_data(self, data_path):
        all_data = []
        all_label = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                all_data.append(data)
                all_label.append(label)
        # e.g., `ply_data_test0.h5`, `ply_data_test1.h5`, etc are concatenated here to form the complete test split
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)

        return all_data, all_label

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
