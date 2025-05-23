import os
import h5py
import numpy as np
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase, subsample_classes


@DATASET_REGISTRY.register()
class PointDA_ScanNet(DatasetBase):
    """ NOTE data augmentations are added in the `dassl` codebase 
            each point has 6 dimension: xyz, rgb
    """
    def __init__(self, cfg):

        self.dataset_dir = cfg.DATASET.ROOT

        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        classnames = self.read_classnames(text_file)

        train_data, train_label = self.load_data(os.path.join(self.dataset_dir, 'train_files.txt'))
        test_data, test_label = self.load_data(os.path.join(self.dataset_dir, 'test_files.txt'))

        train = self.read_data(classnames, train_data, train_label)
        test = self.read_data(classnames, test_data, test_label)

        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # 搞清楚这个参数是哪来的，运行脚本命令行传入的
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, test=test)

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
        # takes only first 3 dimensions which contain coordinates
        all_data = np.concatenate(all_data, axis=0)[:, :, :3]
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
