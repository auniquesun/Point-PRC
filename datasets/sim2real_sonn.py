import os
import numpy as np
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class Sim2Real_SONN(DatasetBase):

    def __init__(self, cfg):
        """
        This dataset is used for testing only.
        """

        self.dataset_dir = cfg.DATASET.ROOT
        self.variant = cfg.DATASET.SONN_VARIANT

        classnames, name2idx = self.read_classnames(self.dataset_dir)
        test_data, test_label = self.load_data(name2idx, 'test')

        test = self.read_data(classnames, test_data, test_label)

        super().__init__(train_x=test, test=test)

    def load_data(self, name2idx, split):
        data_list, label_list = [], []
        for cls in os.listdir(self.dataset_dir):
            cls_dir = os.path.join(self.dataset_dir, cls)   # data/xset/sim2real/shapenet_9/bed
            
            if os.path.isdir(cls_dir):
                dir_f = os.path.join(cls_dir, split)    # data/xset/sim2real/shapenet_9/bed/train
                label = name2idx[cls]

                for f in os.listdir(dir_f):
                    # shape: (2048, 3) -> (1, 2048, 3)
                    points = np.expand_dims(np.load(os.path.join(dir_f, f)), axis=0)
                    data_list.append(points)
                    label_list.append([label])

        data = np.concatenate(data_list, axis=0).astype('float32')
        label = np.array(label_list).astype("int64")

        return data, label

    @staticmethod
    def read_classnames(dataset_dir):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        name2idx = dict()

        names = sorted(os.listdir(dataset_dir))

        for idx, name in enumerate(names):
            if os.path.isdir(os.path.join(dataset_dir, name)):
                classnames[idx] = name
                name2idx[name] = idx
        
        return classnames, name2idx

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
