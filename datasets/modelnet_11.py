import os
import numpy as np
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase, subsample_classes


@DATASET_REGISTRY.register()
class ModelNet_11(DatasetBase):
    """
    ModelNet_11 has 11 shared categories with ScanObjectNN
    """

    def __init__(self, cfg):

        self.dataset_dir = cfg.DATASET.ROOT
        classnames, name2idx = self.read_classnames(self.dataset_dir)

        # train_data: (num_pc, 2048, 3)     train_label: (num_pc)
        train_data, train_label = self.load_data(name2idx, 'train')
        test_data, test_label = self.load_data(name2idx, 'test')

        train = self.read_data(classnames, train_data, train_label)
        test = self.read_data(classnames, test_data, test_label)

        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # NOTE to compare with metasets (cvpr'21) and pdg (neurips'22), SUBSAMPLE_CLASSES is set to `all`
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, test=test)

    def load_data(self, name2idx, split):
        data_list, label_list = [], []
        for cls in os.listdir(self.dataset_dir):
            cls_dir = os.path.join(self.dataset_dir, cls)   # data/xset/sim2real/modelnet_11/bed
            
            if os.path.isdir(cls_dir):
                dir_f = os.path.join(cls_dir, split)    # data/xset/sim2real/modelnet_11/bed/train
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
