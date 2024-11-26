import os
import numpy as np
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc


@DATASET_REGISTRY.register()
class Objaverse_LVIS(DatasetBase):
    def __init__(self, cfg):

        self.dataset_dir = cfg.DATASET.ROOT
        classnames = self.read_classnames(os.path.join(self.dataset_dir, 'classnames.txt'))

        text_file = os.path.join(self.dataset_dir, 'lvis_testset.txt')

        test_data, test_label = self.load_data(text_file)

        test = self.read_data(classnames, test_data, test_label)

        super().__init__(train_x=test, val=test, test=test)

        # 10,000 points for each pc by default
        self.num_points = cfg.PointEncoder.num_points

    @staticmethod
    def read_classnames(class_file):
        classnames = OrderedDict()
        cls_id = 0
        with open(class_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                classname = line.strip()
                if classname not in classnames:
                    classnames[cls_id] = classname
                    cls_id += 1
        return classnames
    
    def load_data(self, text_file):
        all_data = []
        all_label = []
    
        with open(text_file) as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip()
                label = int(line.split(',')[0])
                
                pc_path = self.dataset_dir + line.split(',')[-1]
                pc_data = np.load(pc_path, allow_pickle=True).item()
                pc_data = pc_data['xyz']

                all_data.append(pc_data)
                all_label.append(label)

        all_data = np.array(all_data).astype('float32')
        all_label = np.array(all_label).astype('int64')

        return all_data, all_label

    def read_data(self, classnames, data, labels):
        items = []

        for i, pc in enumerate(data):
            label = int(labels[i])
            classname = classnames[label]

            item = Datum(
                impath=pc,
                label=label,
                classname=classname,
                order = i
            )

            items.append(item)

        return items
