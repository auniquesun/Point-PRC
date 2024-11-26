import os
from plyfile import PlyData
import numpy as np

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class OmniObject3D(DatasetBase):

    def __init__(self, cfg):

        self.dataset_dir = cfg.DATASET.ROOT
        self.num_points = cfg.PointEncoder.num_points

        self.clsnames = []
        self.set_classnames()
        test_data, test_label = self.load_data()
        test = self.read_data(self.clsnames, test_data, test_label)

        super().__init__(train_x=test, val=test, test=test)

    def set_classnames(self):
        clsnames = []
        print('===', f'{self.dataset_dir}/{self.num_points}', '===')
        for cls in os.listdir(f'{self.dataset_dir}/{self.num_points}'):
            if os.path.isdir(os.path.join(f'{self.dataset_dir}/{self.num_points}', cls)):
                clsnames.append(cls)

        # NOTE `sort()` is important
        clsnames.sort()

        print('\n---len(clsnames):', len(clsnames), '---\n')
        for cls in clsnames:
            # print('>>>', cls)
            self.clsnames.append(cls)

    def load_data(self):
        all_data = []
        all_label = []

        data_dir1 = f'{self.dataset_dir}/{self.num_points}'
        for cls in os.listdir(data_dir1):
            data_dir2 = os.path.join(data_dir1, cls)
            for ins in os.listdir(data_dir2):
                data_dir3 = os.path.join(data_dir2, ins)
                if not os.listdir(data_dir3):   # empty dir
                    continue

                data_f = os.path.join(data_dir3, f'pcd_{self.num_points}.ply')
                plydata = PlyData.read(data_f)
                x = plydata.elements[0].data['x']
                y = plydata.elements[0].data['y']
                z = plydata.elements[0].data['z']
                # a whole point cloud
                pts = np.stack([x,y,z], axis=0).T
                # the label of the point cloud
                label = self.clsnames.index(cls)
                all_data.append(pts)
                all_label.append(label)

        all_data = np.array(all_data)
        all_label = np.array(all_label)
        
        return all_data, all_label

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
