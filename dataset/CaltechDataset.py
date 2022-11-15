"""

Contributed by Wenbin Li & Jinglin Xu

"""

import os
import os.path as path
import json
import torch
import torch.utils.data as data
import numpy as np
import random
import scipy.io as scio
from PIL import Image
torch.multiprocessing.set_sharing_strategy('file_system')

class Caltech20AttrData(object):
    """
        Dataloader for Caltech101 datasets
    """

    def __init__(self, data_dir='Caltech101-20.mat',
                mode='train'):

        super(Caltech20AttrData, self).__init__()

        X_list = scio.loadmat(data_dir)['X'][0].tolist()
        Y_list = scio.loadmat(data_dir)['Y'].tolist()              
        class_num_list = scio.loadmat(data_dir)['lenSmp'][0].tolist()      
        view_list = scio.loadmat(data_dir)['feanames'][0].tolist()    

        # Store the index of samples into the data_list
        data_list = []
        class_index_start = 0
        class_index_end = 0
        for iter, class_num in enumerate(class_num_list):
            print('The %d-th class: %d' % (iter, class_num))

            class_index_end += class_num
            sample_index = range(class_index_start, class_index_end)
            target = Y_list[class_index_start][0] 

            class_samples = []
            for i in range(len(sample_index)):
                sample_view_all = np.tile(sample_index[i], len(view_list))
                class_samples.append((sample_view_all, target))

            # divide the data into train, val and test
            random.seed(int(200)) 
            train_index = random.sample(range(0, class_num), int(0.6*class_num))
            rem_index = [rem for rem in range(0, class_num) if rem not in train_index]
            val_index = random.sample(rem_index, int(3/4.0*len(rem_index)))
            test_index = [rem for rem in rem_index if rem not in val_index]

            train_part = [class_samples[i] for i in train_index] 
            val_part = [class_samples[i] for i in val_index]
            test_part = [class_samples[i] for i in test_index]  

            class_index_start = class_index_end
            
            if mode == 'train':
                data_list.extend(train_part)
            elif mode == 'val':
                data_list.extend(val_part)
            else:
                data_list.extend(test_part)

        self.data_list = data_list   
        self.X_list = X_list


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        """
            Load an episode each time
        """
        X_list = self.X_list
        (sample_view_all, target) = self.data_list[index]
        Sample_Fea_Allviews = []

        for i in range(len(sample_view_all)):      

            sample_temp = np.array(X_list[i][sample_view_all[i]])
            sample_temp = sample_temp.astype(float) 
            sample_temp = torch.from_numpy(sample_temp)   
            Sample_Fea_Allviews.append(sample_temp.type(torch.FloatTensor))

        return (index, Sample_Fea_Allviews, target)


if __name__ == '__main__':
    trainset = Caltech20AttrData(data_dir='/mnt/3443FB571902BBEE/hyr/PycharmProject/MvLearning/mvdata/Caltech101-20.mat',
                              mode='test')
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=246, shuffle=True,
        num_workers=int(8), drop_last=True, pin_memory=True
    )
    Trainset = []
    train_iter = iter(train_loader)
    for index, (idx, sample_set, sample_targets) in enumerate(train_loader):

        print(len(sample_set), len(sample_targets))  # 6 32 view num->6 batch size->32
        print(sample_set[0].shape)  # single view -> torch.Size([32, 2688]) torch.Size([])
        print(sample_targets.shape)  # torch.Size([32])
        print(len(idx))
        print('idx',max(idx))
        print('idx', idx)
        print(sample_targets)   # tensor([32,  9,  9, 30, 27, 34, ..., 47, 24, 25, 32,  5, 41,  6, 46])
        scio.savemat('Caltech20test.mat',{'view1':sample_set[0].detach().cpu().numpy(),\
                                    'view2':sample_set[1].detach().cpu().numpy(),\
                                    'view3':sample_set[2].detach().cpu().numpy(),\
                                    'view4':sample_set[3].detach().cpu().numpy(),\
                                    'view5':sample_set[4].detach().cpu().numpy(),\
                                    'view6':sample_set[5].detach().cpu().numpy()})
        scio.savemat('testlabel.mat',
                     {'testlabel': sample_targets.detach().cpu().numpy()})
        scio.savemat('testidx.mat',
                     {'testidx': idx.detach().cpu().numpy()})
