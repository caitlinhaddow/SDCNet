from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import torch
from utils_test import cropping, cropping_ohaze

class dehaze_val_dataset(Dataset):
    def __init__(self, test_dir, crop_method):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test = os.listdir(test_dir)
        self.list_test.sort()
        self.root_hazy = test_dir
        self.file_len = len(self.list_test)
        self.crop_method = crop_method
        # print(self.list_test)

    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.root_hazy +'/'+ self.list_test[index])
        hazy = self.transform(hazy)

        if hazy.shape[0] == 4:
            assert torch.equal(hazy[-1:, :, :], torch.ones(1, hazy.shape[1], hazy.shape[2])), "hazy[-1:, :, :] is not all ones"
            hazy = hazy[:3, :, :]

        hazy, vertical = cropping(hazy, self.crop_method)

        return hazy, vertical

    def __len__(self):
        return self.file_len
    
class dehaze_val_dataset_ohaze(Dataset):
    def __init__(self, test_dir, crop_method):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test = os.listdir(test_dir)
        self.list_test.sort()
        self.root_hazy = test_dir
        self.file_len = len(self.list_test)
        self.crop_method = crop_method
        # print(self.list_test)

    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.root_hazy +'/'+ self.list_test[index])
        hazy = self.transform(hazy)
        hazy_shape = hazy.shape

        if hazy.shape[0] == 5:
            assert torch.equal(hazy[-1:, :, :], torch.ones(1, hazy.shape[1], hazy.shape[2])), "hazy[-1:, :, :] is not all ones"
            hazy = hazy[:3, :, :]

        hazy = cropping_ohaze(hazy, index)
        
        return hazy, (index, hazy_shape)

    def __len__(self):
        return self.file_len
    
class dehaze_val_dataset_overlap(Dataset):
    def __init__(self, test_dir, crop_method):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test = os.listdir(test_dir)
        self.list_test.sort()
        self.root_hazy = test_dir
        self.file_len = len(self.list_test)
        self.crop_method = crop_method
        # print(self.list_test)

    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.root_hazy +'/'+ self.list_test[index])
        hazy = self.transform(hazy)

        if hazy.shape[0] == 4:
            assert torch.equal(hazy[-1:, :, :], torch.ones(1, hazy.shape[1], hazy.shape[2])), "hazy[-1:, :, :] is not all ones"
            hazy = hazy[:3, :, :]


        return hazy

    def __len__(self):
        return self.file_len



class dehaze_val_dataset_depth(Dataset):
    def __init__(self, test_dir, crop_method):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test = []
        for line in open(os.path.join(test_dir, '2024_val.txt')):
            line = line.strip('\n')
            if line != '':
                self.list_test.append(line)
        self.root_hazy = os.path.join(test_dir, '2024_val/')
        self.root_depth = os.path.join(test_dir, '2024_val_depth/')
        self.file_len = len(self.list_test)
        self.crop_method = crop_method

    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.root_hazy + self.list_test[index])
        depth = Image.open(self.root_depth + self.list_test[index])
        hazy = self.transform(hazy)
        depth = self.transform(depth)

        if hazy.shape[0] == 4:
            assert torch.equal(hazy[-1:, :, :],
                               torch.ones(1, hazy.shape[1], hazy.shape[2])), "hazy[-1:, :, :] is not all ones"
            hazy = hazy[:3, :, :]

        hazy, vertical_1 = cropping(hazy, self.crop_method)
        depth, vertical_2 = cropping(depth, self.crop_method)


        return hazy, depth, vertical_1, vertical_2

    def __len__(self):
        return self.file_len



class dehaze_test_dataset_9(Dataset):
    def __init__(self, val_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test_hazy = []

        self.root_hazy = os.path.join(val_dir, '')
        for i in os.listdir(self.root_hazy):

            self.list_test_hazy.append(i)

        # self.root_hazy = os.path.join(test_dir)

        self.file_len = len(self.list_test_hazy)
        print(self.file_len)
    def __getitem__(self, index, is_train=True):

        hazy = Image.open(self.root_hazy + self.list_test_hazy[index])
        hazy = hazy.convert('RGB')
        print(hazy.mode)
        hazy = self.transform(hazy)
        if hazy.shape[1] < hazy.shape[2]:
            hazy_up_left = hazy[:, 0:1600, 0:2432]
            hazy_up_middle = hazy[:, 0:1600, 1800:4232]
            hazy_up_right = hazy[:, 0:1600, 3568:6000]

            hazy_middle_left = hazy[:, 1200:2800, 0:2432]
            hazy_middle_middle = hazy[:, 1200:2800, 1800:4232]
            hazy_middle_right = hazy[:, 1200:2800, 3568:6000]

            hazy_down_left = hazy[:, 2400:4000, 0:2432]
            hazy_down_middle = hazy[:, 2400:4000, 1800:4232]
            hazy_down_right = hazy[:, 2400:4000, 3568:6000]

            name = self.list_test_hazy[index]

        if hazy.shape[1] > hazy.shape[2]:
            hazy_up_left = hazy[:, 0:2432, 0:1600]
            hazy_up_middle = hazy[:, 0:2432, 1200:2800]
            hazy_up_right = hazy[:, 0:2432, 2400:]

            hazy_middle_left = hazy[:, 1800:4232, 0:1600]
            hazy_middle_middle = hazy[:, 1800:4232, 1200:2800]
            hazy_middle_right = hazy[:, 1800:4232, 2400:]

            hazy_down_left = hazy[:, 3568:6000, 0:1600]
            hazy_down_middle = hazy[:, 3568:6000, 1200:2800]
            hazy_down_right = hazy[:, 3568:6000, 2400:]

            name = self.list_test_hazy[index]



        return hazy_up_left, hazy_up_middle, hazy_up_right, hazy_middle_left, hazy_middle_middle, hazy_middle_right, hazy_down_left, hazy_down_middle, hazy_down_right, name

    def __len__(self):
        return self.file_len


