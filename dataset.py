
import os

import cv2
from PIL import Image
import torch
from torch.utils import data
import numpy as np

def get_edge(img, kernel_size):
    [gy, gx] = np.gradient(img)
    edge = gy * gy + gx * gx
    edge[edge != 0.] = 1.
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    edge = cv2.dilate(edge, kernel, iterations=1)
    return edge

def randomCrop(image, label, flow, depth,edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    label = Image.fromarray(label)
    edge = Image.fromarray(edge)
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), np.array(label.crop(random_region)), flow.crop(random_region), depth.crop(
        random_region),np.array(edge.crop(random_region))


class Dataset(data.Dataset): 
    def __init__(self, datasets, mode='train', transform=None, return_size=True):
        self.return_size = return_size
        if type(datasets) != list:
            datasets = [datasets]
        self.datas_id = []
        self.mode = mode
        for (i, dataset) in enumerate(datasets):

            if mode == 'train':
                data_dir = './dataset/{}'.format(dataset)
                imgset_path = data_dir + '/train.txt'

            else:
                # data_dir = './vsod_dataset/test/{}'.format(dataset)
                data_dir = './dataset/{}'.format(dataset)
                imgset_path = data_dir + '/test.txt'

            imgset_file = open(imgset_path)

            for line in imgset_file:
                data = {}
                img_path = line.strip("\n").split(" ")[0]
                gt_path = line.strip("\n").split(" ")[1]
                data['img_path'] = data_dir + img_path
                data['gt_path'] = data_dir + gt_path
                # if dataset == 'DUTS-TR':
                #     data['split'] = dataset
                if dataset == 'DUTS-TR':
                    data['split'] = dataset
                    data['img_path'] = data_dir +'/'+ img_path
                    data['gt_path'] = data_dir +'/'+ gt_path
                    # DUTS Depth
                    # data['depth_path'] = data_dir + line.strip("\n").split(" ")[-1]
                else:
                    data['flow_path'] = data_dir + line.strip("\n").split(" ")[2]
                    data['depth_path'] = data_dir + line.strip("\n").split(" ")[3]
                    data['split'] = img_path.split('/')[-3]
                data['dataset'] = dataset
                self.datas_id.append(data)
        self.transform = transform

    def __getitem__(self, item):

        assert os.path.exists(self.datas_id[item]['img_path']), (
            '{} does not exist'.format(self.datas_id[item]['img_path']))
        assert os.path.exists(self.datas_id[item]['gt_path']), (
            '{} does not exist'.format(self.datas_id[item]['gt_path']))
        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            pass
            # DUTS Depth
            # assert os.path.exists(self.datas_id[item]['depth_path']), (
            #     '{} does not exist'.format(self.datas_id[item]['depth_path']))
        else:
            assert os.path.exists(self.datas_id[item]['depth_path']), (
                '{} does not exist'.format(self.datas_id[item]['depth_path']))
            assert os.path.exists(self.datas_id[item]['flow_path']), (
                '{} does not exist'.format(self.datas_id[item]['flow_path']))

        image = Image.open(self.datas_id[item]['img_path']).convert('RGB')
        label = Image.open(self.datas_id[item]['gt_path']).convert('L')
        edge = get_edge(label, 2)
        label = np.array(label)
        edge = np.array(edge)

        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            flow = np.zeros((image.size[1], image.size[0], 3))
            flow = Image.fromarray(np.uint8(flow))
            depth = np.zeros((image.size[1], image.size[0], 3))
            depth = Image.fromarray(np.uint8(depth))
            # DUTS Depth
            # depth = Image.open(self.datas_id[item]['depth_path']).convert('RGB')
        else:
            flow = Image.open(self.datas_id[item]['flow_path']).convert('RGB')
            depth = Image.open(self.datas_id[item]['depth_path']).convert('RGB')

        if label.max() > 0:
            label = label / 255

        if edge.max() > 0:
            edge = edge / 255

        w, h = image.size
        size = (h, w)

        sample = {'image': image, 'label': label, 'flow': flow, 'depth': depth,'edge': edge}
        if self.mode == 'train':
            sample['image'], sample['label'], sample['flow'], sample['depth'],sample['edge']= randomCrop(
                sample['image'],
                sample['label'],
                sample['flow'],
                sample['depth'],
                sample['edge']
                )
      
        else:
            pass
        # x = image.size()
        # y = depth.size()
        # x = flow.size()
        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)
        # if self.datas_id[item]['dataset'] == 'DUTS-TR':
        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            sample['flow'] = torch.zeros((3, 448, 448))
            sample['depth'] = torch.zeros((3, 448, 448))
            # sample['flow'] = torch.zeros((3, 352, 352))
            # DUTS Depth
            # sample['depth'] = torch.zeros((3, 352, 352))
        name = self.datas_id[item]['gt_path'].split('/')[-1]
        sample['dataset'] = self.datas_id[item]['dataset']
        sample['split'] = self.datas_id[item]['split']
        sample['name'] = name

        return sample

    def __len__(self):
        return len(self.datas_id)
