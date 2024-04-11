# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class Cityscapes(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4,
                 instance=True):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.instance = instance

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if not self.instance:
            self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                                1: ignore_label, 2: ignore_label, 
                                3: ignore_label, 4: ignore_label, 
                                5: ignore_label, 6: ignore_label, 
                                7: 0, 8: 1, 9: ignore_label, 
                                10: ignore_label, 11: 2, 12: 3, 
                                13: 4, 14: ignore_label, 15: ignore_label, 
                                16: ignore_label, 17: 5, 18: ignore_label, 
                                19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                                25: 12, 26: 13, 27: 14, 28: 15, 
                                29: ignore_label, 30: ignore_label, 
                                31: 16, 32: 17, 33: 18}
            self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                            1.0166, 0.9969, 0.9754, 1.0489,
                                            0.8786, 1.0023, 0.9539, 0.9843, 
                                            1.1116, 0.9037, 1.0865, 1.0955, 
                                            1.0865, 1.1529, 1.0507]).cuda()
        else:
            self.label_mapping = {-1: ignore_label, 0: 0, 
                                1: 1, 2: 2, 
                                3: 3, 4: 4, 
                                5: 5, 6: 6, 
                                7: 7, 8: 8, 9: 9, 
                                10: 10, 11: 11, 12: 12, 
                                13: 13, 14: 14, 15: 15, 
                                16: 16, 17: 17, 18: 18}
            self.class_weights = torch.FloatTensor([1, 1, 1, 1, 
                                            1, 1, 1, 1,
                                            1, 1, 1, 1, 
                                            1, 1, 1, 1, 
                                            1, 1, 1]).cuda()
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
    
    def convert_instance(self, label):
        temp = label.copy()
        mask = np.logical_and(temp >= 26000, temp < 27000)
        label = np.where(mask, label, -1)
        unique_intensities, counts = np.unique(label, return_counts=True)
        n = 19
        top_n_indices = np.argsort(counts)[-n-1:-1]
        top_n_intensities = unique_intensities[top_n_indices]
        mapping = {intensity: rank for rank, intensity in enumerate(top_n_intensities)}
        label = np.vectorize(lambda x: mapping.get(x, -1))(label)
        label = self.convert_label(label)
        label = self.label_transform(label)
        return label 
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name
        if not self.instance:
            label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
                            cv2.IMREAD_GRAYSCALE)
            label = self.convert_label(label)
        else:
            label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
                            cv2.IMREAD_ANYDEPTH)
            label = self.convert_instance(label)
        

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)
        np.save('label.npy', label)
        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
