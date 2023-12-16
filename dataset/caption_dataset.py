import json
import numpy as np
import time
import logging
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

import oss2
from io import BytesIO
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption

# 설명 : 데이터셋 클래스입니다.
class coco_dataset(Dataset):
    # 설명 : 데이터셋 클래스의 생성자입니다.
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True, add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        for each in self.ann:
            filename = each["filename"]
            sentences = each["sentences"]
            filepath = each["filepath"]
            if filepath == "val2014":
                file_root = "val2014_img"
            elif filepath == "train2014":
                file_root = "train2014_img"
            else:
                file_root = filepath
            image_path = os.path.join(file_root, filename)
            gold_caption = []
            for sent in sentences:
                caption = sent["raw"]
                gold_caption.append(caption.lower())
            if self.add_object:
                object_list = each["object_label"].split("&&")
                new_object_list = list(set(object_list))
                new_object_list.sort(key=object_list.index)
                object_label = " ".join(new_object_list) 
            else:
                object_label = ""
            if is_train:
                for sent in sentences:
                    caption = sent["raw"].lower()
                    self.ann_new.append({"image": image_path, "caption": caption, "gold_caption": gold_caption, "object_label": object_label})
            else:
                self.ann_new.append({"image": image_path, "caption": sentences[0]["raw"].lower(), "gold_caption": gold_caption, "object_label": object_label})
        self.ann = self.ann_new
        del self.ann_new
            
    # 설명 : 데이터셋의 길이를 반환합니다.
    def __len__(self):
        return len(self.ann)
    
    # 설명 : 데이터셋의 인덱스에 해당하는 데이터를 반환합니다.
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1] 
        object_label = ann['object_label']
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/"+ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, caption, object_label, image_id, ann["gold_caption"]
