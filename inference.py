import sys
sys.path.append("../")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, ToPILImage, PILToTensor

from model import UNet
from predict import Predictor
from data import NucleusDataset
from transform import Rescale
from metrics import iou_score
from losses import bce_and_dice
import json
import os
from scipy import ndimage
import cocoapi.PythonAPI.pycocotools.mask as co

# load test file
f = open("./data/test_img_ids.json")
test_imgs = json.load(f)

model160 = UNet.load("./models/test1/weights_e:100_loss:0.5448.pt")
model160.eval()
predict160 = Predictor(model160)
trans = Compose(
    [
        PILToTensor()
    ]
)
l = []
sum = 0
for img_cnt in range(6):
    img = Image.open(os.path.join("./data/test", test_imgs[img_cnt]["file_name"])).convert("RGB")
    img_arr = np.array(img)
    img_arr = cv2.resize(img_arr, (1024, 1024), cv2.INTER_AREA)
    # print(img_arr.shape)
    segmented = predict160(img_arr)
    segmented_res = np.array(segmented)
    segmented_res = cv2.resize(segmented_res, (1000, 1000), cv2.INTER_AREA)
    label_im, num_labels = ndimage.label(segmented_res)
    cv2.imwrite(str(img_cnt) + ".png", label_im)
    for i in range(num_labels):
        # generate single mask for an instance
        mask_compare = np.full(np.shape(label_im), i+1)
        separate_mask = np.equal(label_im, mask_compare).astype(int)

        # find bbox
        pos = np.where(separate_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        width = xmax - xmin
        height = ymax - ymin
        bbox= [float(xmin), float(ymin), float(width), float(height)]

        j = dict()
        j["image_id"] = img_cnt + 1
        j["bbox"] = bbox
        j["category_id"] = 1
        j["score"] = float(0.875)
        # print(separate_mask.shape)
        # separate_mask = separate_mask.reshape((1000,1000,1))
        rle = co.encode(np.asfortranarray(separate_mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        j["segmentation"] = rle
        l.append(j)
        res = co.decode(rle)
        # print(res.dtype)
        # print(np.where(res))
    sum = sum + num_labels
    print(num_labels)
print(sum)
print(l)
with open("answer.json", 'w') as f:
    f.write(json.dumps(l))