"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import os, math, cv2, tqdm, random, psutil
from multiprocessing.pool import ThreadPool
import numpy as np
import torch
import torch.utils.data

import torchvision

from PIL import Image
import faster_coco_eval
import faster_coco_eval.core.mask as coco_mask
from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register
from typing import Any, Callable, List, Optional, Tuple
from torchvision.datasets.vision import VisionDataset

torchvision.disable_beta_transforms_warning()
faster_coco_eval.init_as_pycocotools()
Image.MAX_IMAGE_PIXELS = None
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html

__all__ = ['CocoDetection']

class CocoDetection_Vision(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        cache_imgsz: int = None,
        ram_cache: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids_key = {x:i for i, x in enumerate(self.ids)}

        self.cache_imgsz = cache_imgsz
        self.ram_cache = ram_cache
        self.cache_flag = False
        self.ni = len(self.ids)
        self.ims, self.labs = [None] * self.ni, [None] * self.ni

        if self.ram_cache and self.check_cache_ram():
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(self.load_image_and_label_from_coco, self.ids)
                pbar = tqdm.tqdm(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
                for i, x in pbar:
                    d_id, d_ims, d_labs = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += np.array(self.ims[d_id]).nbytes
                    pbar.desc = f"Caching images ({b / gb:.1f}GB Ram"
                pbar.close()
            self.cache_flag = True

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        if self.cache_flag:
            image = self.ims[index]
            target = self.labs[index]
        else:
            image = self._load_image(id)
            target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def load_image_and_label_from_coco(self, i):
        im = self.ims[self.ids_key[i]]
        path = self.coco.loadImgs(i)[0]["file_name"]
        label = self.coco.loadAnns(self.coco.getAnnIds(i))
        if im is None:
            im = Image.open(os.path.join(self.root, path)).convert("RGB")
            im = np.array(im)

            h0, w0 = im.shape[:2]  # orig hw
            # resize long side to imgsz while maintaining aspect ratio
            r = self.cache_imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                w, h = (min(math.ceil(w0 * r), self.cache_imgsz), min(math.ceil(h0 * r), self.cache_imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

            # 对 label 进行缩放
            for ann in label:
                # 缩放边界框
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    ann['bbox'] = [
                        bbox[0] * r,  # x
                        bbox[1] * r,  # y
                        bbox[2] * r,  # width
                        bbox[3] * r   # height
                    ]

            self.ims[self.ids_key[i]] = Image.fromarray(im)
            self.labs[self.ids_key[i]] = label

        return self.ids_key[i], self.ims[self.ids_key[i]], self.labs[self.ids_key[i]]

    def check_cache_ram(self, safety_margin=0.5):
            """Check image caching requirements vs available memory."""
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            n = min(self.ni, 30)  # extrapolate from 30 random images
            for _ in range(n):
                im = cv2.imread(os.path.join(self.root, self.coco.loadImgs(random.choice(self.ids))[0]["file_name"]))  # sample image
                ratio = self.cache_imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
                b += im.nbytes * ratio**2
            mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
            mem = psutil.virtual_memory()
            cache = mem_required < mem.available  # to cache or not to cache, that is the question
            if not cache:
                print(
                    f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '  
                    f'with {int(safety_margin * 100)}% safety margin but only '
                    f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '     
                    f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
                )
            return cache

    def __len__(self) -> int:
        return len(self.ids)

@register()
class CocoDetection(CocoDetection_Vision, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category', 'cache_imgsz', 'ram_cache']

    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False, cache_imgsz=None, ram_cache=False):
        super(CocoDetection, self).__init__(img_folder, ann_file, cache_imgsz=cache_imgsz, ram_cache=ram_cache)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return image, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s

    @property
    def categories(self, ):
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
