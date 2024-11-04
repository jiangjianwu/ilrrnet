"""
Load the indoor context data set

Date: 2023.10.21

Author: fengbuxi@glut.edu.cn

"""
import os
import torch
import numpy as np
from PIL import Image
from .icr_classes import *
from .segbase import SegmentationDataset

class IndoorContext(SegmentationDataset):
    def __init__(self, root='../datasets/', split='train', transform=None, **kwargs):
        super(IndoorContext, self).__init__(root, split, transform, **kwargs)
        assert os.path.exists(root), "The data set path does not exist !"
        self.mode = split
        self.images, self.scenes, self.times, self.peoples, self.accesses = self._get_data_pairs(root, split)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in folders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        if self.mode == 'train':
            img = self._sync_transform_no_mask(img)
        elif self.mode == 'val':
            img = self._val_sync_transform_no_mask(img)
        else:
            assert self.mode == 'testval'
            img = self._img_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.scenes[index], self.times[index], self.peoples[index], self.accesses[index]

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    @property
    def scene_classes(self):
        return uscenes
    
    @property
    def sclasses(self): # 场景重识别的场景类别
        return scene_classes
    
    @property
    def _times(self):
        return utimes
    
    @property
    def _peoples(self):
        return upeoples
    
    @property
    def _accesses(self):
        return uaccesses
    
    @property
    def _layouts(self):
        return ulayouts

    def _get_data_pairs(self, folder, split='train'):
        img_paths = [] # img paths
        scenes = [] # scene categories 
        times = [] # day:1 night:4
        peoples = [] # somebody:1 nobody:0
        accesses = [] # passable:1 impassable:0
        if split == 'train':
            img_folder = os.path.join(folder, 'train')
        else:
            img_folder = os.path.join(folder, 'test')

        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                if os.path.isfile(imgpath):
                    tmp = basename.split('_')
                    img_paths.append(imgpath)
                    scenes.append(self.scene_classes[tmp[0]])
                    times.append(self._times[tmp[1]])
                    peoples.append(self._peoples[tmp[2]])
                    accesses.append(self._accesses[tmp[3]])
                else:
                    print('cannot find the image:', imgpath)

        return img_paths, scenes, times, peoples, accesses

class IndoorContextV2(SegmentationDataset):
    def __init__(self, root='../datasets/', split='train', transform=None, **kwargs):
        super(IndoorContextV2, self).__init__(root, split, transform, **kwargs)
        assert os.path.exists(root), "The data set path does not exist !"
        self.mode = split
        # self.images, self.semsegs, self.rationals, self.scenes, self.times, self.peoples, self.accesses = self._get_data_pairs(root, split)
        self.images, self.semsegs, self.rationals = self._get_data_pairs(root, split)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in folders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        semseg = Image.open(self.semsegs[index])
        rational = Image.open(self.rationals[index])
        if self.mode == 'train':
            img, semseg, rational = self._sync_transform(img, semseg, rational)
        elif self.mode == 'val':
            img, semseg, rational = self._val_sync_transform(img, semseg, rational)
        else:
            assert self.mode == 'testval'
            img, semseg, rational = self._img_transform(img), self._mask_transform(semseg), self._mask_transform(rational)

        if self.transform is not None:
            img = self.transform(img)

        # return img, semseg, rational, self.scenes[index], self.times[index], self.peoples[index], self.accesses[index], os.path.basename(self.images[index])
        return img, semseg, rational, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    @property
    def scene_classes(self):
        return uscenes
    
    @property
    def _times(self):
        return utimes
    
    @property
    def _peoples(self):
        return upeoples
    
    @property
    def _accesses(self):
        return uaccesses
    
    @property
    def _elements(self):
        return uelements
    
    @property
    def _layouts(self):
        return ulayouts


    def _get_data_pairs(self, folder, split='train'):
        img_paths = [] # img paths
        semseg_paths = [] # semseg mask paths
        rational_paths = [] # layout mask paths
        # scenes = [] # scene categories 
        # times = [] # day:1 night:4
        # peoples = [] # somebody:1 nobody:0
        # accesses = [] # passable:1 impassable:0_ade20k
        if split == 'train':
            img_folder = os.path.join(folder, 'layout', 'images', 'training') # The semantic segmentation task and layout rational recognition task use the same images
            semseg_folder = os.path.join(folder, 'semseg', 'annotations', 'training')
            rational_folder = os.path.join(folder, 'layout', 'annotations', 'training')
        else:
            img_folder = os.path.join(folder, 'layout', 'images', 'validation')
            semseg_folder = os.path.join(folder, 'semseg', 'annotations', 'validation')
            rational_folder = os.path.join(folder, 'layout', 'annotations', 'validation')

        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                semsegname = basename + '.png'
                semsegpath = os.path.join(semseg_folder, semsegname)
                rationalpath = os.path.join(rational_folder, semsegname)
                if os.path.isfile(imgpath):
                    tmp = basename.split('_')
                    img_paths.append(imgpath)
                    semseg_paths.append(semsegpath)
                    rational_paths.append(rationalpath)
                    # scenes.append(self.scene_classes[tmp[0]])
                    # times.append(self._times[tmp[1]])
                    # peoples.append(self._peoples[tmp[2]])
                    # accesses.append(self._accesses[tmp[3]])
                else:
                    print('cannot find the image:', imgpath)

        # return img_paths, semseg_paths, rational_paths, scenes, times, peoples, accesses
        return img_paths, semseg_paths, rational_paths

# 场景重识别数据集构建
class IndoorContextV4(SegmentationDataset):
    def __init__(self, root='../datasets/', split='train', transform=None, **kwargs):
        super(IndoorContextV4, self).__init__(root, split, transform, **kwargs)
        assert os.path.exists(root), "The data set path does not exist !"
        self.mode = split
        self.images, self.scenes, self.instances, self.instances2label = self._get_data_pairs(root, split)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in folders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        if self.mode == 'train':
            img = self._sync_transform_no_mask(img)
        elif self.mode == 'val':
            img = self._val_sync_transform_no_mask(img)
        elif self.mode == 'quary':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
        else:
            assert self.mode == 'testval'
            img = self._img_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.scenes[index], self.instances[index]

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    @property
    def scene_classes(self):
        return scene_classes
    
    @property
    def instances_classes(self):
        return len(self.instances2label)

    def _get_data_pairs(self, folder, split='train'):
        img_paths = [] # img paths
        scenes = [] # scene categories 
        instances = [] # instance
        instances_container = set() # instance_container
        if split == 'train':
            img_folder = os.path.join(folder, 'training')
        elif split == 'quary':
            img_folder = os.path.join(folder, 'quary')
        else:
            img_folder = os.path.join(folder, 'validation')
        
        # 获取场景实例数量
        for filename in os.listdir(img_folder):
           basename, _ = os.path.splitext(filename)
           if filename.endswith(".jpg"):
               imgpath = os.path.join(img_folder, filename)
               if os.path.isfile(imgpath):
                   tmp = basename.split('_')
                   instances_container.add(int(tmp[1]))
        instances2label = {pid:label for label, pid in enumerate(instances_container)}    
         
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                if os.path.isfile(imgpath):
                    tmp = basename.split('_')
                    img_paths.append(imgpath) # 图片
                    scenes.append(self.scene_classes[tmp[0]]) # 场景类别
                    instances.append(instances2label[int(tmp[1])]) # 实例类别
                else:
                    print('cannot find the image:', imgpath)

        return img_paths, scenes, instances, instances2label


class IndoorContextV5(SegmentationDataset):
    BASE_DIR = ''
    NUM_CLASS = 150 # 语义类别数
    SCENE_CLASS = 6 # 场景数

    def __init__(self, root='../datasets/', split='train', transform=None, **kwargs):
        super(IndoorContextV5, self).__init__(root, split, transform, **kwargs)
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please setup the dataset using ../datasets/ade20k.py"
        self.images, self.masks, self.scenes = self._get_ade20k_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask, _ = self._sync_transform(img, mask, mask)
        elif self.mode == 'val':
            img, mask, _ = self._val_sync_transform(img, mask, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        # return img, mask, os.path.basename(self.images[index]), self.scenes[index]
        return img, mask, self.scenes[index]

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def scene_classes(self):
        '''Scene Category'''
        # 'bedroom','parlor','kitchen','office','conference_room','shop'
        return {
            "bedroom":0,
            "parlor":1,
            "kitchen":2,
            "office":3,
            "conference":4,
            "shop":5
        }
        # return ("bedroom", "parlor", "kitchen", "restaurant", "office", "conference")

    @property
    def classes(self):
        """Category names."""
        return ("wall", "building, edifice", "sky", "floor, flooring", "tree",
                "ceiling", "road, route", "bed", "windowpane, window", "grass",
                "cabinet", "sidewalk, pavement",
                "person, individual, someone, somebody, mortal, soul",
                "earth, ground", "door, double door", "table", "mountain, mount",
                "plant, flora, plant life", "curtain, drape, drapery, mantle, pall",
                "chair", "car, auto, automobile, machine, motorcar",
                "water", "painting, picture", "sofa, couch, lounge", "shelf",
                "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair",
                "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press",
                "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion",
                "base, pedestal, stand", "box", "column, pillar", "signboard, sign",
                "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink",
                "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox",
                "grandstand, covered stand", "path", "stairs, steps", "runway",
                "case, display case, showcase, vitrine",
                "pool table, billiard table, snooker table", "pillow",
                "screen door, screen", "stairway, staircase", "river", "bridge, span",
                "bookcase", "blind, screen", "coffee table, cocktail table",
                "toilet, can, commode, crapper, pot, potty, stool, throne",
                "flower", "book", "hill", "bench", "countertop",
                "stove, kitchen stove, range, kitchen range, cooking stove",
                "palm, palm tree", "kitchen island",
                "computer, computing machine, computing device, data processor, "
                "electronic computer, information processing system",
                "swivel chair", "boat", "bar", "arcade machine",
                "hovel, hut, hutch, shack, shanty",
                "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
                "motorcoach, omnibus, passenger vehicle",
                "towel", "light, light source", "truck, motortruck", "tower",
                "chandelier, pendant, pendent", "awning, sunshade, sunblind",
                "streetlight, street lamp", "booth, cubicle, stall, kiosk",
                "television receiver, television, television set, tv, tv set, idiot "
                "box, boob tube, telly, goggle box",
                "airplane, aeroplane, plane", "dirt track",
                "apparel, wearing apparel, dress, clothes",
                "pole", "land, ground, soil",
                "bannister, banister, balustrade, balusters, handrail",
                "escalator, moving staircase, moving stairway",
                "ottoman, pouf, pouffe, puff, hassock",
                "bottle", "buffet, counter, sideboard",
                "poster, posting, placard, notice, bill, card",
                "stage", "van", "ship", "fountain",
                "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
                "canopy", "washer, automatic washer, washing machine",
                "plaything, toy", "swimming pool, swimming bath, natatorium",
                "stool", "barrel, cask", "basket, handbasket", "waterfall, falls",
                "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle",
                "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",
                "trade name, brand name, brand, marque", "microwave, microwave oven",
                "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna",
                "bicycle, bike, wheel, cycle", "lake",
                "dishwasher, dish washer, dishwashing machine",
                "screen, silver screen, projection screen",
                "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",
                "traffic light, traffic signal, stoplight", "tray",
                "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "
                "dustbin, trash barrel, trash bin",
                "fan", "pier, wharf, wharfage, dock", "crt screen",
                "plate", "monitor, monitoring device", "bulletin board, notice board",
                "shower", "radiator", "glass, drinking glass", "clock", "flag")


    def _get_ade20k_pairs(self, folder, mode='train'):
        img_paths = []
        mask_paths = []
        scenes = []
        if mode == 'train':
            img_folder = os.path.join(folder, 'images/training')
            mask_folder = os.path.join(folder, 'annotations/training')
        else:
            img_folder = os.path.join(folder, 'images/validation')
            mask_folder = os.path.join(folder, 'annotations/validation')
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                    scenes.append(self.scene_classes[basename.split('_')[0]])
                else:
                    print('cannot find the mask:', maskpath)

        return img_paths, mask_paths, scenes


class IndoorContextV6(SegmentationDataset):
    def __init__(self, root='../datasets/', split='train', transform=None, **kwargs):
        super(IndoorContextV6, self).__init__(root, split, transform, **kwargs)
        assert os.path.exists(root), "The data set path does not exist !"
        self.mode = split
        self.images, self.fires = self._get_data_pairs(root, split)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in folders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        if self.mode == 'train':
            img = self._sync_transform_no_mask(img)
        elif self.mode == 'val':
            img = self._val_sync_transform_no_mask(img)
        else:
            assert self.mode == 'testval'
            img = self._img_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.fires[index]

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    @property
    def ufires(self):
        return ufire
    
    def _get_data_pairs(self, folder, split='train'):
        img_paths = [] # img paths
        fires = [] # scene categories 
        
        if split == 'train':
            img_folder = os.path.join(folder, 'train')
        else:
            img_folder = os.path.join(folder, 'test')

        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                if os.path.isfile(imgpath):
                    tmp = basename.split('_')
                    img_paths.append(imgpath)
                    fires.append(self.ufires[tmp[0]])
                else:
                    print('cannot find the image:', imgpath)

        return img_paths, fires


if __name__ == '__main__':
    train_dataset = IndoorContext()
