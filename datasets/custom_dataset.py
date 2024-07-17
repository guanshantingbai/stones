import os
from collections import defaultdict

import torch.utils.data as data
from PIL import Image

from datasets import transforms


def default_loader(path):
    return Image.open(path).convert('RGB')


def generate_transform_dict(origin_width=256, width=227, ratio=0.16, net="BN_Inception"):

    # for BN_Inception
    if net == "bn_inception":
        normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                         std=[1.0/255, 1.0/255, 1.0/255])
    else:
        print('ImageNet normalize')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform_dict = {}

    transform_dict['rand-crop'] = \
        transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize((origin_width)),
            transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['center-crop'] = \
        transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize((origin_width)),
            transforms.CenterCrop(width),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['resize'] = \
        transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize((width)),
            transforms.ToTensor(),
            normalize,
        ])
    return transform_dict


class custom(data.dataset.Dataset):
    def __init__(self, root, label_txt, transform=None, loader=default_loader) -> None:
        images = []
        labels = []
        with open(label_txt) as f:
            images_anon = f.readlines()
            for img_anon in images_anon:
                [img, label] = img_anon.split(' ')
                images.append(img)
                labels.append(int(label))
        classes = list(set(labels))

        index = defaultdict(list)
        for i, label in enumerate(labels):
            index[label].append(i)

        self.root = root
        self.images = images
        self.labels = labels
        self.loader = loader
        self.transform = transform
        self.classes = classes
        self.Index = index

    def __getitem__(self, index) -> tuple:
        image, label = self.images[index], self.labels[index]
        image = os.path.join(self.root, image)
        img = self.loader(image)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)


def main():
    root = "/media/data3/gdliu_data/CUB200/"
    label_txt = os.path.join(root, "origin_train.txt")
    transform_dict = generate_transform_dict(width=224)
    test_data = custom(root, label_txt, transform_dict['rand-crop'])
    print(test_data.images[0], test_data.labels[0])
    print(test_data[0])


if __name__ == "__main__":
    main
