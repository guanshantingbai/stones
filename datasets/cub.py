"""
CUB-200-2011 dataset
"""
import os

from datasets.custom_dataset import custom, generate_transform_dict


class cub200:
    def __init__(self, root=None, width=224, origin_width=256, ratio=0.16, net="BN_Inception") -> None:
        if root is None:
            root = "/media/data3/gdliu_data/CUB200/"

        transform_dict = generate_transform_dict(
            origin_width=origin_width, width=width, ratio=ratio, net=net)

        train_txt = os.path.join(root, "new_train.txt")
        test_txt = os.path.join(root, "new_test.txt")
        origin_test_txt = os.path.join(root, "origin_test.txt")

        self.train = custom(root, train_txt, transform_dict["rand-crop"])
        self.test = custom(root, test_txt, transform_dict["center-crop"])
        self.origin_test = custom(
            root, origin_test_txt, transform_dict["center-crop"])


def main():
    cub_data = cub200()
    print(len(cub_data.train))
    print(len(cub_data.test))
    print(len(cub_data.origin_test))
    print(cub_data.train[0])


if __name__ == "__main__":
    main()