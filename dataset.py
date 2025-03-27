import logging
import random

import numpy as np
import torch
from PIL import ImageFilter, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class GaussianBlur:
    """随机高斯模糊"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize:
    """随机曝光"""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)


class RandomErasing:
    """随机擦除"""

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img_tensor):
        if random.random() > self.p:
            return img_tensor

        c, h, w = img_tensor.shape
        area = h * w

        for _ in range(10):
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            erase_h = int(round(np.sqrt(erase_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(erase_area / aspect_ratio)))

            if erase_h < h and erase_w < w:
                i = random.randint(0, h - erase_h)
                j = random.randint(0, w - erase_w)

                img_tensor[:, i:i + erase_h, j:j + erase_w] = torch.rand(c, erase_h, erase_w)
                break

        return img_tensor


class CustomDataset(Dataset):
    def __init__(self, dataset, processor, augmentation_config=None, is_train=True):
        self.dataset = dataset
        self.processor = processor
        self.is_train = is_train
        self._length = len(dataset)  # 显式记录长度

        # 计算类别数量，用于自适应调整增强强度
        self.num_classes = len(set(item["label"] for item in dataset))
        logger.info(f"数据集包含 {self.num_classes} 个类别")

        # 根据类别数量自适应调整增强强度
        self.augmentation_strength = 1.0
        if is_train and augmentation_config.get("adaptive_strength", False):
            self.augmentation_strength = min(1.0, max(0.5, 1000 / self.num_classes))
            logger.info(f"使用自适应增强强度: {self.augmentation_strength:.2f}")

        # 获取DINOv2的归一化参数
        self.mean = self.processor.image_mean
        self.std = self.processor.image_std
        logger.info(f"使用DINOv2归一化参数 - 均值: {self.mean}, 标准差: {self.std}")

        if self.is_train:
            if not augmentation_config:
                raise ValueError("训练模式需要数据增强配置")

            # 基础变换
            base_transforms = [
                transforms.RandomResizedCrop(
                    224,
                    scale=tuple(augmentation_config["train_crop_scale"]),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]

            # 颜色变换
            color_strength = augmentation_config["color_jitter"]
            if augmentation_config.get("adaptive_strength", False):
                adjusted_color = [c * self.augmentation_strength for c in color_strength]
            else:
                adjusted_color = color_strength

            color_transforms = [
                transforms.ColorJitter(
                    brightness=adjusted_color[0],
                    contrast=adjusted_color[1],
                    saturation=adjusted_color[2],
                    hue=augmentation_config.get("color_jitter_hue", 0.1) * self.augmentation_strength
                ),
            ]

            # 几何变换
            rotate_angle = augmentation_config["random_rotate"]
            if augmentation_config.get("adaptive_strength", False):
                rotate_angle = rotate_angle * self.augmentation_strength

            geometric_transforms = [
                transforms.RandomRotation(rotate_angle),
            ]

            # 添加随机透视变换
            if "random_perspective" in augmentation_config:
                perspective_prob = augmentation_config["random_perspective"]
                if augmentation_config.get("adaptive_strength", False):
                    perspective_prob = perspective_prob * self.augmentation_strength
                geometric_transforms.append(transforms.RandomPerspective(p=perspective_prob))

            # 添加随机仿射变换
            if "random_affine" in augmentation_config:
                affine_config = augmentation_config["random_affine"]
                translate = affine_config.get("translate", [0, 0])
                scale = affine_config.get("scale", [1.0, 1.0])
                shear = affine_config.get("shear", 0)

                if augmentation_config.get("adaptive_strength", False):
                    translate = [t * self.augmentation_strength for t in translate]
                    shear = shear * self.augmentation_strength

                geometric_transforms.append(
                    transforms.RandomAffine(
                        degrees=0,
                        translate=tuple(translate),
                        scale=tuple(scale),
                        shear=shear
                    )
                )

            # 高级变换
            advanced_transforms = []

            # 添加高斯模糊
            if augmentation_config.get("gaussian_blur", {}).get("enabled", False):
                blur_sigma = augmentation_config.get("gaussian_blur", {}).get("sigma", [0.1, 2.0])
                if augmentation_config.get("adaptive_strength", False):
                    blur_sigma[1] = blur_sigma[1] * self.augmentation_strength
                advanced_transforms.append(GaussianBlur(sigma=blur_sigma))

            # 添加随机灰度化
            if "grayscale" in augmentation_config:
                grayscale_prob = augmentation_config["grayscale"]
                if augmentation_config.get("adaptive_strength", False):
                    grayscale_prob = grayscale_prob * self.augmentation_strength
                advanced_transforms.append(transforms.RandomGrayscale(p=grayscale_prob))

            # 添加随机曝光
            if augmentation_config.get("solarize", {}).get("enabled", False):
                solarize_threshold = augmentation_config.get("solarize", {}).get("threshold", 128)
                if augmentation_config.get("adaptive_strength", False):
                    # 调整阈值，使其随着增强强度变化
                    solarize_threshold = int(256 * (1 - 0.5 * self.augmentation_strength))
                advanced_transforms.append(Solarize(threshold=solarize_threshold))

            # 组合所有变换
            transform_list = [
                *base_transforms,
                *color_transforms,
                *geometric_transforms,
                *advanced_transforms,
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),  # 使用DINOv2的归一化参数
            ]

            # 组合所有变换
            self.transform = transforms.Compose(transform_list)

            # 随机擦除作为单独的步骤，因为它需要在ToTensor和Normalize之后应用
            self.random_erasing = None
            if "random_erasing" in augmentation_config:
                erasing_prob = augmentation_config["random_erasing"]
                if augmentation_config.get("adaptive_strength", False):
                    erasing_prob = erasing_prob * self.augmentation_strength
                self.random_erasing = RandomErasing(p=erasing_prob)

        else:
            # 验证集使用标准预处理
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),  # 使用DINOv2的归一化参数
            ])
            self.random_erasing = None

    def __len__(self):
        return self._length  # 返回预先计算的长度

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]

        # 应用预处理（包括归一化）
        image = self.transform(image)

        # 应用随机擦除（如果启用）
        if self.random_erasing is not None:
            image = self.random_erasing(image)

        # 不需要再次归一化，因为已经在transform中完成
        return {
            "pixel_values": image,  # 直接使用已经归一化的图像张量
            "labels": torch.tensor(label)
        }


def safe_split_dataset(full_dataset, val_ratio=0.1, min_samples=2):
    label_indices = {}
    for idx, item in enumerate(full_dataset):
        label = str(item["label"])
        label_indices.setdefault(label, []).append(idx)

    train_indices, val_indices = [], []
    error_count = 0

    # 计算类别分布情况
    class_distribution = {label: len(indices) for label, indices in label_indices.items()}
    num_classes = len(class_distribution)
    min_class_samples = min(class_distribution.values())
    max_class_samples = max(class_distribution.values())

    logger.info(f"类别总数: {num_classes}, 最小类别样本数: {min_class_samples}, 最大类别样本数: {max_class_samples}")

    for label, indices in label_indices.items():
        if len(indices) < min_samples:
            train_indices.extend(indices)
            continue

        try:
            # 对于样本数较少的类别，确保至少有一个样本进入验证集
            actual_val_ratio = max(val_ratio, 1 / len(indices))
            _train, _val = train_test_split(
                indices,
                test_size=actual_val_ratio,
                random_state=42
            )
            train_indices.extend(_train)
            val_indices.extend(_val)
        except:
            error_count += 1
            train_indices.extend(indices[:-1])
            val_indices.append(indices[-1])

    logger.info(f"总类别: {len(label_indices)}, 异常类别: {error_count}")
    logger.info(f"训练样本: {len(train_indices)}, 验证样本: {len(val_indices)}")

    return (
        full_dataset.select(train_indices),
        full_dataset.select(val_indices)
    )
