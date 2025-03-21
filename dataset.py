import os
from torchvision import transforms
from datasets import load_dataset, Image
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, dataset, processor, augmentation_config=None, is_train=True):
        self.dataset = dataset
        self.processor = processor
        self.is_train = is_train
        self._length = len(dataset)  # 显式记录长度

        if self.is_train:
            if not augmentation_config:
                raise ValueError("训练模式需要数据增强配置")

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    224,
                    scale=tuple(augmentation_config["train_crop_scale"])
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(*augmentation_config["color_jitter"]),
                transforms.RandomRotation(augmentation_config["random_rotate"]),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])

    def __len__(self):
        return self._length  # 返回预先计算的长度

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]

        # 应用预处理
        image = self.transform(image)

        processed = self.processor(
            images=image,
            return_tensors="pt",
            do_rescale=False,
            do_normalize=True
        )
        return {
            "pixel_values": processed.pixel_values.squeeze(),
            "labels": torch.tensor(label)
        }


def safe_split_dataset(full_dataset, val_ratio=0.1, min_samples=2):
    label_indices = {}
    for idx, item in enumerate(full_dataset):
        label = str(item["label"])
        label_indices.setdefault(label, []).append(idx)

    train_indices, val_indices = [], []
    error_count = 0

    for label, indices in label_indices.items():
        if len(indices) < min_samples:
            train_indices.extend(indices)
            continue

        try:
            _train, _val = train_test_split(
                indices,
                test_size=max(val_ratio, 1 / len(indices)),
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