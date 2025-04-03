import os
import cv2
import json
import torch
import pandas as pd
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Tuple
from supervision.draw.color import ColorPalette
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def single_mask_to_rle(mask: np.ndarray) -> dict:
    """将单个掩码转换为RLE格式"""
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_single_image(
        image_path: str,
        processor: AutoProcessor,
        grounding_model: AutoModelForZeroShotObjectDetection,
        sam2_predictor: SAM2ImagePredictor,
        text_prompt: str,
        output_dir: Path,
        dump_json: bool,
        visualize: bool = False,
) -> Optional[Image.Image]:
    """处理单张图片并返回处理后的 PIL Image 对象"""
    # 读取图片
    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)
    original_size = image.size

    # 设置SAM2的图像嵌入
    sam2_predictor.set_image(image_rgb)

    # Grounding DINO处理
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(grounding_model.device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # 后处理检测结果
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[original_size[::-1]]
    )

    # 处理无检测结果的情况
    if not results or len(results[0]["boxes"]) == 0:
        print(f"返回原图，未检测到对象: {image_path}")
        return image

    # 提取检测信息
    input_boxes = results[0]["boxes"].cpu().numpy()
    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]

    # SAM2预测
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # 修复维度问题：将掩码从 (N, 1, H, W) 转换为 (N, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # 创建检测对象
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=np.arange(len(class_names))
    )

    # 选择居中掩码
    image_center = np.array(original_size) / 2
    min_distance = float('inf')
    selected_mask = None

    for idx, (mask, box) in enumerate(zip(detections.mask, detections.xyxy)):
        # 计算边界框中心
        box_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        distance = np.linalg.norm(box_center - image_center)

        if distance < min_distance:
            min_distance = distance
            selected_mask = mask

    # 生成掩码区域图像
    mask_image = Image.fromarray((selected_mask * 255).astype(np.uint8))
    processed_image = Image.composite(image, Image.new('RGB', original_size, (255, 255, 255)), mask_image)

    # 获取掩码边界框
    y, x = np.where(selected_mask)
    if len(x) == 0 or len(y) == 0:
        return None

    crop_box = (x.min(), y.min(), x.max(), y.max())
    cropped = processed_image.crop(crop_box)

    # 计算扩展尺寸
    w, h = cropped.size
    max_dim = max(w, h)
    expand_size = int(max_dim * 1.2)
    square_size = (expand_size, expand_size)

    # 创建最终图像
    final_image = Image.new('RGB', square_size, (255, 255, 255))
    offset = ((expand_size - w) // 2, (expand_size - h) // 2)
    final_image.paste(cropped, offset)

    # 保存处理结果
    if visualize:
        output_filename = f"{Path(image_path).stem}.jpg"
        output_path = output_dir / output_filename
        final_image.save(output_path)
        print(f"保存处理结果到: {output_path}")

    # 生成JSON输出
    if dump_json:
        mask_rles = [single_mask_to_rle(mask) for mask in masks]
        result_data = {
            "image_path": str(image_path),
            "detections": [
                {
                    "class": cls,
                    "confidence": float(conf),
                    "bbox": box.tolist(),
                    "segmentation": rle
                }
                for cls, conf, box, rle in zip(class_names, confidences, input_boxes, mask_rles)
            ],
            "dimensions": {"width": original_size[0], "height": original_size[1]}
        }
        json_path = output_dir / f"{Path(image_path).stem}_results.json"
        with open(json_path, "w") as f:
            json.dump(result_data, f, indent=2)

    return final_image


def run_grounded_sam2(
        grounding_model: str = "IDEA-Research/grounding-dino-tiny",
        text_prompt: str = "Industrial parts. Industrial equipment. Mechanical parts.",
        img_path: str = "notebooks/images/truck.jpg",
        sam2_checkpoint: str = "./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        output_dir: str = "outputs",
        no_dump_json: bool = True,
        visualize: bool = False,
        force_cpu: bool = False
) -> List[Optional[Image.Image]]:
    """
    增强版处理函数，返回处理后的 PIL Image 对象列表
    """
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        sam_model = build_sam2(sam2_model_config, sam2_checkpoint, device=device)
        sam_predictor = SAM2ImagePredictor(sam_model)

        processor = AutoProcessor.from_pretrained(grounding_model)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(device)

        input_path = Path(img_path)
        if input_path.is_file():
            image_paths = [input_path]
        elif input_path.is_dir():
            image_paths = sorted([p for p in input_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        else:
            raise ValueError(f"无效的输入路径: {img_path}")

        results = []
        for img_path in image_paths:
            result = process_single_image(
                image_path=str(img_path),
                processor=processor,
                grounding_model=dino_model,
                sam2_predictor=sam_predictor,
                text_prompt=text_prompt,
                output_dir=output_path,
                dump_json=not no_dump_json,
                visualize=visualize
            )
            results.append(result)
            print(f"处理完成: {img_path}")

    return results


if __name__ == "__main__":
    # 示例调用 - 处理并保存结果
    processed_images = run_grounded_sam2(
        img_path="./test",
        text_prompt="Mechanical parts.",
        output_dir="visual",
        visualize=False
    )