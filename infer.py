import logging
import os
import time
from safetensors import safe_open
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Config
import pandas as pd
from collections import defaultdict

from grounded_sam2_hf_model import run_grounded_sam2
# 导入自定义模型
from models import Dinov2FineGrained

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FineGrainedClassifier:
    def __init__(self, model_path="./results/final_model"):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # 加载处理器和配置
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            config = Dinov2Config.from_pretrained(model_path)

            # 构建模型
            self.model = self._build_model(config).to(self.device)

            # 加载safetensors权重
            with safe_open(f"{model_path}/model.safetensors", framework="pt", device=self.device.type) as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}
                self.model.load_state_dict(state_dict)

            self.model.eval()

            # 标签映射
            self.id2label = {str(k): v for k, v in config.id2label.items()}

            # TTA参数
            self.tta_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
            self.scales = [0.9, 1.0, 1.1]

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _build_model(self, config):
        return Dinov2FineGrained(config)

    def _preprocess_image(self, image):
        processed = []
        base_size = image.size

        for scale in self.scales:
            target_size = (int(base_size[0] * scale), int(base_size[1] * scale))
            scaled_img = transforms.functional.resize(
                image,
                target_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
            transformed = self.tta_transforms(scaled_img)
            tensor_img = transforms.ToTensor()(transformed)
            processed.append(
                self.processor(
                    tensor_img,
                    return_tensors="pt",
                    do_rescale=False
                ).pixel_values.to(self.device)
            )
        return torch.cat(processed)

    def predict(self, image_path, top_k=5):
        """核心预测方法"""
        try:
            processed_images = run_grounded_sam2(
                img_path=image_path,
                text_prompt="Mechanical parts.",
                output_dir="visual",
                visualize=False
            )
            # image = Image.open(image_path).convert("RGB")
            inputs = self._preprocess_image(processed_images[0])

            with torch.no_grad():
                logits = self.model(inputs)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            avg_probs = torch.mean(probs, dim=0)

            top_probs, top_indices = torch.topk(avg_probs, top_k)

            results = []
            for prob, idx in zip(top_probs, top_indices):
                label_id = str(idx.item())
                results.append({
                    "label": self.id2label.get(label_id, f"UNKNOWN_{label_id}"),
                    "confidence": f"{prob.item():.4f}"
                })
            return results

        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {str(e)}")
            raise

    def predict_single(self, image_path, top_k=5):
        """单张图片预测"""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"\n{'=' * 30} Processing single image {'=' * 30}")
        start_time = time.perf_counter()
        result = self.predict(image_path, top_k)
        end_time = time.perf_counter()
        duration = end_time - start_time

        # 获取真实标签
        true_label = os.path.basename(os.path.dirname(image_path))
        predicted_label = result[0]['label']
        accuracy = 1.0 if predicted_label == true_label else 0.0

        self._log_results(image_path, result)
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        logger.info(f"Inference time: {duration:.6f} seconds")
        return result

    def predict_folder(self, folder_path, top_k=5):
        """单层文件夹预测"""
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Invalid folder: {folder_path}")

        logger.info(f"\n{'=' * 30} Processing folder: {folder_path} {'=' * 30}")
        processed_count = 0
        correct_predictions = 0
        total_time = 0.0

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if self._is_image_file(file_path):
                try:
                    start_time = time.perf_counter()
                    result = self.predict(file_path, top_k)
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    total_time += duration
                    processed_count += 1

                    # 统计准确率
                    true_label = os.path.basename(folder_path)
                    predicted_label = result[0]['label']
                    if predicted_label == true_label:
                        correct_predictions += 1

                    self._log_results(file_path, result)
                except Exception as e:
                    logger.error(f"Skipped {filename}: {str(e)}")
                    continue

        logger.info(f"Processed {processed_count} images in folder")
        if processed_count > 0:
            accuracy = correct_predictions / processed_count
            avg_time = total_time / processed_count
            logger.info(f"Average accuracy: {accuracy * 100:.2f}%")
            logger.info(f"Average inference time per image: {avg_time:.6f} seconds")
        else:
            logger.info("No images processed, accuracy and time statistics unavailable.")
        return processed_count

    def predict_test(self, folder_path, top_k=5):
        """单层文件夹预测（直接使用文件名作为series_code）"""
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Invalid folder: {folder_path}")

        # 硬编码路径
        PKL_PATH = "./data_seriesol.pkl"
        EXCEL_PATH = "category_statistics.xlsx"

        # 读取数据
        try:
            pkl_df = pd.read_pickle(PKL_PATH)
            pkl_df['series_code'] = pkl_df['series_code'].astype(str)
        except Exception as e:
            raise RuntimeError(f"加载PKL文件失败: {str(e)}")

        # 读取统计文件
        try:
            df_excel = pd.read_excel(EXCEL_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(f"统计文件 {EXCEL_PATH} 不存在")

        # 初始化统计字典
        category_counts = defaultdict(int)

        logger.info(f"\n{'=' * 30} 处理文件夹: {folder_path} {'=' * 30}")
        processed_count = 0
        correct_predictions = 0
        valid_labels_count = 0
        total_time = 0.0

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if self._is_image_file(file_path):
                try:
                    # 执行预测
                    start_time = time.perf_counter()
                    result = self.predict(file_path, top_k)
                    duration = time.perf_counter() - start_time

                    total_time += duration
                    processed_count += 1

                    # 获取真实标签
                    true_label = None
                    series_code = os.path.splitext(filename)[0]

                    # 从PKL查找
                    pkl_match = pkl_df[pkl_df['series_code'] == series_code]
                    if not pkl_match.empty:
                        pkl_row = pkl_match.iloc[0]
                        category_code = pkl_row['category_code']
                        true_label = f"{category_code}"
                        valid_labels_count += 1

                        # 统计匹配次数
                        count = 0
                        for i in range(top_k):
                            pred_label = result[i]['label']
                            if (true_label == pred_label) or (pred_label in true_label) or (true_label in pred_label):
                                count += 1

                        # 更新准确率统计
                        if count > 0:
                            correct_predictions += 1
                            logger.info(f"文件 {filename} 匹配成功，匹配次数: {count}")

                        # 累加到分类统计
                        category_counts[category_code] += count

                    else:
                        logger.warning(f"PKL中未找到series_code: {series_code}")

                    self._log_results(file_path, result)
                except Exception as e:
                    logger.error(f"跳过 {filename}: {str(e)}")
                    continue

        # 输出统计结果
        logger.info(f"\n处理完成，有效图片数: {processed_count}")
        if valid_labels_count > 0:
            logger.info(f"识别准确率: {correct_predictions / valid_labels_count * 100:.2f}%")
        if processed_count > 0:
            logger.info(f"平均推理时间: {total_time / processed_count:.4f}s/张")

        # 更新Excel文件
        for code, total in category_counts.items():
            mask = df_excel.iloc[:, 0] == code
            if mask.any():
                df_excel.loc[mask, df_excel.columns[2]] = total
            else:
                logger.warning(f"未找到category_code: {code}")

        try:
            df_excel.to_excel(EXCEL_PATH, index=False)
            logger.info(f"统计结果已保存至 {EXCEL_PATH}")
        except Exception as e:
            logger.error(f"保存统计文件失败: {str(e)}")

        return processed_count

    def predict_nested_folders(self, root_folder, top_k=5):
        """嵌套文件夹预测"""
        if not os.path.isdir(root_folder):
            raise NotADirectoryError(f"Invalid root folder: {root_folder}")

        logger.info(f"\n{'=' * 30} Processing nested folders: {root_folder} {'=' * 30}")
        processed_count = 0
        correct_predictions = 0
        total_time = 0.0

        for root, _, files in os.walk(root_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if self._is_image_file(file_path):
                    try:
                        start_time = time.perf_counter()
                        result = self.predict(file_path, top_k)
                        end_time = time.perf_counter()
                        duration = end_time - start_time
                        total_time += duration
                        processed_count += 1

                        # 统计准确率
                        true_label = os.path.basename(root)
                        predicted_label = result[0]['label']
                        if predicted_label == true_label:
                            correct_predictions += 1

                        self._log_results(file_path, result)
                    except Exception as e:
                        logger.error(f"Skipped {file_path}: {str(e)}")
                        continue

        logger.info(f"Processed {processed_count} images in nested folders")
        if processed_count > 0:
            accuracy = correct_predictions / processed_count
            avg_time = total_time / processed_count
            logger.info(f"Average accuracy: {accuracy * 100:.2f}%")
            logger.info(f"Average inference time per image: {avg_time:.6f} seconds")
        else:
            logger.info("No images processed, accuracy and time statistics unavailable.")
        return processed_count

    def _is_image_file(self, file_path):
        """验证图片文件格式"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        return os.path.isfile(file_path) and file_path.lower().endswith(valid_extensions)

    def _log_results(self, file_path, results):
        """统一日志记录"""
        logger.info(f"\nResults for: {file_path}")
        for i, item in enumerate(results, 1):
            confidence = float(item['confidence']) * 100
            logger.info(f"Top-{i}: {item['label']} ({confidence:.2f}%)")
        logger.info("-" * 80)


if __name__ == "__main__":
    try:
        # 初始化分类器
        classifier = FineGrainedClassifier(model_path="./results/final_model")

        # 示例用法 ---------------------------------------------------------
        # 单张图片推理
        # classifier.predict_single("search_data/1-M0802010000-滚珠轴承/110300107560.jpeg")

        # 单层文件夹推理
        # classifier.predict_folder("search_data/2-T1703020100-防尘口罩")
        classifier.predict_test("./test")

        # 嵌套文件夹推理（自动遍历子目录）
        # classifier.predict_nested_folders("search_data")

    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
