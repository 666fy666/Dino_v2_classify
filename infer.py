import logging
from safetensors import safe_open
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Config, Dinov2ForImageClassification

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
        class CustomDinov2(Dinov2ForImageClassification):
            def __init__(self, config):
                super().__init__(config)
                hidden_size = config.hidden_size

                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size * 2),
                    torch.nn.BatchNorm1d(hidden_size * 2),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(hidden_size * 2, hidden_size),
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(hidden_size, config.num_labels)
                )

                for module in self.classifier:
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.xavier_normal_(module.weight)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)

                self.dinov2 = self.dinov2

            def forward(self, pixel_values):
                outputs = self.dinov2(pixel_values)
                features = outputs.last_hidden_state[:, 0]
                return self.classifier(features)

        return CustomDinov2(config)

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
        try:
            logger.info(f"Predicting: {image_path}")
            image = Image.open(image_path).convert("RGB")

            inputs = self._preprocess_image(image)

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

            logger.info("Prediction succeeded")
            return results

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        classifier = FineGrainedClassifier(
            model_path="/media/search3/E/Intern/fengyu/Dino_v2_classify/results/final_model")

        result = classifier.predict("search_data/1-M0802010000-滚珠轴承/110300107560.jpeg")

        print("\nPrediction results:")
        for i, item in enumerate(result, 1):
            print(f"{i}. {item['label']} (Confidence: {float(item['confidence']) * 100:.2f}%)")

    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")