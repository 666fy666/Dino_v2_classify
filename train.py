import yaml
import logging
import warnings
from pathlib import Path
import torch
import numpy as np
from datasets import load_dataset, Image
from sklearn.model_selection import train_test_split
from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    Dinov2Config,
    EvalPrediction
)
import evaluate
from models import Dinov2FineGrained
from dataset import CustomDataset, safe_split_dataset

# 配置加载
with open("config.yaml",encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 初始化路径
train_conf = config["training"]
output_dir = Path(train_conf["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(output_dir / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 过滤警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

# 定义评估指标
metric = evaluate.load("accuracy")

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)

def train_pipeline():
    try:
        logger.info("=== 初始化训练流程 ===")

        # 加载数据集
        logger.info("加载数据集中...")
        full_dataset = load_dataset(
            "imagefolder",
            data_dir=train_conf["dataset_path"],
            split="train"
        ).cast_column("image", Image())

        # 拆分数据集
        logger.info("拆分训练集/验证集...")
        train_ds, val_ds = safe_split_dataset(
            full_dataset,
            val_ratio=train_conf["val_ratio"],
            min_samples=train_conf["min_samples"]
        )

        # 初始化处理器
        logger.info("初始化图像处理器...")
        processor = AutoImageProcessor.from_pretrained(
            train_conf["model_name"],
            do_rescale=True,
            do_normalize=True
        )

        # 准备数据集
        logger.info("准备训练数据...")
        train_data = CustomDataset(
            train_ds,
            processor,
            augmentation_config=train_conf["augmentation"],
            is_train=True
        )

        logger.info("准备验证数据...")
        val_data = CustomDataset(
            val_ds,
            processor,
            is_train=False
        )

        # 初始化模型
        logger.info("创建模型...")
        model_config = Dinov2Config.from_pretrained(train_conf["model_name"])
        model_config.num_labels = len(train_ds.features["label"].names)
        model_config.id2label = {i: str(n) for i, n in enumerate(train_ds.features["label"].names)}
        model_config.label2id = {str(n): i for i, n in enumerate(train_ds.features["label"].names)}

        model = Dinov2FineGrained.from_pretrained(
            train_conf["model_name"],
            num_labels=model_config.num_labels,
            id2label=model_config.id2label,
            label2id=model_config.label2id
        )
        model.freeze_backbone()

        # 训练参数配置（完整版）
        training_args = TrainingArguments(
            # === 核心路径 ===
            output_dir=str(output_dir),  # 模型和日志保存路径
            logging_dir=str(output_dir / "logs"),  # TensorBoard日志目录

            # === 训练控制 ===
            num_train_epochs=train_conf["head_epochs"],  # 训练总轮数
            per_device_train_batch_size=train_conf["batch_size"],  # 单设备训练批次
            per_device_eval_batch_size=train_conf["batch_size"] * 2,  # 单设备评估批次

            # === 优化器与调度 ===
            learning_rate=train_conf["optimizer"]["learning_rate"],  # 初始学习率
            weight_decay=train_conf["optimizer"]["weight_decay"],  # 权重衰减系数
            warmup_ratio=train_conf["optimizer"]["warmup_ratio"],  # 学习率预热比例
            optim=train_conf["optimizer"].get("name", "adamw_torch"),  # 优化器类型
            adam_beta1=train_conf["optimizer"].get("adam_beta1", 0.9),  # Adam参数β1
            adam_beta2=train_conf["optimizer"].get("adam_beta2", 0.999),  # Adam参数β2
            lr_scheduler_type=train_conf["optimizer"].get("scheduler", "linear"),  # 调度策略

            # === 评估策略 ===
            evaluation_strategy=train_conf.get("evaluation_strategy", "epoch"),  # epoch/steps/no
            eval_steps=train_conf.get("eval_steps", None),  # 按步评估时生效
            eval_accumulation_steps=train_conf.get("eval_accumulation_steps", None),  # 评估梯度累积

            # === 保存策略 ===
            save_strategy=train_conf.get("save_strategy", "epoch"),  # epoch/steps/no
            save_steps=train_conf.get("save_steps", None),  # 按步保存时生效
            save_total_limit=train_conf.get("save_total_limit", 3),  # 最大保存检查点数

            # === 性能优化 ===
            fp16=train_conf.get("fp16", torch.cuda.is_available()),  # FP16混合精度
            bf16=train_conf.get("bf16", False),  # BF16混合精度
            gradient_accumulation_steps=train_conf.get("gradient_accumulation_steps", 1),  # 梯度累积
            gradient_checkpointing=train_conf.get("gradient_checkpointing", False),  # 梯度检查点
            torch_compile=train_conf.get("torch_compile", False),  # PyTorch 2.0编译优化
            dataloader_num_workers=train_conf.get("dataloader_num_workers", 8),  # 数据加载线程数
            dataloader_pin_memory=train_conf.get("dataloader_pin_memory", True),  # 内存固定

            # === 日志与调试 ===
            logging_steps=train_conf.get("logging_steps", 500),  # 日志记录间隔
            report_to=train_conf.get("report_to", "tensorboard"),  # 日志后端
            log_level=train_conf.get("log_level", "passive"),  # 日志级别
            disable_tqdm=train_conf.get("disable_tqdm", False),  # 禁用进度条

            # === 模型选择 ===
            load_best_model_at_end=train_conf.get("load_best_model_at_end", True),  # 加载最佳模型
            metric_for_best_model=train_conf.get("metric_for_best_model", "eval_accuracy"),  # 指标名
            greater_is_better=train_conf.get("greater_is_better", True),  # 指标方向

            # === 高级控制 ===
            max_grad_norm=train_conf["optimizer"].get("max_grad_norm", 1.0),  # 梯度裁剪
            remove_unused_columns=train_conf.get("remove_unused_columns", False),  # 保留未用列
            deepspeed=train_conf.get("deepspeed_config", None),  # DeepSpeed配置文件路径
            local_rank=train_conf.get("local_rank", -1),  # 分布式训练参数（自动检测）
        )

        # 第一阶段训练
        logger.info("\n=== 第一阶段：训练分类头 ===")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics  # 新增参数
        )
        trainer.train()

        # 第二阶段微调
        logger.info("\n=== 第二阶段：微调骨干网络 ===")
        model.unfreeze_backbone()
        training_args.learning_rate = train_conf["optimizer"]["fine_tune_lr"]
        training_args.num_train_epochs = train_conf["fine_tune_epochs"]
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics  # 新增参数
        )
        trainer.train()

        # 保存最终模型
        final_dir = output_dir / "final_model"
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        logger.info(f"模型已保存至：{final_dir}")

    except Exception as e:
        logger.error(f"训练流程异常终止: {str(e)}")
        raise

if __name__ == "__main__":
    train_pipeline()
