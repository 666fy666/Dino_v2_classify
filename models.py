import torch
from transformers import Dinov2Config, Dinov2ForImageClassification


class LabelSmoothCrossEntropy(torch.nn.Module):
    """标签平滑损失函数"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1.0 - self.smoothing
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return (confidence * nll_loss + self.smoothing * smooth_loss).mean()


class Dinov2FineGrained(Dinov2ForImageClassification):
    """自定义Dinov2分类模型"""

    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.hidden_size

        # 替换原始分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.BatchNorm1d(hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, config.num_labels)
        )
        self._init_weights(self.classifier)
        self.loss_fct = LabelSmoothCrossEntropy(smoothing=0.1)

    def _init_weights(self, module):
        """自定义权重初始化"""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, pixel_values, labels=None):
        outputs = self.dinov2(pixel_values, output_hidden_states=True)
        features = outputs.hidden_states[-1][:, 0]  # CLS token
        logits = self.classifier(features)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return (loss, logits)
        return logits

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """改进的预训练模型加载方法"""
        # 加载原始配置
        config = Dinov2Config.from_pretrained(pretrained_model_name_or_path)

        # 更新分类相关参数
        if "num_labels" in kwargs:
            config.num_labels = kwargs.pop("num_labels")
        if "id2label" in kwargs:
            config.id2label = kwargs.pop("id2label")
        if "label2id" in kwargs:
            config.label2id = kwargs.pop("label2id")

        # 创建模型实例
        model = cls(config)

        # 加载预训练权重
        base_model = Dinov2ForImageClassification.from_pretrained(pretrained_model_name_or_path)
        model.dinov2 = base_model.dinov2

        return model

    def freeze_backbone(self):
        """冻结骨干网络"""
        for param in self.dinov2.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_layers=6):
        """解冻部分骨干网络"""
        total_layers = len(self.dinov2.encoder.layer)
        for i, layer in enumerate(self.dinov2.encoder.layer):
            if i >= total_layers - unfreeze_layers:
                layer.requires_grad_(True)
            elif i >= total_layers - unfreeze_layers - 4:
                for param in layer.mlp.parameters():
                    param.requires_grad = True
