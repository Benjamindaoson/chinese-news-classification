"""
演示
    bert模型的 动态量化DQ
需要掌握的API:
    torch.quantization.quantize_dynamic()
    只能在CPU上使用，不能在GPU上使用
注意：
    1.直接在config中修改device为torch.device("cpu")
    2.或者新建一个虚拟环境nlp_cpu, conda create -n nlp_cpu python=3.10
    安装cpu版本的pytorch, pip3 install torch torchvision
    其余包自己搞定
"""

# 导包
import torch
from torch import nn
from config import Config
from bert_train import MyBertModel, evaluate, build_dataloader

# 0.查看量化引擎
# 常量量化引擎: onednn(AMD/intel), fbgemm(intel)
print(f"当前量化引擎: {torch.backends.quantized.supported_engines}")
# 1.创建配置对象
config = Config()
print(f"当前设备: {config.device}")
# 2.创建数据加载器
_, _, test_dataloader = build_dataloader()
# 3.加载模型，权重参数精度float32
model = MyBertModel().to(config.device)
# 加载模型参数
model.load_state_dict(
    torch.load(
        config.bert_model_path, # 本地保存的模型参数文件
        weights_only=True, # 只加载模型权重参数，不加载其他对象，更加安全
        map_location=config.device, # 将模型加载到指定设备上，自动适配设备，防止训练或推理阶段的设备不一致
    )
)
# print(model)
# 4.评估原始模型
test_loss, test_acc, test_f1 = evaluate(
    val_dataloader=test_dataloader,
    model=model,
    loss_fn=nn.CrossEntropyLoss()

)
print(f"原始模型测试结果: loss: {test_loss:.4f}, acc: {test_acc:.4f}, f1: {test_f1:.4f}")
# 5.进行动态量化DQ, torch.quantization.quantize_dynamic()
# 设为评估模式
model.eval()
model_quant = torch.quantization.quantize_dynamic(
    model,  # 需要量化的模型对象
    {nn.Linear}, # 需要进行动态量化的层
    dtype=torch.qint8,  # 量化精度，常用int8
)
# 6.评估量化后模型
test_loss, test_acc, test_f1 = evaluate(
    val_dataloader=test_dataloader,
    model=model_quant,
    loss_fn=nn.CrossEntropyLoss()
)
print(f"量化后模型测试结果: loss: {test_loss:.4f}, acc: {test_acc:.4f}, f1: {test_f1:.4f}")
# 7.保存量化后模型
torch.save(model_quant.state_dict(), config.bert_quantization_path)
print(f"保存量化模型到: {config.bert_quantization_path}")

