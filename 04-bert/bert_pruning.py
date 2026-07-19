"""
演示
    BERT模型的剪枝,全局非结构化剪枝
    对所有encoder层的注意力权重剪枝0.3，L1范数
非结构化剪枝：
    1.不能减小体积，主要用于测试阶段，把部分权重设为精确0，找到最优的剪枝比例
需要掌握API:
    prune.global_unstructured()
"""
# 导包
import os
from torch.nn.utils import prune    # pytorch的剪枝模块，提供各种剪枝方法
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from bert_train import *
import matplotlib.pyplot as plt # 绘图
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False            # 解决负号显示问题

# 0.全局配置
# 创建配置对象
config = Config()

# 1.定义函数，计算BERT模型的 Encoder层 的query权重的稀疏度，零值占比
def calculate_sparsity(model):
    """
    计算BERT模型的 Encoder层 的query权重的稀疏度，公式：0值参数数量/总参数量
    :param model: 模型对象，这里BERT下游分类微调模型的模型对象
    :return: 稀疏度，flot32,取值范围0~1
    """
    # 1.定义变量，记录 总参数数量，0值参数数量，BERT中的Encoder层数量
    total_params = 0
    zero_params = 0
    num_encoder_layers = len(model.bert.encoder.layer)  # 12
    # 2.遍历Encoder层，计算query权重的零值数量，总参数数量，稀疏度
    for i in range(num_encoder_layers):
        # 1.获取当前Encoder层的query权重
        query_weight = model.bert.encoder.layer[i].attention.self.query.weight
        # print(query_weight)
        # 2.统计query权重的零值数量，总参数数量
        total_params += query_weight.numel()
        zero_params += (query_weight == 0).sum().item()
    sparsity = zero_params / total_params
    return sparsity

# 2.定义函数，进行模型剪枝
def prune_model(model):
    """
    模型剪枝函数，把输入的模型对象，进行非结构化剪枝，并返回剪枝后的模型对象
    :param model: 剪枝前的模型对象
    :return: 剪枝后的模型对象
    """
    # 1.定义进行全局非结构化剪枝的参数，所有Encoder层 的query权重参数，获取列表
    num_encoder_layers = len(model.bert.encoder.layer)
    prune_params = [
        (model.bert.encoder.layer[i].attention.self.query, "weight") for i in range(num_encoder_layers)
    ]
    # 2.进行全局非结构化剪枝，prune.global_unstructured()
    prune.global_unstructured(
        prune_params,   # 剪枝参数列表
        pruning_method=prune.L1Unstructured,    # 剪枝方式: L1范数
        # pruning_method=prune.RandomUnstructured,    # 剪枝方式: 随机剪枝
        amount=0.6, # 剪枝比例
    )
    # 3.固化剪枝，固定剪枝后的权重参数，删除剪枝的临时结果
    for module, params in prune_params:
        prune.remove(module, params)
    # 4.保存剪枝后的模型参数
    torch.save(model.state_dict(), config.bert_pruning_path)
    return model

# 测试
if __name__ == '__main__':
    # 1.加载模型对象，BERT下游分类微调模型
    model = MyBertModel().to(config.device)
    model.load_state_dict(
        torch.load(config.bert_model_path,weights_only=True,map_location=config.device)
    )
    # 2.测试剪枝前模型
    train_dataloader, valid_dataloader, test_dataloader = build_dataloader()
    # test_loss, test_acc, test_f1 = evaluate(test_dataloader, model, loss_fn=nn.CrossEntropyLoss())
    # print(f"剪枝前模型F1分数：{test_f1}")
    # 3.打印剪枝前模型的稀疏度
    result = calculate_sparsity(model)
    print(f"剪枝前模型稀疏度：{result}")
    # 4.进行模型剪枝
    model = prune_model(model)
    # 5.测试剪枝后的模型
    test_loss, test_acc, test_f1 = evaluate(test_dataloader, model, loss_fn=nn.CrossEntropyLoss())
    print(f"剪枝后模型F1分数：{test_f1}")
    # 6.打印剪枝后模型的稀疏度
    result = calculate_sparsity(model)
    print(f"剪枝后模型稀疏度：{result}")
    ...