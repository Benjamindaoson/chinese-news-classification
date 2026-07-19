"""
演示
    基于预训练BERT模型进行下游文本分类微调的流程
NLP中神经网络训练的工作流/BERT微调工作流：
    1.加载分词器/词表
        bert_tokenizer自带词表
    2.准备数据集
        拆分为训练集、验证集、测试集
        构造样本的输入输出, MyDataset类
        DataSet,collate_fn,DataLoader关系:
            Dataset 提供单个样本，collate_fn 将多个样本打包成标准等长批次，Dataloader自动按批次取数据。
            Dataset: 孙悟空拿到一个桃子；collate_fn: 把孙悟空拿到的多个桃子打包成一个篮子，并且每个篮子有64个桃子；
            DataLoader: 负责自动按批次从collate_fn打包好的篮子里取桃子
        张量 -> 数据集对象 -> 数据加载器
    3.搭建神经网络
        预训练BERT，输出last_hidden_state(表示token语义),pooler_output(池化输出,整个句子的语义表示)
        pooler_output句子表示
        线性层nn.Linear
    4.模型训练 - 训练集 + 验证集
        1.前向传播
        2.计算损失
        3.梯度清零
        4.反向传播
        5.更新参数
    5.模型测试
        1.前向传播
        2.计算损失
        # 3.梯度清零
        # 4.反向传播
        # 5.更新参数

"""

# 导包
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tqdm import tqdm   # 可视化训练过程
import torch    # pytorch框架
import torch.nn as nn   # 神经网络模块
import torch.optim as optim # 优化器模块，提供各种优化器，比如SGD,Adam,AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig  # huggingface的transformers框架的具体模型方法
from config import Config   # 自定义的全局配置类
import time

# 计算 accuracy准确率 和 F1(精确率和召回率的调和平均)
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt # 绘图
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False            # 解决负号显示问题

# 0.全局配置
config = Config()
# 创建BERT分词器对象，BERT配置对象
BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)
BERT_CONFIG = BertConfig.from_pretrained(config.bert_path)
# print(BERT_TOKENIZER)
# print(BERT_CONFIG.hidden_size)

# 1.加载分词器 / 词表
# bert_tokenizer自带词表
# print(BERT_TOKENIZER.vocab)
# 2.准备数据集
# 拆分为训练集、验证集、测试集
# 2.1 定义函数，加载数据集，返回(文本，标签索引)的列表
def load_data(data_path):
    """
    加载数据文件，返回 (文本，标签索引)的列表
    :param data_path: 原始文件路径
    :return: 元组列表，格式[(文本1，5),(文本2，4),...]
    """
    # 1.初始化结果列表
    results = []
    # 2.加载原始文件，进行处理
    with open(data_path, 'r', encoding='utf-8') as f:
        # 1.遍历行，获取每行内容
        for line in f:
            # 1.去除首尾空白
            line = line.strip()
            # 2.跳过空行
            if not line:
                continue
            # 3.分割文本text和标签label,分割符为\t
            text, label = line.split('\t')
            # 4.添加元组(文本，标签索引)到结果列表中
            results.append((text, int(label)))
    # 3.返回结果
    return results
# 构造样本的输入输出, MyDataset类
# 2.2 自定义MyDataset类,构造样本的输入和输出，继承自Dataset
class MyDataset(Dataset):
    """
    自定义的数据集对象类，构造单个样本的输入和输出
    __init__: 初始化方法，接收原始的数据列表
    __len__: 返回数据集大小
    __getitem__: 根据索引来获取单个样本
    """
    # 1.__init__: 初始化方法，接收原始的数据列表
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
    # 2.__len__: 返回数据集大小
    def __len__(self):
        return len(self.data_list)
    # 3.__getitem__: 根据索引来获取单个样本
    def __getitem__(self, index):
        # 0.获取当前索引对应的元组(文本，标签索引)
        # sample = self.data_list[index]
        # 1.获取当前索引对应样本的输入text 和 输出label
        text, label = self.data_list[index]
        return text, label

# DataSet, collate_fn, DataLoader关系:
# Dataset:提供单个样本，
# collate_fn:将多个样本打包成标准等长批次
# Dataloader:自动按批次取数据
# 2.3 定义collate_fn函数 - 建议手敲
def collate_fn(batch):
    """
    collate_fn函数，将一个批次的样本整理为相同长度的样本，从而封装为一个等长的张量
    数据处理流程：batch文本 -> bert分词器编码：input_ids, attention_mask, token_type_ids(不需要句子id) -> 转为张量
    :param batch: 一个批次的样本，内容是[(文本1，5),(文本2，4),...]
    :return: 元组，代表输入张量 和 标签张量
    """
    # 1.获取当前批次的文本元组和标签元组
    texts, labels = zip(*batch)
    # 2.BERT分词器编码
    output = BERT_TOKENIZER(
        list(texts), # 文本列表
        padding=True, # 填充到当前批次的最大序列长度
        truncation=True, # 截断到最大长度
        max_length=config.max_len, # 最大长度
        return_tensors='pt', # 修复：返回pytorch张量
    )
    # output: 输出字典，包含input_ids, attention_mask, token_type_ids
    # input_ids: 输入句子的token id
    # attention_mask: 注意力机制的填充掩码
    # token_type_ids: 句子id,用于区分上下句
    # 形状都是BS
    input_ids = output.input_ids
    attention_mask = output.attention_mask
    labels = torch.tensor(labels, dtype=torch.long)
    # 5.返回输入张量和标签张量
    return input_ids, attention_mask, labels

# 2.4 构造数据加载器
def build_dataloader():
    # 1.加载数据集
    train_data = load_data(config.train_path)
    dev_data = load_data(config.dev_path)
    test_data = load_data(config.test_path)
    # 2.创建数据集对象
    train_dataset = MyDataset(train_data)
    dev_dataset = MyDataset(dev_data)
    test_dataset = MyDataset(test_data)
    # 3.创建数据加载器对象
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,  # 批次处理函数，把一个批次的样本整理为相同长度的样本
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_dataloader, dev_dataloader, test_dataloader

# 张量 -> 数据集对象 -> 数据加载器
# 3.搭建神经网络 - 要求手敲
# 预训练BERT，输出last_hidden_state(表示token语义), pooler_output(池化输出, 整个句子的语义表示)
# pooler_output句子表示
# 线性层nn.Linear
class MyBertModel(nn.Module):
    """
    基于预训练BERT的下游分类模型
    输入:
        input_ids: 输入的 token id序列, BS
        attention_mask: 填充掩码, BS
    返回:
        logits: 预测分数,(B,10)
    """
    # 1.定义__init__方法
    def __init__(self):
        super().__init__()
        # 1.初始化预训练BERT模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 2.初始化线性层，只有一层
        self.linear1 = nn.Linear(BERT_CONFIG.hidden_size, len(config.id2class))
        # 3.可选：冻结预训练BERT参数.可以调整为：刚开始冻结(0~2)，后期训练解冻(3~4)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
    # 2.定义forward方法
    def forward(self, input_ids, attention_mask):
        # input_ids: 输入的 token id序列, BS
        # attention_mask: 填充掩码, BS
        # 1.经过预训练BERT
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
        # 2.获取pooler_output
        # 修复：不能用torch.tensor，会截断梯度计算
        pooler_output = output.pooler_output
        # 3.经过线性层
        logits = self.linear1(pooler_output)    # (B,10)
        return logits

# 4.模型训练 - 训练集 + 验证集
# 4.1 训练一个轮次 - 要求手敲
def train_one_epoch(
            train_dataloader, model, loss_fn, optimizer):
    """
    训练一个轮次，返回训练损失、准确率、f1分数
    :param train_dataloader: 训练集的数据加载器
    :param model: 模型对象
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :return: 训练损失、准确率、f1分数
    """
    # 1.设为训练模式
    model.train()
    # 2.初始化总损失、总样本数、预测结果列表、真实标签列表
    total_loss = 0.0
    total_samples = 0
    total_preds = []
    total_labels = []
    # 3.遍历数据加载器，分批训练
    for batch in tqdm(train_dataloader, desc="Training"):
        # input_ids: BS 输入句子的token id
        # attention_mask: BS 注意力机制的填充掩码
        # labels: B 句子id,用于区分上下句
        # 0.迁移数据到设备
        input_ids, attention_mask, labels = [x.to(config.device) for x in batch]
        # 1.前向传播
        logits = model(input_ids, attention_mask)   # (B,10)
        # 2.计算损失
        loss = loss_fn(logits, labels)
        # 3.梯度清零
        optimizer.zero_grad()
        # 4.反向传播
        loss.backward()
        # 5.更新参数
        optimizer.step()
        # 6.更新训练指标
        total_loss += loss.item()*input_ids.shape[0]
        total_samples += input_ids.shape[0]
        # 获取当前批次的预测结果
        y_preds = torch.argmax(logits, dim=-1)  # (B,)
        total_preds.extend(y_preds.to("cpu").numpy().tolist())
        total_labels.extend(labels.to("cpu").numpy().tolist())
    # 4.计算 平均损失、准确率、f1分数
    avg_loss = total_loss / total_samples
    acc = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average="macro")
    # 5.返回结果
    return avg_loss, acc, f1

# 4.2 评估函数
@torch.no_grad()    # 关闭梯度计算,节省显存
def evaluate(val_dataloader, model, loss_fn):
    """
    评估一个轮次，返回 平均损失、准确率、f1分数
    :param val_dataloader: 评估集的数据加载器
    :param model: 模型对象
    :param loss_fn: 损失函数
    :return: 评估损失、准确率、f1分数
    """
    # 1.设为评估模式
    model.eval()
    # 2.初始化总损失、总样本数、预测结果列表、真实标签列表
    total_loss = 0.0
    total_samples = 0
    total_preds = []
    total_labels = []
    # 3.遍历数据加载器，分批评估
    for batch in tqdm(val_dataloader, desc="Evaluating"):
        # input_ids: BS 输入句子的token id
        # attention_mask: BS 注意力机制的填充掩码
        # labels: B 句子id,用于区分上下句
        # 0.迁移数据到设备
        input_ids, attention_mask, labels = [x.to(config.device) for x in batch]
        # 1.前向传播
        logits = model(input_ids, attention_mask)  # (B,10)
        # 2.计算损失
        loss = loss_fn(logits, labels)
        # # 3.梯度清零
        # optimizer.zero_grad()
        # # 4.反向传播
        # loss.backward()
        # # 5.更新参数
        # optimizer.step()
        # 6.更新评估指标
        total_loss += loss.item() * input_ids.shape[0]
        total_samples += input_ids.shape[0]
        # 获取当前批次的预测结果
        y_preds = torch.argmax(logits, dim=-1)  # (B,)
        total_preds.extend(y_preds.to("cpu").numpy().tolist())
        total_labels.extend(labels.to("cpu").numpy().tolist())
    # 4.计算 平均损失、准确率、f1分数
    avg_loss = total_loss / total_samples
    acc = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average="macro")
    # 5.返回结果
    return avg_loss, acc, f1

# 4.3 训练主函数 - 要求手敲
def train():
    # 1.创建数据加载器对象
    train_dataloader, dev_dataloader, _ = build_dataloader()
    # 2.创建模型对象
    model = MyBertModel().to(config.device)
    # 3.定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # 4.初始化训练指标，最优验证F1分数、训练损失、训练准确率、训练F1分数、验证损失、验证准确率、验证F1分数
    best_val_f1 = 0.0
    train_losses = []
    train_accs = []
    train_f1s = []
    val_losses = []
    val_accs = []
    val_f1s = []
    # 5.开始训练，遍历轮数
    for epoch in range(config.epochs):
        # 0.初始化开始时间
        start_time = time.time()
        # 1.训练一个轮次
        train_loss, train_acc, train_f1 = train_one_epoch(
            train_dataloader, model, loss_fn, optimizer)
        # 2.评估模型
        val_loss, val_acc, val_f1 = evaluate(dev_dataloader, model, loss_fn)
        # 3.根据best_val_f1来保存最优模型
        if val_f1 > best_val_f1:
            # 1.更新当前最优f1
            best_val_f1 = val_f1
            # 2.保存当前模型
            torch.save(model.state_dict(), config.bert_model_path)
            print(f"保存模型到{config.bert_model_path}, 最优验证F1分数: {val_f1:.4f}")
        # 4.添加训练指标到列表中
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        # 5.打印训练指标，损失、准确率、f1分数
        print(f"Epoch: {epoch}/{config.epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"train_f1: {train_f1:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_acc: {val_acc:.4f} | "
              f"val_f1: {val_f1:.4f} | "
              f"time: {time.time() - start_time:.2f}s")
    # 6.返回训练结果
    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "train_f1s": train_f1s,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_f1s": val_f1s,
    }

# 4.4 可视化训练过程
def plot_history(history):
    # 1.设置x轴刻度，epoch轮次
    epochs = range(1, len(history['train_losses'])+1)
    # 2.设置画布大小
    plt.figure(figsize=(15, 5))
    # 3.绘制loss曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_losses'], label='train_loss')
    plt.plot(epochs, history['val_losses'], label='val_loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    # 4.绘制acc曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_accs'], label='train_acc')
    plt.plot(epochs, history['val_accs'], label='val_acc')
    plt.title('Acc Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.grid(True)
    plt.legend()
    # 5.绘制f1曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_f1s'], label='train_f1')
    plt.plot(epochs, history['val_f1s'], label='val_f1')
    plt.title('F1 Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.grid(True)
    plt.legend()

    # 优化布局并显示
    plt.tight_layout()
    plt.show()

# 测试
if __name__ == '__main__':
    # 1.准备数据集
    # train_data = load_data(config.train_path)
    # dev_data = load_data(config.dev_path)
    # test_data = load_data(config.test_path)
    # print(f"训练集样本量: {len(train_data)}")
    # print(f"验证集样本量: {len(dev_data)}")
    # print(f"测试集样本量: {len(test_data)}")
    # 创建数据加载器对象
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    # print(f"训练集批次数: {len(train_dataloader)}")
    # print(f"验证集批次数: {len(dev_dataloader)}")
    # print(f"测试集批次数: {len(test_dataloader)}")
    # 2.创建神经网络对象
    model = MyBertModel().to(config.device)
    # print(model)
    # 3.模型训练
    results = train()
    plot_history(results)
    # 4.模型测试
    # 加载模型参数
    model.load_state_dict(
        torch.load(
            config.bert_model_path, # 本地保存的模型参数文件
            weights_only=True, # 只加载模型权重参数，不加载其他对象，更加安全
            map_location=config.device, # 将模型加载到指定设备上，自动适配设备，防止训练或推理阶段的设备不一致
        )
    )
    # 评估模型-测试集
    test_loss, test_acc, test_f1 = evaluate(test_dataloader, model, loss_fn=nn.CrossEntropyLoss())
    print(f"模型测试结果: loss: {test_loss:.4f}, acc: {test_acc:.4f}, f1: {test_f1:.4f}")