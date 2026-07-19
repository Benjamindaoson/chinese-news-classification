"""
演示
    BERT微调方案的模型预测
"""

# 导包
import time
import torch
from config import Config
from bert_train import MyBertModel
from transformers import BertTokenizer

# 1.创建配置对象
config = Config()
# 2.加载模型
model = MyBertModel().to(config.device)
model.load_state_dict(
    torch.load(
        config.bert_model_path, # 本地保存的模型参数文件
        weights_only=True, # 只加载模型权重参数，不加载其他对象，更加安全
        map_location=config.device, # 将模型加载到指定设备上，自动适配设备，防止训练或推理阶段的设备不一致
    )
)
# print(model)
# 3.初始化分词器
BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)
# print(BERT_TOKENIZER)
# 4.定义函数，实现模型预测
@torch.no_grad()
def predict_fun(data):
    """
    预测函数，输入文本字典，返回带有预测结果的字典
    :param data: 文本字典，格式{text: 文本内容}
    :return: 字典，格式{text: 文本内容, pred_class: 预测类别名称}
    """
    # 1.设为评估模式
    model.eval()
    # 2.获取输入文本
    text = data['text']
    # 3.分词器编码
    output = BERT_TOKENIZER(
        [text], # 输入文本列表
        padding=True,   # 填充到当前批次最大长度
        truncation=True, # 截断到最大长度
        max_length=config.max_len,  # 最大长度
        return_tensors='pt',    # 返回pytorch张量
    )
    # 4.模型预测，前向传播
    logits = model(
        input_ids = output["input_ids"].to(config.device),
        attention_mask = output["attention_mask"].to(config.device),
    )   # 2D: (batch_size, 10)
    # 5.获取预测类别
    y_pred_class = logits.argmax(dim=-1)[0].item()    # 1D: (batch_size,)
    y_pred_class = config.id2class[y_pred_class]
    # 6.返回结果
    data['pred_class'] = y_pred_class
    return data

# 测试
if __name__ == '__main__':
    # 1.创建输入文本
    data = {
        'text': '今天感觉很奇怪，因为肖彰竟然不嚣张了，是不是生病了，让我很担心他'
    }
    # 2.进行预测
    result = predict_fun(data)
    print(result)
    # 3.计算推理延迟
    start_time = time.time()
    for i in range(200):
        result = predict_fun(data)
    total_time = time.time() - start_time
    print(f"推理延迟为：{total_time / 200 * 1000:.3f}ms")
    ...





