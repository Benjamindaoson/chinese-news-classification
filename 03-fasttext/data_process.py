"""
进行数据预处理，将原始文本数据 整理为 fasttext有监督学习需要的输入格式txt

fasttext有监督学习的数据格式为：
    __label__finance 传 郭 晶 晶 欲 落 户 香 港 战 伦 敦 奥 运
    __label__标签名称 分词之后的结果，分隔符为空格
这里构造两种文本格式：按字分词 + 按词分词
"""
# 导包
import jieba
from config import Config

# 1.创建配置对象
config = Config()

# 2.定义函数，将原始文本 整理为 fasttext有监督学习需要的输入格式
def process_data(data_path, save_path, is_char=True):
    """
    对原始文本进行分词，整理为fasttext有监督学习需要的输入格式
    格式为：__label__标签名称 空格隔开的分词之后的结果
    :param data_path: 原始文本文件路径
    :param save_path: 处理之后的文本文件路径
    :param is_char: 是否按照字符进行分词，也就是按字分词，如果为False则进行jieba分词
    :return: 无
    """
    # 1.打开原始文本文件
    with open(data_path, 'r', encoding='utf-8') as f:
        # 2.写入处理之后的文本到save_path
        with open(save_path, 'w', encoding='utf-8') as fw:
            # 3.循环读取原始文本文件每行，进行处理
            for line in f:
                # 4.去除首尾空白
                line = line.strip()
                # 5.分割文本和标签
                text, label = line.split('\t')
                # 6.将label转换为标签名称
                label_name = config.id2class.get(int(label), "finance")
                # 7.分词，根据is_char
                # is_char=True 按字分词
                if is_char:
                    seq_tokens = " ".join(list(text))
                # is_char=False 按词分词,jieba分词
                else:
                    seq_tokens = " ".join(jieba.lcut(text))
                # 8.构建需要的数据格式：__label__标签名称 空格隔开的分词之后结果
                seq_tokens = f"__label__{label_name} {seq_tokens}\n"
                # 9.写入save_path文件
                fw.write(seq_tokens)

# 测试
if __name__ == '__main__':
    # 1.按字分词
    process_data(config.train_path, config.char_train_path, is_char=True)
    process_data(config.dev_path, config.char_dev_path, is_char=True)
    process_data(config.test_path, config.char_test_path, is_char=True)
    # 2.按词分词
    process_data(config.train_path, config.word_train_path, is_char=False)
    process_data(config.dev_path, config.word_dev_path, is_char=False)
    process_data(config.test_path, config.word_test_path, is_char=False)