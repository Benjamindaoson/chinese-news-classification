"""
    配置文件，统一管理全局配置信息，包括数据路径、类别信息等。方便项目的管理和后期维护。
"""
import os   # 文件操作模块
from pathlib import Path  # 路径操作模块
# 不显示警告信息
import warnings
warnings.filterwarnings("ignore")

# 1.定义配置类，集中管理全局配置信息，比如数据路径、类别信息
class Config():
    # 1.定义__init__方法，初始化配置项
    def __init__(self):
        # 1.初始化根路径
        self.root_path = os.path.join(Path(__file__).parent.parent, "01-data")
        # self.root_path = r"D:\tmf_project_shenzhen_AI_4\07_live_code\01-data"
        # 2.初始化数据文件路径
        self.train_path = os.path.join(self.root_path, "train.txt")
        self.dev_path = os.path.join(self.root_path, "dev.txt")
        self.test_path = os.path.join(self.root_path, "test.txt")
        # 停用词文件路径
        self.stopwords_path = os.path.join(self.root_path, "stopwords.txt")
        # 3.初始化类别文件路径
        self.class_path = os.path.join(self.root_path, "class.txt")
        # 4.初始化 处理后的数据文件路径
        self.process_train_path = os.path.join(self.root_path, "process_train.csv")
        self.process_dev_path = os.path.join(self.root_path, "process_dev.csv")
        self.process_test_path = os.path.join(self.root_path, "process_test.csv")
        # 5.模型保存路径
        os.makedirs(os.path.join(self.root_path, "model"), exist_ok=True)
        # pkl格式，用于保存sklearn机器学习模型
        self.rf_model_path = os.path.join(self.root_path, "model", "rf_model.pkl")
        self.tfidf_model_path = os.path.join(self.root_path, "model", "tfidf_model.pkl")
        # 6.模型预测结果路径，txt
        os.makedirs(os.path.join(self.root_path, "result"), exist_ok=True)
        self.model_predict_result = os.path.join(self.root_path, "result", "model_predict_result.txt")
        # 7.API配置
        self.api_host = "127.0.0.1" # 本地地址
        self.api_port = 5000 # 端口号

# 2.测试
if __name__ == "__main__":
    # 1.创建配置对象
    config = Config()
    # 2.打印配置信息，验证是否正确加载
    print("根路径:", config.root_path)
    print("训练数据路径:", config.train_path)
    print("验证数据路径:", config.dev_path)
    print("测试数据路径:", config.test_path)
    print("停用词文件路径:", config.stopwords_path)
    print("类别文件路径:", config.class_path)
    print(f"处理后的训练数据路径: {config.process_train_path}")
    print(f"服务器地址: api_host: {config.api_host} | "
          f"端口号:{config.api_port}")
