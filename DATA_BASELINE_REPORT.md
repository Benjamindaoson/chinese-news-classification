# 1. 当前Python环境

- Python版本：`Python 3.12.10`
- `pandas`：可导入，版本 `3.0.3`
- `matplotlib`：可导入，版本 `3.10.9`
- `seaborn`：可导入，版本 `0.13.2`
- 本轮未安装、升级任何依赖。

# 2. config.py真实路径逻辑

审计文件：`data/config.py`

- 当前生效代码没有使用绝对路径。
- `root_path` 当前由 `os.path.join(Path(__file__).parent.parent, "data")` 计算。
- 最终解析出的 `data` 绝对路径为：`D:\投满条配套资料\08-live_code\data`
- 文件中仍保留一行旧绝对路径注释：`D:\tmf_project_shenzhen_AI_4\07_live_code\data`，但该行未执行。
- `remove_duplicates(input_file, output_file)` 会读入输入文件、按完整行去重、写出输出文件。
- 三次 `remove_duplicates(...)` 调用位于 `if __name__ == "__main__":` 下；导入 `Config` 时不会自动去重或覆盖数据。

# 3. data_eda.py真实执行逻辑

审计文件：`data/data_eda.py`

- 使用 `from config import Config` 导入配置类。
- 以脚本方式运行 `python data\data_eda.py` 时，Python会把 `data` 加入脚本搜索路径，因此能正确导入同目录的 `config.py`。
- 代码在模块顶层直接执行；没有 `if __name__ == "__main__":` 保护。
- 顶层执行内容包括：创建 `Config()`、读取 `train.txt`、`dev.txt`、`test.txt`、打印形状和训练集样例、统计训练集标签分布、计算训练集文本长度、绘制两个直方图。
- 它同时读取 train/dev/test 三份 cleaned 数据，但详细标签分布和文本长度统计只针对 `train_data`。
- 静态检查未发现文件写入调用；只发现 `pd.read_csv(...)` 和 `plt.show()`。因此本轮使用 `MPLBACKEND=Agg` 安全运行，避免打开交互式图窗。
- 与课程讲义展示代码的逐字差异无法确认，因为本轮边界不允许读取讲义文件；只能确认当前源码与源码内注释/课程预期输出一致。

# 4. 文件存在性检查

由 `Config()` 解析出的路径如下，均存在：

| 配置项 | 解析路径 | 是否存在 |
|---|---|---|
| `root_path` | `D:\投满条配套资料\08-live_code\data` | 是 |
| `train_raw_path` | `D:\投满条配套资料\08-live_code\data\train_raw.txt` | 是 |
| `dev_raw_path` | `D:\投满条配套资料\08-live_code\data\dev_raw.txt` | 是 |
| `test_raw_path` | `D:\投满条配套资料\08-live_code\data\test_raw.txt` | 是 |
| `train_path` | `D:\投满条配套资料\08-live_code\data\train.txt` | 是 |
| `dev_path` | `D:\投满条配套资料\08-live_code\data\dev.txt` | 是 |
| `test_path` | `D:\投满条配套资料\08-live_code\data\test.txt` | 是 |
| `stopwords_path` | `D:\投满条配套资料\08-live_code\data\stopwords.txt` | 是 |
| `class_path` | `D:\投满条配套资料\08-live_code\data\class.txt` | 是 |

五个必需输入文件 `train_raw.txt`、`dev_raw.txt`、`test_raw.txt`、`class.txt`、`stopwords.txt` 均存在。

# 5. 数据行数检查

| 文件 | 行数 |
|---|---:|
| `data/train_raw.txt` | 180000 |
| `data/dev_raw.txt` | 10000 |
| `data/test_raw.txt` | 10000 |
| `data/train.txt` | 179000 |
| `data/dev.txt` | 9993 |
| `data/test.txt` | 9990 |

# 6. 数据格式检查

检查规则：每行应能按 Tab 分成两列，第二列标签非空，标签必须属于 `0` 到 `9`。

| 文件 | 空行 | Tab非两列异常行 | 标签为空 | 标签不在0-9 |
|---|---:|---:|---:|---:|
| `data/train_raw.txt` | 0 | 0 | 0 | 0 |
| `data/dev_raw.txt` | 0 | 0 | 0 | 0 |
| `data/test_raw.txt` | 0 | 0 | 0 | 0 |
| `data/train.txt` | 0 | 0 | 0 | 0 |
| `data/dev.txt` | 0 | 0 | 0 | 0 |
| `data/test.txt` | 0 | 0 | 0 | 0 |

结论：六个数据文件本轮未发现格式异常。

# 7. 内部重复检查

| 文件 | 完整行重复数量 |
|---|---:|
| `data/train_raw.txt` | 1000 |
| `data/dev_raw.txt` | 7 |
| `data/test_raw.txt` | 10 |
| `data/train.txt` | 0 |
| `data/dev.txt` | 0 |
| `data/test.txt` | 0 |

结论：raw文件中存在重复行；cleaned文件内部重复已被清除。本轮未重新生成或覆盖 cleaned 文件。

# 8. EDA运行结果

运行命令：`$env:MPLBACKEND='Agg'; python 'data\data_eda.py'`

运行结果：

- `train_data.shape`：`(179000, 2)`
- `dev_data.shape`：`(9993, 2)`
- `test_data.shape`：`(9990, 2)`
- 训练集标签分布：`Counter({7: 18000, 9: 17999, 5: 17998, 3: 17982, 4: 17982, 6: 17982, 8: 17963, 0: 17936, 2: 17746, 1: 17412})`
- 训练集文本长度：
  - count：`179000`
  - mean：`19.209056`
  - std：`3.854947`
  - min：`3`
  - 25%：`17`
  - 50%：`19`
  - 75%：`22`
  - max：`38`

运行时出现两条 `FigureCanvasAgg is non-interactive` 警告，这是因为本轮使用非交互式绘图后端避免弹窗；不影响数据统计结果。

# 9. 与课程参考结果对比

| 指标 | 课程期望 | 本轮结果 | 是否一致 |
|---|---:|---:|---|
| 训练集条数 | 179000 | 179000 | 是 |
| 平均文本长度 | 约19.209 | 19.209056 | 是 |
| 最短文本长度 | 3 | 3 | 是 |
| 最长文本长度 | 38 | 38 | 是 |
| 10类大体均衡 | 是 | 是 | 是 |

结论：当前 `data_eda.py` 能复现课程基线EDA结果。

# 10. 当前阻碍Day 1运行的问题

- `config.py` 直接运行会覆盖 `train.txt`、`dev.txt`、`test.txt`，因此本轮没有执行 `python data\config.py`。
- `data_eda.py` 没有主函数保护，导入该模块也会立即读取数据并绘图；作为课程脚本可接受，作为工程模块不理想。
- 当前源码仍保留旧绝对路径注释，虽然不影响运行，但容易误导后续学员。
- 本轮没有发现阻止 `data_eda.py` 运行的实际问题。

# 11. 下一阶段建议修改项

- 不要先改模型代码；下一阶段应只处理 Day 1 数据层。
- 给 `data_eda.py` 增加 `main()` 和 `if __name__ == "__main__":`，避免导入即执行。
- 将 `config.py` 的去重写文件操作拆成显式命令或新增安全确认，避免误覆盖 cleaned 数据。
- 移除或更新旧绝对路径注释。
- 后续再检查 train/dev/test 之间的跨集合重复；本阶段只确认了文件内部重复。
