# 1. 执行摘要

This is a course-style Chinese news/text classification project. The real implementation has 7 modules: basic demos, data preparation, Random Forest, FastText, BERT fine-tuning, LLM API classification, and BERT compression/distillation. The material is fairly complete because it includes raw/cleaned/processed data, a local `bert-base-chinese` artifact, and saved model files. It is not directly production-runnable: there is no `requirements.txt` or README runbook, the dataset splits overlap, RF validation leaks TF-IDF vocabulary, LLM API keys are committed in `.env`, and Flask/Streamlit are wired as local demos. Recommended conclusion: continue from the original course base, but fix Day 1/data hygiene and environment issues before retraining.

# 2. 项目真实定位

- Course claim: an end-to-end text classification system with RF, FastText, BERT, LLM prompts, compression, APIs, and UI.
- Actual code: a local course demo for 10-class news title classification. Data and model artifacts live in `01-data`; each course day has its own folder.
- Missing engineering layer: dependency lock/spec, unified configuration, tests, production serving, structured LLM evaluation, model registry, and reproducible experiment tracking.

# 3. 完整目录说明

- `00-test/`: Flask and Python basics demos, plus one image.
- `01-data/`: raw/cleaned/processed datasets, labels, stopwords, local BERT model files, and trained artifacts.
- `02-rf/`: RF preprocessing, training, testing, prediction function, Flask API, Streamlit page.
- `03-fasttext/`: FastText word/char data conversion, default/auto train scripts, prediction function, Flask API, Streamlit page.
- `04-bert/`: BERT training, prediction, quantization, pruning, Flask API, Streamlit page.
- `05-llm/`: DeepSeek/Qwen API demos, prompt classification, Flask API, Streamlit page, and `.env`.
- `06-bert_distill/`: copied BERT teacher flow, student distillation, copied quantization script, teacher/student API and UI.
- `.idea/`, `__pycache__/`: IDE metadata and runtime cache, not part of the source spine.

# 4. 七天课程与文件对应关系

| Day | Topic | Files | Input | Output | Trains model |
|---|---|---|---|---|---|
| Day 0 | Flask/Python demos | `00-test/*.py` | manual text/image | local demo page | no |
| Day 1 | Data cleaning and EDA | `01-data/config.py`, `01-data/data_eda.py` | `*_raw.txt` | `train/dev/test.txt`, EDA output | no |
| Day 2 | RF + TF-IDF | `02-rf/*.py` | `process_*.csv` | `rf_model.pkl`, `tfidf_model.pkl`, API/UI | yes |
| Day 3 | FastText | `03-fasttext/*.py` | `process_*.csv` | `word_*.txt`, `char_*.txt`, `ft_model_*_1.bin` | yes |
| Day 4 | BERT fine-tuning | `04-bert/*.py` | `bert-base-chinese`, `train/dev/test.txt` | `bert_model.pt`, API/UI | yes |
| Day 5 | LLM prompt classification | `05-llm/*.py` | `.env`, input text | API classification text | no local training |
| Day 6/7 | Compression/distillation | `04-bert/bert_quantization.py`, `04-bert/bert_pruning.py`, `06-bert_distill/*.py` | `bert_model.pt`, dataset | `bert_quantization.pt`, `bert_pruning.pt`, `student_model.pt` | partly |

# 5. 数据流

| File | Lines | Duplicate lines | Format | Notes |
|---|---:|---:|---|---|
| `01-data/train_raw.txt` | 180000 | 1000 | tab text/label | balanced raw training split |
| `01-data/dev_raw.txt` | 10000 | 7 | tab text/label | raw validation split |
| `01-data/test_raw.txt` | 10000 | 10 | tab text/label | raw test split |
| `01-data/train.txt` | 179000 | 0 | tab text/label | cleaned training split |
| `01-data/dev.txt` | 9993 | 0 | tab text/label | cleaned validation split |
| `01-data/test.txt` | 9990 | 0 | tab text/label | cleaned test split |
| `01-data/process_train.csv` | 179001 | 0 | csv: text,label,words,seq_len | RF/FastText processed train data |
| `01-data/process_dev.csv` | 9994 | 0 | csv: text,label,words,seq_len | processed dev data |
| `01-data/process_test.csv` | 9991 | 0 | csv: text,label,words,seq_len | processed test data |
| `01-data/word_train.txt` | 179000 | 0 | FastText labels | word-level FastText train data |
| `01-data/char_train.txt` | 179000 | 0 | FastText labels | char-level FastText train data |

Cross-split overlap found by exact line comparison:

- `train.txt` and `dev.txt`: 87 overlapping lines.
- `train.txt` and `test.txt`: 105 overlapping lines.
- `dev.txt` and `test.txt`: 7 overlapping lines.
- `process_train.csv` and `process_dev.csv`: 88 overlapping rows.
- `process_train.csv` and `process_test.csv`: 106 overlapping rows.
- `process_dev.csv` and `process_test.csv`: 8 overlapping rows.

# 6. 模型流

- Random forest: `02-rf/rf_train.py:54` runs `fit_transform` before `02-rf/rf_train.py:60` splits train/validation, so the validation vocabulary leaks. It saves `01-data/model/tfidf_model.pkl` and `01-data/model/rf_model.pkl`.
- FastText: default word/char scripts save `_word_1.bin` and `_char_1.bin`. Auto-tune scripts would save `_2.bin` artifacts, but those are not present. The scripts compute a harmonic value from FastText P@1/R@1 and print it as F1; that is not macro-F1.
- BERT: `04-bert/bert_train.py` has train/dev/test dataloaders, best dev-F1 saving, and final test evaluation. It lacks fixed global seed, learning-rate scheduler, gradient clipping, and error sample analysis.
- LLM: `05-llm/*.py` calls DeepSeek/Qwen-compatible APIs. It has no structured output contract, timeout, retry, or batch evaluation.
- Quantization: `04-bert/bert_quantization.py` and the copy under `06-bert_distill` use dynamic quantization for `nn.Linear` and save `bert_quantization.pt`.
- Pruning: `04-bert/bert_pruning.py` globally prunes BERT attention query weights and saves `bert_pruning.pt`. Its comment correctly notes that unstructured pruning does not directly reduce model size.
- Distillation: `06-bert_distill/student_train.py` uses CE + KL with `alpha=0.7` and `T=4`, then saves `student_model.pt`.

# 7. 部署流

All model families expose a Flask script and a Streamlit script. RF/FastText/BERT/LLM use `/predict`; distillation uses `/predict01` and `/predict02`. Every Flask server uses `app.run(host='127.0.0.1', port=5000, debug=True)`, so these are local development servers. Streamlit pages call `http://127.0.0.1:5000/...` and require the matching Flask process to be running first.

# 8. 逐文件审计

Important source files:

- `01-data/config.py`: defines data paths and `remove_duplicates`; uses current relative root but keeps an old absolute-path comment.
- `01-data/data_eda.py`: reads `train/dev/test.txt` and prints/plots basic stats; safe as EDA, not a pipeline.
- `02-rf/data_process.py`: reads cleaned txt splits, tokenizes with jieba, writes `process_*.csv`.
- `02-rf/rf_train.py`: trains TF-IDF + RandomForest; has validation leakage because vectorizer is fitted before split.
- `02-rf/rf_test.py`: loads saved RF/vectorizer and evaluates `process_test.csv`.
- `02-rf/rf_predict_fun.py`: loads RF/vectorizer at import and predicts one text.
- `02-rf/api_flask_server.py`: local RF Flask API.
- `02-rf/streamlit_app.py`: RF UI; fixed local API URL and old absolute-path comment.
- `03-fasttext/data_process.py`: converts processed CSV to FastText word/char text files.
- `03-fasttext/fasttext_*_default.py`: trains default FastText models and saves `_1.bin`.
- `03-fasttext/fasttext_*_auto.py`: auto-tune FastText scripts; generated `_2.bin` files are absent.
- `03-fasttext/fasttext_predict_fun.py`: loads `ft_model_char_1.bin` at import and predicts one text.
- `04-bert/bert_train.py`: BERT dataset/model/training/evaluation; course-suitable but not fully reproducible.
- `04-bert/bert_predict_fun.py`: loads `bert_model.pt` and predicts one text.
- `04-bert/bert_quantization.py`: dynamic quantization; PyTorch API compatibility should be pinned.
- `04-bert/bert_pruning.py`: non-structured pruning; keeps evaluation but not speed/size benchmark.
- `05-llm/deepseek_predict_fun.py`: remote API classifier; lacks timeout/retry/structured schema and depends on `.env`.
- `06-bert_distill/student_train.py`: student distillation; meaningful course script, still lacks seed/scheduler/clip.
- `06-bert_distill/bert_predict_fun.py`: loads teacher and student once at import.

# 9. 可运行性审计

- Static syntax check by AST: no `SyntaxError` found in 46 Python files.
- Notebooks: none found.
- Dependency files: no `requirements.txt`, `pyproject.toml`, `environment.yml`, or README runbook found.
- Current Python: 3.12.10. Cached pyc names and comments indicate Python 3.10 was used in the original course environment.
- External dependencies inferred statically: `pandas`, `jieba`, `sklearn`, `fasttext`, `torch`, `transformers`, `flask`, `streamlit`, `requests`, `openai`, `python-dotenv`, `matplotlib`, `seaborn`, `tqdm`, `numpy`.
- No model training, API call, dependency install, or source mutation was executed during this audit.

# 10. 问题清单

## P0

- `05-llm/.env`: contains live API keys. Why it matters: key leakage can cause account abuse and cost. Fix: rotate the keys, remove real secrets from the course package, and create `.env.example`. Do not wait.
- Missing dependency/run specification at project root. Why it matters: a new machine cannot reliably run the project. Fix: add requirements and a minimal runbook before Day 1 execution.

## P1

- `02-rf/rf_train.py:54` and `02-rf/rf_train.py:60`: TF-IDF is fitted before validation split. Fix: split first, then fit only on training text.
- Dataset split overlap: train/dev/test share exact rows. Fix: deduplicate globally by text+label before splitting.
- `03-fasttext/fasttext_*`: P@1/R@1 harmonic score is printed as F1. Fix: batch-predict test data and compute macro-F1 with sklearn for comparability.
- `05-llm/*.py`: remote model name and response format are not locally verifiable, and calls have no timeout/retry. Fix: verify current official model names, use JSON output, and add bounded retries.

## P2

- Repeated `config.py` files drift across modules. Fix later by sharing config, after Day 1 is stable.
- Streamlit comments point to `02-rf/streamlit_app.py` even in other modules. Fix documentation when entering each day.
- Flask development server is used as the deployment path. Fix docs to say local demo only; use a real WSGI/ASGI server for production.
- BERT training lacks seed, scheduler, gradient clipping, and error analysis. Add after baseline reproduction.
- `torch.quantization.quantize_dynamic` may drift across PyTorch versions. Pin PyTorch or update to the currently supported API.

## P3

- `.idea/` and `__pycache__/` are cache/IDE artifacts.
- `04-bert/bert_train.py` duplicates `06-bert_distill/bert_train.py`; quantization is duplicated too.
- No tests or smoke scripts exist for prediction functions or APIs.

# 11. 建议保留与删除

- Must keep: `01-data/*.txt`, `01-data/*.csv`, `01-data/bert-base-chinese/*`, `01-data/model/*`, source under `02-rf` to `06-bert_distill`.
- Can archive: `.idea/`, `__pycache__/`, and `00-test/` once course basics are no longer needed.
- Generated after runs: `process_*.csv`, `word_*.txt`, `char_*.txt`, `01-data/model/*`.
- Do not touch yet: data and model artifacts until Day 1 reproducibility is confirmed.

# 12. 推荐实操顺序

1. Day 1: fix/verify data paths, deduplication, EDA, and split overlap. Do not train.
2. Day 2: fix RF TF-IDF split leakage, retrain RF, save artifacts.
3. Day 3: regenerate FastText input, run default train first, compute unified macro-F1. Auto-tune later.
4. Day 4: pin BERT environment, set seeds, then fine-tune.
5. Day 5: rotate API keys, add structured LLM output and batch eval before any API calls.
6. Day 6: run quantization/pruning from a known-good `bert_model.pt`.
7. Day 7: run distillation, then API/UI smoke checks.

# 13. Day 1详细计划

- Open first: `01-data/config.py`, `01-data/data_eda.py`, `train_raw.txt`, `dev_raw.txt`, `test_raw.txt`, `class.txt`, `stopwords.txt`.
- First operation: verify resolved paths and dataset counts only.
- Fix candidates for next phase: stale absolute-path comments, split overlap, and the absence of a requirements/runbook file.
- Check outputs: raw vs cleaned row counts, label distribution, empty rows, duplicate rows, and cross-split overlap.
- Do not do yet: BERT training, FastText auto-tune, LLM calls, config refactor.
- Completion standard: data paths resolve, splits are reproducible, and train/dev/test have no cross overlap.

# 14. 最终结论

Main conclusion: fix on top of the original course base. The project has enough real assets and runnable-looking scripts to continue, but the data leakage, secret exposure, missing dependencies, and local-only deployment assumptions must be handled before serious training or demo delivery.

# Appendix: Audit Counts

- Directories scanned: 19
- Files scanned before audit artifacts: 97
- Python files read: 46
- Notebooks found/read: 0
- Data files found: 17
- Model/weight files found: 9
- Static syntax errors: 0
- Hardcoded absolute-path mentions: 11
- Generated audit files: `PROJECT_AUDIT.md`, `FILE_INVENTORY.csv`, `PROJECT_TREE.txt`, `CODE_DEPENDENCY_MAP.md`
