# CODE_DEPENDENCY_MAP

## 1. Module Call Graph

```text
01-data/config.py
-> 01-data/{train_raw,dev_raw,test_raw}.txt
-> 01-data/{train,dev,test}.txt
-> 01-data/data_eda.py

02-rf/data_process.py
-> 01-data/{train,dev,test}.txt
-> 01-data/process_{train,dev,test}.csv
-> 02-rf/rf_train.py
-> 01-data/model/{tfidf_model.pkl,rf_model.pkl}
-> 02-rf/rf_test.py
-> 02-rf/rf_predict_fun.py
-> 02-rf/api_flask_server.py
-> 02-rf/streamlit_app.py and 02-rf/api_test.py

03-fasttext/data_process.py
-> 01-data/process_{train,dev,test}.csv
-> 01-data/{word,char}_{train,dev,test}.txt
-> 03-fasttext/fasttext_{word,char}_1_default.py
-> 01-data/model/ft_model_{word,char}_1.bin
-> 03-fasttext/fasttext_predict_fun.py
-> 03-fasttext/api_flask_server.py
-> 03-fasttext/streamlit_app.py

04-bert/bert_train.py
-> 01-data/bert-base-chinese + 01-data/{train,dev,test}.txt
-> 01-data/model/bert_model.pt
-> 04-bert/bert_predict_fun.py
-> 04-bert/api_flask_server.py
-> 04-bert/streamlit_app.py

04-bert/bert_quantization.py
-> 01-data/model/bert_model.pt
-> 01-data/model/bert_quantization.pt

04-bert/bert_pruning.py
-> 01-data/model/bert_model.pt
-> 01-data/model/bert_pruning.pt

05-llm/deepseek_predict_fun.py
-> 05-llm/.env + remote DeepSeek API
-> 05-llm/api_flask_server.py
-> 05-llm/streamlit_app.py

06-bert_distill/student_train.py
-> 01-data/model/bert_model.pt + 01-data/{train,dev,test}.txt
-> 01-data/model/student_model.pt
-> 06-bert_distill/bert_predict_fun.py
-> 06-bert_distill/api_flask_server.py
-> 06-bert_distill/streamlit_app.py
```

## 2. Data Flow

```text
raw data:
  train_raw.txt, dev_raw.txt, test_raw.txt
cleaned data:
  train.txt, dev.txt, test.txt
processed CSV:
  process_train.csv, process_dev.csv, process_test.csv
FastText input:
  word_train/dev/test.txt, char_train/dev/test.txt
model artifacts:
  tfidf_model.pkl, rf_model.pkl, ft_model_*_1.bin,
  bert_model.pt, bert_quantization.pt, bert_pruning.pt, student_model.pt
web/API:
  api_flask_server.py -> streamlit_app.py/api_test.py
```

## 3. Training Flow

- Random forest: `02-rf/rf_train.py` reads `process_train.csv`, fits TF-IDF, splits a validation subset, trains `RandomForestClassifier`, and saves `tfidf_model.pkl` plus `rf_model.pkl`.
- FastText: `03-fasttext/fasttext_word_1_default.py` and `fasttext_char_1_default.py` train the saved `_word_1.bin` and `_char_1.bin` models. The `_2_auto.py` scripts would save `_2.bin` artifacts; those artifacts are not present.
- BERT: `04-bert/bert_train.py` builds train/dev/test dataloaders, saves best dev-F1 weights to `bert_model.pt`, then evaluates the saved model on test data.
- Quantization: `04-bert/bert_quantization.py` loads `bert_model.pt`, applies dynamic quantization to `nn.Linear`, evaluates, and saves `bert_quantization.pt`.
- Pruning: `04-bert/bert_pruning.py` loads `bert_model.pt`, prunes attention query weights globally, evaluates, and saves `bert_pruning.pt`.
- Distillation: `06-bert_distill/student_train.py` loads teacher `bert_model.pt`, trains a smaller student with CE + KL loss, saves best dev-F1 weights to `student_model.pt`, then evaluates teacher/student.

## 4. Inference Flow

- RF, FastText, BERT, and distillation prediction modules load model artifacts at module import time. They do not reload the model on every request.
- LLM prediction calls the remote API per request.
- Flask endpoints:
  - RF/FastText/BERT/LLM: `/predict`
  - Distillation: `/predict01` for teacher BERT, `/predict02` for student/distilled model
- Streamlit pages call `http://127.0.0.1:5000/...`, so the matching Flask server must already be running.

## 5. Shared Files

- `01-data/model/*` is the shared artifact directory for RF, FastText, BERT, quantization, pruning, and distillation.
- `01-data/{train,dev,test}.txt` feeds BERT and upstream preprocessing.
- `01-data/process_*.csv` feeds RF training/testing and FastText input conversion.
- There is no single shared application config. Each model folder has its own `config.py` copy.

## 6. Duplicate Names

- `api_flask_server.py`: repeated in `02-rf`, `03-fasttext`, `04-bert`, `05-llm`, `06-bert_distill`.
- `streamlit_app.py`: repeated in `02-rf`, `03-fasttext`, `04-bert`, `05-llm`, `06-bert_distill`.
- `api_test.py`: identical in `02-rf`, `04-bert`, `06-bert_distill`.
- `bert_train.py`: identical in `04-bert` and `06-bert_distill`.
- `bert_quantization.py`: identical in `04-bert` and `06-bert_distill`.
- `config.py`: repeated with similar but drifting path/model settings in `01-data`, `02-rf`, `03-fasttext`, `04-bert`, `06-bert_distill`.

## 7. Main Execution Spine

```text
01-data/config.py
-> 01-data/data_eda.py
-> 02-rf/data_process.py
-> 02-rf/rf_train.py
-> 02-rf/rf_test.py
-> 03-fasttext/data_process.py
-> 03-fasttext/fasttext_*_1_default.py
-> 04-bert/bert_train.py
-> 05-llm/02_提示词工程完成文本分类.py
-> 04-bert/bert_quantization.py and 04-bert/bert_pruning.py
-> 06-bert_distill/student_train.py
-> matching Flask API
-> matching Streamlit UI
```
