# CODE_DEPENDENCY_MAP

## 1. Module Call Graph

```text
data/config.py
-> data/{train_raw,dev_raw,test_raw}.txt
-> data/{train,dev,test}.txt
-> data/data_eda.py

random_forest/data_process.py
-> data/{train,dev,test}.txt
-> data/process_{train,dev,test}.csv
-> random_forest/rf_train.py
-> data/model/{tfidf_model.pkl,rf_model.pkl}
-> random_forest/rf_test.py
-> random_forest/rf_predict_fun.py
-> random_forest/api_flask_server.py
-> random_forest/streamlit_app.py and random_forest/api_test.py

fasttext/data_process.py
-> data/process_{train,dev,test}.csv
-> data/{word,char}_{train,dev,test}.txt
-> fasttext/fasttext_{word,char}_1_default.py
-> data/model/ft_model_{word,char}_1.bin
-> fasttext/fasttext_predict_fun.py
-> fasttext/api_flask_server.py
-> fasttext/streamlit_app.py

bert_finetuning/bert_train.py
-> data/bert-base-chinese + data/{train,dev,test}.txt
-> data/model/bert_model.pt
-> bert_finetuning/bert_predict_fun.py
-> bert_finetuning/api_flask_server.py
-> bert_finetuning/streamlit_app.py

bert_finetuning/bert_quantization.py
-> data/model/bert_model.pt
-> data/model/bert_quantization.pt

bert_finetuning/bert_pruning.py
-> data/model/bert_model.pt
-> data/model/bert_pruning.pt

llm_prompting/deepseek_predict_fun.py
-> llm_prompting/.env + remote DeepSeek API
-> llm_prompting/api_flask_server.py
-> llm_prompting/streamlit_app.py

bert_compression/student_train.py
-> data/model/bert_model.pt + data/{train,dev,test}.txt
-> data/model/student_model.pt
-> bert_compression/bert_predict_fun.py
-> bert_compression/api_flask_server.py
-> bert_compression/streamlit_app.py
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

- Random forest: `random_forest/rf_train.py` reads `process_train.csv`, fits TF-IDF, splits a validation subset, trains `RandomForestClassifier`, and saves `tfidf_model.pkl` plus `rf_model.pkl`.
- FastText: `fasttext/fasttext_word_1_default.py` and `fasttext_char_1_default.py` train the saved `_word_1.bin` and `_char_1.bin` models. The `_2_auto.py` scripts would save `_2.bin` artifacts; those artifacts are not present.
- BERT: `bert_finetuning/bert_train.py` builds train/dev/test dataloaders, saves best dev-F1 weights to `bert_model.pt`, then evaluates the saved model on test data.
- Quantization: `bert_finetuning/bert_quantization.py` loads `bert_model.pt`, applies dynamic quantization to `nn.Linear`, evaluates, and saves `bert_quantization.pt`.
- Pruning: `bert_finetuning/bert_pruning.py` loads `bert_model.pt`, prunes attention query weights globally, evaluates, and saves `bert_pruning.pt`.
- Distillation: `bert_compression/student_train.py` loads teacher `bert_model.pt`, trains a smaller student with CE + KL loss, saves best dev-F1 weights to `student_model.pt`, then evaluates teacher/student.

## 4. Inference Flow

- RF, FastText, BERT, and distillation prediction modules load model artifacts at module import time. They do not reload the model on every request.
- LLM prediction calls the remote API per request.
- Flask endpoints:
  - RF/FastText/BERT/LLM: `/predict`
  - Distillation: `/predict01` for teacher BERT, `/predict02` for student/distilled model
- Streamlit pages call `http://127.0.0.1:5000/...`, so the matching Flask server must already be running.

## 5. Shared Files

- `data/model/*` is the shared artifact directory for RF, FastText, BERT, quantization, pruning, and distillation.
- `data/{train,dev,test}.txt` feeds BERT and upstream preprocessing.
- `data/process_*.csv` feeds RF training/testing and FastText input conversion.
- There is no single shared application config. Each model folder has its own `config.py` copy.

## 6. Duplicate Names

- `api_flask_server.py`: repeated in `random_forest`, `fasttext`, `bert_finetuning`, `llm_prompting`, `bert_compression`.
- `streamlit_app.py`: repeated in `random_forest`, `fasttext`, `bert_finetuning`, `llm_prompting`, `bert_compression`.
- `api_test.py`: identical in `random_forest`, `bert_finetuning`, `bert_compression`.
- `bert_train.py`: identical in `bert_finetuning` and `bert_compression`.
- `bert_quantization.py`: identical in `bert_finetuning` and `bert_compression`.
- `config.py`: repeated with similar but drifting path/model settings in `data`, `random_forest`, `fasttext`, `bert_finetuning`, `bert_compression`.

## 7. Main Execution Spine

```text
data/config.py
-> data/data_eda.py
-> random_forest/data_process.py
-> random_forest/rf_train.py
-> random_forest/rf_test.py
-> fasttext/data_process.py
-> fasttext/fasttext_*_1_default.py
-> bert_finetuning/bert_train.py
-> llm_prompting/02_提示词工程完成文本分类.py
-> bert_finetuning/bert_quantization.py and bert_finetuning/bert_pruning.py
-> bert_compression/student_train.py
-> matching Flask API
-> matching Streamlit UI
```
