# Controlled TF-IDF Experiment

This folder isolates one question: with the same v2 data, same sampled 20,000 rows,
same 16,000/4,000 split, and same RandomForest parameters, how much does fitting
TF-IDF on the full 20,000 rows change results compared with fitting TF-IDF only on
the 16,000 training rows?

Run:

```powershell
python 02-rf\controlled\controlled_tfidf_experiment.py
```

Outputs go to `artifacts/day02/controlled` and `reports/day02/controlled`.
