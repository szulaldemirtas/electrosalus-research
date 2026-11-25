# Cough Classification

Notebooks that classify recordings as cough vs. other sounds, and coughs as dry vs. wet.

| Notebook | What it does |
|---|---|
| `cough-detection.ipynb` | Extracts audio features and compares several classifiers for cough vs. other sounds, with grid search and 5-fold cross-validation |
| `classification-knn-svm.ipynb` | KNN and SVM classifiers on frequency-percentile and MFCC features, with principal component analysis |
| `cough-detection-neural-network.ipynb` | Neural network on mel spectrograms (work in progress) |
| `cough-dry-wet-detection.ipynb` | Classifies coughs as dry or wet (work in progress) |
| `test-filter.ipynb` | Checks that the signal filter implementation is correct |
