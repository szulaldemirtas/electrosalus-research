# Cough Detection from Audio

Signal-processing and machine-learning methods for detecting coughs in audio recordings, built during my machine learning internship at Electrosalus (2024). The work covers three tasks:

1. **Cough extraction:** finding the start and end time of each cough in a recording
2. **Cough vs. other sounds:** telling coughs apart from laughing, clapping, knocking, speech and crowd noise
3. **Dry vs. wet coughs:** an early attempt at classifying cough type

## Repository structure

| Folder | Contents |
|---|---|
| [`recordings-and-timestamps/`](recordings-and-timestamps) | The dataset: self-recorded, online and [COUGHVID](https://zenodo.org/records/4048312) audio, with start/end timestamps for each event |
| [`cough-extraction/`](cough-extraction) | Notebooks that detect and cut out cough segments, in the time domain and the frequency domain, plus an evaluation notebook |
| [`cough-classification/`](cough-classification) | Notebooks for feature extraction and classification (KNN, SVM, random forest, neural network) |
| `filename-editor.ipynb` | Helper for renaming files and converting COUGHVID labels into timestamp files |

Each folder has its own README with a description of every notebook.

## Methods

- **Preprocessing:** band-pass filtering, time binning, moving averages and normalisation
- **Extraction:** threshold-based peak detection in the time domain and on spectrograms, evaluated against the labelled timestamps with precision, recall and F1
- **Features:** frequency percentiles, MFCCs, spectral centroid and energy, with PCA for analysis
- **Classification:** KNN, SVM and random forest tuned with grid search and 5-fold cross-validation; a PyTorch neural network on mel spectrograms with data augmentation (unfinished)

## Results

- **Cough vs. other sounds:** mean 5-fold cross-validation F1 of 0.96–0.97 across the models tested
- **Dry vs. wet coughs:** mean F1 of 0.47–0.56; this task was still in progress when the internship ended, and the small labelled set is the main limitation

## Running the notebooks

The notebooks use Python with NumPy, SciPy, librosa, scikit-learn, PyTorch and Matplotlib. Some cells contain absolute paths from my machine; change them to your local copy of `recordings-and-timestamps/` before running.
