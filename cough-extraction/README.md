# Cough Extraction

Notebooks that find and cut out the cough segments in audio recordings, using either time-domain or frequency-domain analysis.

| Notebook | What it does |
|---|---|
| `cough-extraction-single-recording-time-domain.ipynb` | Extracts coughs from one recording using the time-domain signal |
| `cough-extraction-single-recording-frequency-domain.ipynb` | Extracts coughs from one recording using its spectrogram |
| `cough-extraction-multiple-recordings-time-domain.ipynb` | Runs the time-domain extraction on every recording in a folder |
| `cough-extraction-multiple-recordings-frequency-domain.ipynb` | Runs the frequency-domain extraction on every recording in a folder |
| `measure-extraction-performance.ipynb` | Compares predicted cough timestamps with the labelled ones |

## How it works

1. **Preprocessing:** band-pass filtering, moving average and normalisation
2. **Detection:** peaks above a threshold are marked as possible coughs
3. **Extraction:** each detected cough is cut out as a time segment
4. **Evaluation:** predicted timestamps are matched against the labels, and precision, recall and F1 are computed
