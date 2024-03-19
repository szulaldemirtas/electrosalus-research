# Cough Extraction in Time Domain and Frequency Domain

## Overview
This folder contains notebooks for extracting cough signals from audio recordings. The extraction process is performed in both the time domain and the frequency domain, offering different perspectives on the analysis.



## Contents
- **cough-extraction-single-recordings-time-domain**: This notebook analyzes the single cough audio, and extract the cough parts from the audio.
                                                  The analysis is based on time domain.
                                                  
- **cough-extraction-single-recordings-frequency-domain**: This notebook analyzes the single cough audio, and extract the cough parts from the audio. The                                                         analysis is based on frequency domain.
  
- **cough-extraction-multiple-recordings-time-domain**: This notebook extends the analysis to handle multiple cough audio recordings present in a                                                              given folder. It iterates through all the cough audio files in the specified directory,                                                                performing analysis and extraction in the time domain.
  
- **cough-extraction-multiple-recordings-frequency-domain**: This notebook extends the analysis to handle multiple cough audio recordings present in a                                                              given folder. It iterates through all the cough audio files in the specified directory,                                                                performing analysis and extraction in the time domain.

## Working Principle

The codes in the notebooks follows a multi-step process for cough detection and evaluation:

1. **Preprocessing**: Audio data is preprocessed using bandpass filtering, moving average computation, and normalization. This prepares the data for cough event detection.

2. **Cough Detection**: Cough events are detected in the preprocessed data using threshold-based peak detection techniques. Peaks exceeding a certain threshold are identified as potential cough events.

3. **Cough Extraction**: Detected cough events are extracted from the audio recordings as cough segments. These segments represent the time intervals during which coughs occur in the audio.

4. **Evaluation**: The performance of cough detection models is evaluated by comparing predicted cough timestamps with actual timestamps of cough events. Precision, recall, and F1-score metrics are computed to assess the model's performance.
