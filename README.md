# Cough Detection Project: An Exploration in Audio Signal Processing
Welcome to the Cough Detection Project repository, an in-depth exploration into the domains of audio signal processing and machine learning. This project documents the development and implementation of algorithms dedicated to detecting cough timestamps and counts within audio signals. Below is an overview of the project's key components and milestones:

## Objective
The primary objective of this project is to develop accurate and robust algorithms for detecting cough instances in audio recordings. This involves both time-based and frequency-based analysis of audio signals to identify distinct cough patterns amidst background noise.

## Dataset Creation
The project initiated with the creation of a curated dataset comprising recordings of coughs and various ambient sounds. Each recording was carefully labeled and prepared to serve as training and testing data for the detection algorithms.

## Methodology
### Time-Domain Analysis
- **Signal Preprocessing**: Raw audio signals underwent preprocessing techniques such as time binning, moving average computation, and peak detection to enhance signal clarity and isolate cough instances.
- **Algorithm Developmen**t: Custom algorithms were designed to detect cough timestamps and counts based on time-domain signal characteristics. Special attention was paid to eliminating false positives and optimizing algorithm performance.

### Frequency-Domain Analysis
- **Spectrogram Analysis**: Frequency-based analysis involved the generation and analysis of spectrograms to identify distinct frequency patterns associated with coughs.
- **Filtering Techniques**: Various filtering techniques, including low-pass and high-pass filters, were applied to enhance signal clarity and isolate cough-related frequency components.

### Feature Extraction
- **Feature Selection**: Features such as percentile frequency values, Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroids, and energy were extracted to characterize cough instances.
- **Model Training**: Machine learning models including K-Nearest Neighbors, Support Vector Machines, Random Forests, and neural networks were trained using the extracted features to classify cough instances.

### Model Optimization
- **Hyperparameter Tuning**: Grid search and cross-validation techniques were employed to optimize model performance by fine-tuning model parameters.
- **Data Augmentation**: Noise addition technique was applied to augment the training dataset and improve model generalization.

## Conclusion
Despite the conclusion of the internship, the project continues to evolve. Ongoing efforts include refining detection algorithms, exploring new feature extraction methods, and enhancing model performance. Additionally, emphasis is placed on the integration of neural networks for improved accuracy and the classification of dry and wet coughs. 
