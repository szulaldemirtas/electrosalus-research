# Datasets

## Overview
This folder contains datasets for coughs and other noises including laughing, clapping, knocking, speaking, and crowded noise. 

- The **audio-recordings** directory comprises recordings from three distinct sources: recordings obtained firsthand, sourced from the internet, and data from the COUGHVID dataset.- The audio-timestamps folder includes timestamps of each file. Timestamp files are in .txt format, and each line contains two values. The first value is starting timepoint in recording and the second value is finishing timepoint.

* Within the **audio-timestamps** directory, you'll find timestamp files in .txt format, detailing the start and end timepoints for each audio recording. Each line within these files contains two values: the starting timepoint and the corresponding finishing timepoint.


## Contents

- **audio-recordings/cough-recordings**: This directory contains cough recordings gathered through collaborative efforts, including contributions from myself, friends, and colleagues.
- **audio-recordings/cough-recordings-internet**: This directory contains cough recordings obtained from online sources.  
- **audio-recordings/coughvid-dry-coughs**: This directory contains dry cough recordings sourced from the COUGHVID dataset, labeled by experts.
- **audio-recordings/coughvid-wet-coughs**: This directory contains wet cough recordings sourced from the COUGHVID dataset, labeled by experts.
- **audio-recordings/other-recordings**: This directory contains various recordings such as laughter, clapping, knocking, speaking, and crowded noise. 
- **audio-recordings/other-recordings-internet**: This directory contains various recordings such as laughter, clapping, knocking, speaking, and crowded noise. The recordings are collected from the internet.
- **audio-timestamps**: This directory contains timestamps for each recording. For coughs, timestamps indicate the start and end times of cough events. Timestamps for other noises are also included, although they are extracted more randomly to improve classification performance.
- **cough-recordings.csv**: This file provides dataset details curated by the researcher. It includes file name, age, gender, health status, cough count, smoker status, asthma presence, and known lung disease information.
