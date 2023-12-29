# README for High-Pitch Environmental Sound Detection Project

## Project Overview

This project focuses on detecting and classifying high-pitched environmental sounds using a novel machine learning approach. The methodology includes Adaptive Variational Modal Decomposition (AVMD) and Pseudo Wigner-Ville Distribution (PWVD) for feature extraction, followed by classification using a Convolutional Neural Network (CNN).


## Contents of Submission

1. `AVMD_PWVD_Processing.py` - Python code for AVMD-PWVD image processing.
2. `cnn.py` - Python code for the CNN model used for classification.
3. `cnndata/` - Directory containing processed data after AVMD-PWVD processing.
4. `entropy_features.json` - JSON file with entropy features, located in the `cnndata/` directory.


## Preprocessing Notes

1. **Data Preprocessing Steps Not Included**: The initial data preprocessing involves adding noise to audio files using Audacity software. This step is manually done and is not included in the code due to its software-specific nature.

2. **AVMD-PWVD Processing Time**: The conversion of noisy audio files to AVMD-PWVD processed images is computationally intensive. For reference, processing 120 files took approximately 18 hours on my setup. 

    ### System Specifications Used for Processing:
    - **CPU**: AMD Ryzen 9 5900HX with Radeon Graphics
      - Cores: 8
      - Threads per Core: 2
      - Max Frequency: 3300 MHz
    - **Memory**: 30 GB RAM
    - **Graphics**:
      - NVIDIA GeForce RTX 3070 Mobile / Max-Q
      - AMD Cezanne
    - **Operating System**: Ubuntu 22.04.3 LTS
    - **Virtualization**: AMD-V
    - **TensorFlow with CUDA**: Enabled for GPU accelerated computation

   Please consider this duration and the specifications when planning to replicate the experiment, as performance may vary based on system configuration.

3. **Data Provided Post AVMD-PWVD Process**: To facilitate easier replication of the experiment, the data provided in the `cnndata/` directory is already processed through the AVMD-PWVD method. This includes the time-frequency images and the associated entropy features.


## Replication Instructions

1. **Path Configuration in `cnn.py`**: Ensure that the file paths in the `cnn.py` script correctly point to the `cnndata/` directory and the `entropy_features.json` file. Incorrect paths will lead to errors in data loading and model training.

2. **Running the CNN Model**: Execute the `cnn.py` script to train the CNN model on the provided data. The script will also evaluate the model's performance and display the results.

3. **Model Training and Evaluation**: The training of the CNN model on my system using TensorFlow with CUDA for GPU acceleration took approximately 59.22 seconds for 100 epochs. This relatively short duration is attributed to the utilization of GPU resources, enhancing the computational efficiency of the training process. Please note that the duration of the training process can significantly vary depending on the system's computational capabilities and whether GPU acceleration is available and utilized.


## Project Aim and Scope

This project aims to demonstrate the effectiveness of combining AVMD-PWVD with deep learning for environmental sound analysis, particularly in detecting and classifying high-pitched sounds. The methodology and results have potential applications in areas such as auditory surveillance and ecological monitoring.


## Contact Information

For any queries or further clarification regarding the project setup or replication process, please feel free to contact Name: Tulasi Sainath Polisetty; EMAIL: tpoliset@asu.edu.
