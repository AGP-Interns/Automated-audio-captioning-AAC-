#  Audio Captioning

## Overview

This script processes audio files to generate textual captions and classify background sounds. It utilizes two models:

1. **Circulus Canvers Audio Caption Model**: For generating textual descriptions of the audio content.
2. **YAMNet**: For classifying background sounds in the audio.

## Components

### 1. Model and Processor Loading

- **Circulus Canvers Audio Caption Model**:
  - The script loads this model and its processor using the `transformers` library. This model is designed to generate captions for audio inputs.

- **YAMNet**:
  - This model is loaded from TensorFlow Hub. It is used to classify background sounds in the audio.

### 2. Audio Preprocessing

- **Audio Loading and Resampling**:
  - The audio is loaded from a specified file path using `torchaudio`. It is then resampled to a standard rate of 16kHz and converted to mono if necessary.

### 3. Audio Chunking

- **Splitting Long Audio**:
  - If the audio file is longer than 10 seconds, it is split into chunks of 10 seconds each. This ensures that each chunk can be processed efficiently by the models.

### 4. Caption Generation

- **Processing Chunks**:
  - Each audio chunk is processed by the Circulus Canvers model to generate a textual caption. The results from all chunks are combined to form the final caption.

### 5. Sound Classification

- **YAMNet Classification**:
  - Each audio chunk is also classified using the YAMNet model. The top 5 sound classes are identified and combined across all chunks.

### 6. Output

- **Final Results**:
  - The script prints the combined captions and classification results. This provides an overview of the audio content and background sounds.

## Usage

1. **Update Audio Path**:
   Replace the `input_audio` variable with the path to your audio file.

2. **Run the Script**:
   Execute the script to process the audio file and obtain the results.




## **How to Use**

### **Download Models:**
        *Use the instructions above to load the models.*
        *Ensure that yamnet_class_map.csv is placed in the correct directory*

 **Run the Code:**
        *Set the path to your audio file in the input_audio variable.*
        *Execute the script to process the audio file and obtain the results.*

### **Troubleshooting**

**Model Loading Errors:**
        *Ensure you have an active internet connection and access to the online models.*
        *Check for compatibility between library versions.*

    **Audio Processing Errors:**
       *Verify that the audio file is correct and in a supported format.*
       *Ensure that all required libraries are installed properly.*

*With this information, you should be able to download the models and use them in your project. If you have any questions or issues, feel free to ask for help.&

