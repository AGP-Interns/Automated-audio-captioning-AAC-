import gradio as gr
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import torchaudio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from collections import Counter

# Load the model and processor
model_name = "circulus/canvers-audio-caption-v1"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)

# Load YAMNet model for background sound classification
def load_yamnet():
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yamnet_model

yamnet_model = load_yamnet()

# Load class names from CSV
class_map_df = pd.read_csv('/content/yamnet_class_map.csv')
class_names = class_map_df['display_name'].tolist()

# Preprocess audio for both models
def preprocess_audio(audio_path):
    audio, original_sampling_rate = torchaudio.load(audio_path, normalize=True)
    resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=16000)
    audio_resampled = resampler(audio)
    if audio_resampled.shape[0] > 1:
        audio_resampled = audio_resampled.mean(dim=0)  # Convert to mono
    return audio_resampled.numpy(), 16000

# Split audio into chunks for YAMNet
def split_audio_yamnet(audio_np, sampling_rate, frame_length=1024, frame_step=512):
    audio_tf = tf.convert_to_tensor(audio_np, dtype=tf.float32)
    framed_audio = tf.signal.frame(audio_tf, frame_length=frame_length, frame_step=frame_step)
    return framed_audio

# Split audio into chunks for speech-to-text model
def split_audio(audio_np, chunk_length_sec, sampling_rate):
    chunk_length_samples = int(chunk_length_sec * sampling_rate)
    num_chunks = len(audio_np) // chunk_length_samples
    return np.array_split(audio_np[:num_chunks * chunk_length_samples], num_chunks)

# Main processing function
def process_audio(input_audio):
    audio_np, sampling_rate = preprocess_audio(input_audio)
    audio_length_sec = len(audio_np) / sampling_rate
    chunk_length_sec = 10
    if audio_length_sec > chunk_length_sec:
        # Split audio into chunks of 10 seconds
        chunks = split_audio(audio_np, chunk_length_sec=chunk_length_sec, sampling_rate=sampling_rate)
    else:
        # Use the entire audio as one chunk
        chunks = [audio_np]

    # Initialize variables to store combined results
    combined_captions = set()
    all_results = []

    # Process each chunk with the speech-to-text model
    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(input_features=inputs["input_features"])
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        combined_captions.add(caption)

    # Process each chunk with YAMNet
    yamnet_chunks = split_audio_yamnet(audio_np, sampling_rate)
    for chunk in yamnet_chunks:
        # Run the model on the audio input
        scores, embeddings, spectrogram = yamnet_model(chunk)
        scores = scores.numpy()

        # Process YAMNet scores
        top_scores_indices = np.argsort(scores[0])[-5:]  # Get top 5 scores
        top_classes = [class_names[idx] for idx in top_scores_indices]
        all_results.extend(top_classes)

    # Find the most frequent classes
    class_counts = Counter(all_results)
    # Get the top 10 most common classes
    most_common_classes = class_counts.most_common(10)

    # Prepare the final results
    final_caption = " | ".join(sorted(combined_captions))
    final_results = ", ".join([f"{cls} ({count})" for cls, count in most_common_classes])
    return final_caption, final_results

# Create a Gradio interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Combined Caption"),
        gr.Textbox(label="Top 10 Most Frequent Classifications")
    ],
    title="Audio Captioning and Classification",
    description="Record an audio file or upload one to get captions and background sound classification."
)

# Launch the interface
iface.launch()
