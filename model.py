from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import torchaudio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the model and processor
model_name = "circulus/canvers-audio-caption-v1"
try:
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model or processor: {e}")
    raise

# Load YAMNet model for background sound classification
def load_yamnet():
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yamnet_model

# Preprocess audio for both models
def preprocess_audio(audio_path):
    try:
        audio, original_sampling_rate = torchaudio.load(audio_path, normalize=True)
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=16000)
        audio_resampled = resampler(audio)
        if audio_resampled.shape[0] > 1:
            audio_resampled = audio_resampled.mean(dim=0)  # Convert to mono
        return audio_resampled.numpy(), 16000
    except Exception as e:
        print(f"Error loading or processing audio file: {e}")
        raise

# Split audio into chunks
def split_audio(audio_np, chunk_length_sec, sampling_rate):
    chunk_length_samples = int(chunk_length_sec * sampling_rate)
    num_chunks = len(audio_np) // chunk_length_samples
    return np.array_split(audio_np[:num_chunks * chunk_length_samples], num_chunks)

# Load and preprocess audio
input_audio = "/content/mom.m4a"
audio_np, sampling_rate = preprocess_audio(input_audio)

# Determine if audio is long enough to split
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
combined_results = set()

# Process each chunk with the speech-to-text model
for i, chunk in enumerate(chunks):
    # print(f"Processing chunk {i + 1}/{len(chunks)}...")
    try:
        inputs = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(input_features=inputs["input_features"])
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        combined_captions.add(caption)
    except Exception as e:
        print(f"Error generating or decoding caption for chunk {i + 1}: {e}")

# Process each chunk with YAMNet
yamnet_model = load_yamnet()

for i, chunk in enumerate(chunks):
    # print(f"Classifying chunk {i + 1}/{len(chunks)}...")
    try:
        # Convert chunk to 1D tensor for YAMNet
        audio_tf = tf.convert_to_tensor(chunk, dtype=tf.float32)
        audio_tf = tf.reshape(audio_tf, [-1])  # Reshape to 1D tensor

        # Run the model on the audio input
        scores, embeddings, spectrogram = yamnet_model(audio_tf)
        scores = scores.numpy()

        # Process YAMNet scores
        top_scores_indices = np.argsort(scores[0])[-5:]  # Get top 5 scores
        combined_results.add(", ".join([f'Class_{idx}' for idx in top_scores_indices]))
    except Exception as e:
        print(f"Error processing audio with YAMNet for chunk {i + 1}: {e}")

# Combine and print the final results
final_caption = " | ".join(sorted(combined_captions))
final_results = " | ".join(sorted(combined_results))
print(f"Combined Caption: {final_caption}")
#print(f"Combined Classification: {final_results}")
