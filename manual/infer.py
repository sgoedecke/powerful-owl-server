from pydub import AudioSegment
import os
import numpy as np
from transformers import AutoModelForAudioClassification, Wav2Vec2Processor
import torch
import time
import sys

if len(sys.argv) < 2:
    print("Usage: python infer.py <paths_to_audio_files>")
    sys.exit(1)

def list_files(paths):
    all_files = []
    for path in paths:
        if os.path.isfile(path):
            all_files.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    all_files.append(os.path.join(root, file))
        else:
            print(f"Warning: '{path}' is neither a file nor a directory.")
    return all_files

files = list_files(sys.argv[1:] )
print "Processing files: ", files

# Load your model
print("Loading classifier model...")
model_name = "sgoedecke/wav2vec2_owl_classifier_sew_d" # faster, smaller
model = AutoModelForAudioClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def look_for_owls(file):
    print(f"Segmenting {file} audio segment into 5 second chunks...")
    audio_segment = AudioSegment.from_file(file)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

    num_elements_to_keep = len(audio) - (len(audio) % 80000)  # Trim to nearest 5 seconds
    audio = audio[:num_elements_to_keep]
    samples = audio.reshape(-1, 80000)  # Reshape samples into 5-second chunks
    batch_size = 30
    total_batches = len(samples) // batch_size + (1 if len(samples) % batch_size else 0)  # Calculate total number of batches

    print(f"Segmented into {len(samples)} chunks in {total_batches} batches...")
    results = []

    for batch_index in range(total_batches):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        inputs = processor(samples[start_index:end_index], sampling_rate=16000, return_tensors="pt", padding=True)
        print(f"Done preprocessing batch {batch_index}, calculating logits...")

        with torch.no_grad():  # Skip calculating gradients in the forward pass
            logits = model(inputs.input_values).logits
            print("Calculated logits")
            print(logits)
            for i, logit in enumerate(logits):
                label = "owl" if logit[0] > logit[1] else "not_owl"
                start_time = (batch_index * batch_size + i) * 5  # Adjusted start time calculation
                end_time = start_time + 5
                # Yield predictions for streaming
                if label == "owl":
                    results.append((file, start_time, end_time))
                    print(f"Owl detected at {start_time} - {end_time} seconds!")
    
    print(f"Done processing {file}!")
    return results

results = []
for file in files:
    results.append(look_for_owls(file))

print("Finished processing all files! Owls found:", results)