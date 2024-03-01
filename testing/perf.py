from transformers import pipeline
from pydub import AudioSegment
from io import BytesIO
import os
from huggingface_hub import InferenceClient
import json
import time


# Load your model
# model_name = "sgoedecke/wav2vec2_owl_classifier_v3"
model_name = "sgoedecke/wav2vec2_owl_classifier_sew_d" # faster, smaller

classifier = pipeline("audio-classification", model=model_name)

# hf_client = InferenceClient(model=model_name)
# classifier = hf_client.audio_classification

def convert_audio_to_wav(path):
    # Load audio file from bytes and convert to WAV with 16kHz
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    return audio

def chunk_audio(audio, chunk_length_ms=5000):
    # Chunk audio into 5-second blocks
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

#21s
#8.8s for sew-d
def generate_predictions(audio_bytes):
    ts_start = time.time()
    audio = convert_audio_to_wav(audio_bytes)
    chunks = chunk_audio(audio)
    chunk_length_seconds = 5  # Duration of each audio chunk
    print("Starting inference on " + str(len(chunks)) + "...")
    res = []
    for i, chunk in enumerate(chunks):
        # Export chunk to bytes
        buffer = BytesIO()
        chunk.export(buffer, format="wav")
        buffer.seek(0)
        # Perform inference
        chunk_prediction = classifier(buffer.read())#, top_k=1)
        buffer.close()  # Close buffer after reading
        start_time = i * chunk_length_seconds
        end_time = start_time + chunk_length_seconds
        print(f"Predicted {chunk_prediction} from {start_time} to {end_time} seconds.")
        res.append(chunk_prediction)
    ts_end = time.time()
    print(f"Time taken: {ts_end - ts_start}")
    return res


from concurrent.futures import ThreadPoolExecutor, as_completed

def infer_chunk(chunk, chunk_length_seconds, i):
    buffer = BytesIO()
    chunk.export(buffer, format="wav")
    buffer.seek(0)
    chunk_prediction = classifier(buffer.read())
    buffer.close()
    return chunk_prediction, i * chunk_length_seconds, (i + 1) * chunk_length_seconds

#20.5s at n=5, 18/19 at n=2
def generate_predictions_parallel(audio_bytes, n_cores=2):
    ts_start = time.time()
    audio = convert_audio_to_wav(audio_bytes)
    chunks = chunk_audio(audio)
    print(f"Starting parallel inference on {len(chunks)} chunks...")
    res = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(infer_chunk, chunk, n_cores, i) for i, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            chunk_prediction, start_time, end_time = future.result()
            print(f"Predicted {chunk_prediction} from {start_time} to {end_time} seconds.")
            res.append(chunk_prediction)
    ts_end = time.time()
    print(f"Time taken: {ts_end - ts_start}")
    return res


from transformers import AutoModelForAudioClassification, Wav2Vec2Processor
model = AutoModelForAudioClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

import numpy as np
import torch

#21s
#3.7s with sew-d!?!?
def generate_predictions_batched(audio_bytes):
    ts_start = time.time()
    audio = convert_audio_to_wav(audio_bytes)
    samples = np.array(audio.get_array_of_samples())
    samples = samples.astype(np.float32) / 2**15
    num_elements_to_keep = len(samples) - (len(samples) % 80000) # 5 second chunks
    samples = samples[:num_elements_to_keep]
    samples = samples.reshape(-1, 80000)
    inputs = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
    # 10 OK, 15 not OK
    with torch.no_grad(): # skip calculating grads in forwar dpass
        logits = model(inputs.input_values[:30]).logits
        # model(inputs.input_values[10:]).logits
    ts_end = time.time()
    print(f"Time taken: {ts_end - ts_start}")
    for i in range(len(logits)):
        label = ""
        if logits[i][0] > logits[i][1]:
            label = "owl"
        else:
            label = "not_owl"
        print(f"Predicted {label} from {i * 5} to {(i + 1) * 5} seconds.")
    return logits

#37s
#12s with sew-d
def generate_predictions_unchunked(audio_bytes):
    ts_start = time.time()
    audio = convert_audio_to_wav(audio_bytes)
    chunk_length_seconds = 60  # Duration of each audio chunk, 120 kills process
    chunks = chunk_audio(audio, chunk_length_ms=chunk_length_seconds*1000)
    print("Starting inference on " + str(len(chunks)) + "...")
    res = []
    for i, chunk in enumerate(chunks):
        # Export chunk to bytes
        buffer = BytesIO()
        chunk.export(buffer, format="wav")
        buffer.seek(0)
        # Perform inference
        chunk_prediction = classifier(buffer.read())#, top_k=1)
        buffer.close()  # Close buffer after reading
        start_time = i * chunk_length_seconds
        end_time = start_time + chunk_length_seconds
        print(f"Predicted {chunk_prediction} from {start_time} to {end_time} seconds.")
        res.append(chunk_prediction)
    ts_end = time.time()
    print(f"Time taken: {ts_end - ts_start}")
    return res

audio_bytes = '../1owl.m4a'
generate_predictions_batched('../1owl.m4a')
