from flask import Flask, request, jsonify, render_template, Response
from transformers import pipeline
from pydub import AudioSegment
from io import BytesIO
import os
from huggingface_hub import InferenceClient
import json
import numpy as np
from transformers import AutoModelForAudioClassification, Wav2Vec2Processor
import torch
import librosa
import tempfile


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024


# Load your model
model_name = "sgoedecke/wav2vec2_owl_classifier_sew_d" # faster, smaller
model = AutoModelForAudioClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def convert_audio_to_wav(audio_bytes):
    # Load audio file from bytes and convert to WAV with 16kHz
    audio = AudioSegment.from_file(BytesIO(audio_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1)
    return audio

def convert_audio_to_wav_librosa(audio_bytes, target_sr=16000):
    # Create a temporary file to write the audio bytes to
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.seek(0)  # Go back to the start of the file
        audio, sr = librosa.load(tmp_file.name, sr=target_sr)
    return audio

@app.route('/', methods=['GET'])
def home():
      return render_template('index.html')


@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    print("Beginning request...")
      # This is a file-like object.
    # with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as tmp:
    #     file = request.files['file'] 
    #     file.save(tmp.name)  # Save the uploaded file's data to the temporary file
    #     tmp.flush()  # Ensure all data is written to disk
    #     print("Saved to tmp file...")
    #     audio, sr = librosa.load(tmp.name, sr=16000, mono=True, dtype=np.float32)  # Load the audio data with librosa

    file = request.files['file'] 
    audio_segment = AudioSegment.from_file(file)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

    print("Loaded audio")
    def generate_predictions_batched(audio):
        print("Trimming...")
        num_elements_to_keep = len(audio) - (len(audio) % 80000)  # Trim to nearest 5 seconds
        audio = audio[:num_elements_to_keep]
        print("Reshaping...")

        samples = audio.reshape(-1, 80000)  # Reshape samples into 5-second chunks
        batch_size = 10 #30 works fine
        total_batches = len(samples) // batch_size + (1 if len(samples) % batch_size else 0)  # Calculate total number of batches
        print("Batching...")
        for batch_index in range(total_batches):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            inputs = processor(samples[start_index:end_index], sampling_rate=16000, return_tensors="pt", padding=True)
            print("Done processing batch...")

            with torch.no_grad():  # Skip calculating gradients in the forward pass
                logits = model(inputs.input_values).logits
                print("Calculated logits...")
                print(logits)
                for i, logit in enumerate(logits):
                    label = "owl" if logit[0] > logit[1] else "not_owl"
                    start_time = (batch_index * batch_size + i) * 5  # Adjusted start time calculation
                    end_time = start_time + 5
                    # Yield predictions for streaming
                    yield json.dumps({
                        "detected": label == "owl",
                        "start_time": start_time,
                        "end_time": end_time,
                        "chunk_count": len(samples),
                    }) + "\n\n"

    # Stream response back to the client
    resp = Response(generate_predictions_batched(audio), mimetype='text/event-stream')
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)