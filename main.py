from flask import Flask, request, jsonify, render_template, Response
from transformers import pipeline
from pydub import AudioSegment
from io import BytesIO
import os
from huggingface_hub import InferenceClient
import json
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024


# Load your model
fast_model_name = "sgoedecke/wav2vec2_owl_classifier_sew_d" # faster, smaller
classifier = pipeline("audio-classification", model=fast_model_name)

# hf_client = InferenceClient(model=model_name)
# classifier = hf_client.audio_classification

def convert_audio_to_wav(audio_bytes):
    # Load audio file from bytes and convert to WAV with 16kHz
    audio = AudioSegment.from_file(BytesIO(audio_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1)
    return audio

def chunk_audio(audio, chunk_length_ms=5000):
    # Chunk audio into 5-second blocks
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

@app.route('/', methods=['GET'])
def home():
      return render_template('index.html')

@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    def generate_predictions_batched(audio_bytes):
        audio = convert_audio_to_wav(audio_bytes)
        samples = np.array(audio.get_array_of_samples())
        samples = samples.astype(np.float32) / 2**15
        num_elements_to_keep = len(samples) - (len(samples) % 80000)  # 5-second chunks
        samples = samples[:num_elements_to_keep]
        samples = samples.reshape(-1, 80000)  # Reshape samples into 5-second chunks
        batch_size = 30
        total_batches = len(samples) // batch_size + (1 if len(samples) % batch_size else 0)  # Calculate total number of batches

        for batch_index in range(total_batches):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            inputs = processor(samples[start_index:end_index], sampling_rate=16000, return_tensors="pt", padding=True)

            with torch.no_grad():  # Skip calculating gradients in the forward pass
                logits = model(inputs.input_values).logits
                for i, logit in enumerate(logits):
                    label = "owl" if logit[0] > logit[1] else "not_owl"
                    start_time = i * 5
                    end_time = (i + 1) * 5
                    # Yield predictions for streaming
                    if logit[0] > logit[1]:
                        yield json.dumps({
                            "detected": True,
                            "start_time": start_time,
                            "end_time": end_time,
                            "chunk_count": total_batches * 6,
                        }) + "\n\n"  # Adding \n\n for easier parsing and to distinguish between messages
                    else:
                        yield json.dumps({
                            "detected": False,
                            "start_time": start_time,
                            "end_time": end_time,
                            "chunk_count": total_batches * 6,
                        }) + "\n\n"

    # Stream response back to the client
    resp = Response(generate_predictions_batched(request.files['file'].read()), mimetype='text/event-stream')
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)