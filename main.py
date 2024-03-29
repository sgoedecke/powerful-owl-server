from flask import Flask, request, jsonify, render_template, Response
from pydub import AudioSegment
from io import BytesIO
import os
import json
import numpy as np
from transformers import AutoModelForAudioClassification, Wav2Vec2Processor
import torch
import time


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024


# Load your model
model_name = "sgoedecke/wav2vec2_owl_classifier_sew_d" # faster, smaller
model = AutoModelForAudioClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

@app.route('/', methods=['GET'])
def home():
      return render_template('index.html')


@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    print("Getting file handle...")
    ts_start = time.time()
    file = request.files['file']  # 15-17 seconds for 30mb. Slow because https://github.com/pallets/werkzeug/issues/875#issuecomment-309779076 ?
    # file = request.stream.read() # also like 20s. maybe it's just the upload speed
    ts_end = time.time()
    print(f"Time taken: {ts_end - ts_start}")

    print("Grabbing audio segment...")
    audio_segment = AudioSegment.from_file(file)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

    def generate_predictions_batched(audio):
        num_elements_to_keep = len(audio) - (len(audio) % 80000)  # Trim to nearest 5 seconds
        audio = audio[:num_elements_to_keep]
        samples = audio.reshape(-1, 80000)  # Reshape samples into 5-second chunks
        batch_size = 10 #30 works fine
        total_batches = len(samples) // batch_size + (1 if len(samples) % batch_size else 0)  # Calculate total number of batches

        for batch_index in range(total_batches):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            inputs = processor(samples[start_index:end_index], sampling_rate=16000, return_tensors="pt", padding=True)
            print("Done processing batch, calculating logits...")

            with torch.no_grad():  # Skip calculating gradients in the forward pass
                logits = model(inputs.input_values).logits
                print("Calculated logits!")
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