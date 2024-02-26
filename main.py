from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from pydub import AudioSegment
from io import BytesIO
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024


# Load your model
model_name = "sgoedecke/wav2vec2_owl_classifier_v3"
classifier = pipeline("audio-classification", model=model_name)

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

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            audio_bytes = file.read()
            audio = convert_audio_to_wav(audio_bytes)
            chunks = chunk_audio(audio)

            predictions = []
            for chunk in chunks:
                # Export chunk to bytes
                buffer = BytesIO()
                chunk.export(buffer, format="wav")
                buffer.seek(0)

                # Perform inference
                chunk_prediction = classifier(buffer.read(), top_k=1)
                predictions.append(chunk_prediction)

                # Reset buffer for next iteration
                buffer.close()
        chunk_length_seconds = 5  # Duration of each audio chunk


        resp = "Here's the 🦉 I found:"
        for i, result in enumerate(predictions):
            start_time = i * chunk_length_seconds
            end_time = start_time + chunk_length_seconds

            if isinstance(result, list) and result[0]['label'] == 'owl':
                resp = resp + f"Owl sound detected from {start_time} to {end_time} seconds.\n"

        return resp

    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)