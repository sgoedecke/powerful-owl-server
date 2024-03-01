from flask import Flask, request, jsonify, render_template, Response
from transformers import pipeline
from pydub import AudioSegment
from io import BytesIO
import os
from huggingface_hub import InferenceClient
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024


# Load your model
# model_name = "sgoedecke/wav2vec2_owl_classifier_v3"
model_name = "sgoedecke/wav2vec2_owl_classifier_sew_d" # faster, smaller
classifier = pipeline("audio-classification", model=model_name)

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
    def generate_predictions(audio_bytes):
        print("Starting inference...")
        audio = convert_audio_to_wav(audio_bytes)
        chunks = chunk_audio(audio)

        chunk_length_seconds = 5  # Duration of each audio chunk

        print("Starting inference on " + str(len(chunks)) + "...")
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

            # Yield predictions for streaming
            if isinstance(chunk_prediction, list) and chunk_prediction[0]['label'] == 'owl':
                yield json.dumps({
                    "detected": True,
                    "start_time": start_time,
                    "end_time": end_time,
                    "chunk_count": len(chunks),
                }) + "\n\n"  # Adding \n\n for easier parsing and to distinguish between messages
            else:
                yield json.dumps({
                    "detected": False,
                    "start_time": start_time,
                    "end_time": end_time,
                    "chunk_count": len(chunks),
                }) + "\n\n"

    # Stream response back to the client
    resp = Response(generate_predictions(request.files['file'].read()), mimetype='text/event-stream')
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)