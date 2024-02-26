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
    def generate_predictions():
        audio_bytes = file.read()
        audio = convert_audio_to_wav(audio_bytes)
        chunks = chunk_audio(audio)

        chunk_length_seconds = 5  # Duration of each audio chunk

        for i, chunk in enumerate(chunks):
            # Export chunk to bytes
            buffer = BytesIO()
            chunk.export(buffer, format="wav")
            buffer.seek(0)

            # Perform inference
            chunk_prediction = classifier(buffer.read(), top_k=1)
            buffer.close()  # Close buffer after reading

            start_time = i * chunk_length_seconds
            end_time = start_time + chunk_length_seconds

            # Yield predictions for streaming
            if isinstance(chunk_prediction, list) and chunk_prediction[0]['label'] == 'owl':
                yield f"<p style='color: green;'>Owl sound detected from {start_time} to {end_time} seconds.</p>"
            else:
                yield f"<p>No owl sound detected from {start_time} to {end_time} seconds.</p>"

    # Stream response back to the client
    return Response(generate_predictions(), content_type='text/html')

def generate_batched_predictions(chunks):
    # Prepare all chunks for inference
    buffers = [BytesIO() for _ in chunks]
    for buffer, chunk in zip(buffers, chunks):
        chunk.export(buffer, format="wav")
        buffer.seek(0)
    
    # Perform inference in batches
    predictions = classifier([buffer.read() for buffer in buffers], top_k=1)
    
    # Don't forget to close the buffers
    for buffer in buffers:
        buffer.close()
    
    return predictions

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    # Your existing file handling code here

    audio_bytes = file.read()
    audio = convert_audio_to_wav(audio_bytes)
    chunks = chunk_audio(audio)

    # Generate predictions for all chunks at once
    predictions = generate_batched_predictions(chunks)
    
    chunk_length_seconds = 5  # Duration of each audio chunk
    resp = "Here's the 🦉 I found:\n"
    for i, result in enumerate(predictions):
        start_time = i * chunk_length_seconds
        end_time = start_time + chunk_length_seconds
        
        if isinstance(result, list) and result[0]['label'] == 'owl':
            resp += f"<p>Owl sound detected from {start_time} to {end_time} seconds.</p>"
    
    return Response(resp, content_type='text/plain')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)