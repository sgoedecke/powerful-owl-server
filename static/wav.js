async function processWavFile(file) {
    const header = await readWavHeader(file);
    const chunkDurationMs = 20 * 1000; // 20 seconds in milliseconds
    const bytesPerSample = header.bitDepth / 8;
    const frameSize = header.channels * bytesPerSample; // Size of each frame in bytes
    const chunkSize = header.sampleRate * chunkDurationMs / 1000 * frameSize; // Size of each chunk in bytes

    let offset = 44; // Skip the original WAV header
    let chunks = []
    while (offset < file.size) {
        let end = Math.min(file.size, offset + chunkSize);
        let chunkData = file.slice(offset, end);
        let chunkHeader = createWavHeader(header, chunkData.size);
        let chunkBlob = new Blob([chunkHeader, chunkData], {type: 'audio/wav'});
        chunks.push(chunkBlob)
        offset += chunkSize;
    }
    return chunks
}

async function readWavHeader(file) {
    const reader = new FileReader();
    const headerBlob = file.slice(0, 44); // WAV header is the first 44 bytes
    reader.readAsArrayBuffer(headerBlob);

    return new Promise((resolve, reject) => {
        reader.onload = () => {
            const buffer = reader.result;
            const view = new DataView(buffer);
            resolve({
                // Assuming little-endian
                channels: view.getUint16(22, true),
                sampleRate: view.getUint32(24, true),
                bitDepth: view.getUint16(34, true),
            });
        };
        reader.onerror = () => {
            reject(new Error('Could not read file'));
        };
    });
}

function createWavHeader(headerInfo, dataSize) {
    const buffer = new ArrayBuffer(44);
    const view = new DataView(buffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true); // File size
    writeString(view, 8, 'WAVE');
    // fmt subchunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true); // AudioFormat (PCM = 1)
    view.setUint16(22, headerInfo.channels, true);
    view.setUint32(24, headerInfo.sampleRate, true);
    view.setUint32(28, headerInfo.sampleRate * headerInfo.channels * (headerInfo.bitDepth / 8), true); // ByteRate
    view.setUint16(32, headerInfo.channels * (headerInfo.bitDepth / 8), true); // BlockAlign
    view.setUint16(34, headerInfo.bitDepth, true);
    // data subchunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true); // Subchunk2Size (data size)

    return buffer;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

