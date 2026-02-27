/**
 * audio.js — MediaRecorder recording + WAV/audio file upload
 *
 * Provides:
 *  - startRecording() → streams to WaveformRenderer live
 *  - stopRecording()  → returns decoded Float32Array PCM
 *  - loadFile(file)   → returns decoded Float32Array PCM
 *  - loadUrl(url)     → fetch + decode → Float32Array PCM
 */

'use strict';

let _mediaRecorder = null;
let _chunks = [];
let _stream = null;
let _audioCtx = null;

function getAudioCtx() {
    if (!_audioCtx || _audioCtx.state === 'closed') {
        _audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    }
    return _audioCtx;
}

/**
 * Decode an ArrayBuffer to a mono 16kHz Float32Array.
 */
async function decodeToMono(arrayBuffer) {
    const ctx = getAudioCtx();
    const buf = await ctx.decodeAudioData(arrayBuffer);
    // Resample to 16000 if needed
    const targetSR = 16000;
    if (buf.sampleRate !== targetSR) {
        const offCtx = new OfflineAudioContext(1, Math.ceil(buf.duration * targetSR), targetSR);
        const src = offCtx.createBufferSource();
        src.buffer = buf;
        src.connect(offCtx.destination);
        src.start(0);
        const resampled = await offCtx.startRendering();
        return resampled.getChannelData(0);
    }
    // Already mono channel 0, or mix down stereo
    if (buf.numberOfChannels === 1) return buf.getChannelData(0);
    // Mix down stereo
    const L = buf.getChannelData(0);
    const R = buf.getChannelData(1);
    const mono = new Float32Array(L.length);
    for (let i = 0; i < L.length; i++) mono[i] = (L[i] + R[i]) / 2;
    return mono;
}

/**
 * Start microphone recording.
 * @param {function} onAnalyser — called with AnalyserNode for live waveform
 * @returns {Promise<void>}
 */
export async function startRecording(onAnalyser) {
    _stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    _chunks = [];

    // Live waveform analyser
    const ctx = getAudioCtx();
    const src = ctx.createMediaStreamSource(_stream);
    const analyser = ctx.createAnalyser();
    src.connect(analyser);
    if (onAnalyser) onAnalyser(analyser);

    // Use webm/opus if available, else default
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus' : '';
    _mediaRecorder = new MediaRecorder(_stream, mimeType ? { mimeType } : {});
    _mediaRecorder.ondataavailable = e => { if (e.data.size > 0) _chunks.push(e.data); };
    _mediaRecorder.start(100);
}

/**
 * Stop recording and return decoded PCM.
 * @returns {Promise<Float32Array>}
 */
export function stopRecording() {
    return new Promise((resolve, reject) => {
        if (!_mediaRecorder) { reject(new Error('Not recording')); return; }
        _mediaRecorder.onstop = async () => {
            const blob = new Blob(_chunks, { type: _mediaRecorder.mimeType || 'audio/webm' });
            const ab = await blob.arrayBuffer();
            try {
                const pcm = await decodeToMono(ab);
                resolve(pcm);
            } catch (e) { reject(e); }
            // Stop microphone tracks
            _stream?.getTracks().forEach(t => t.stop());
        };
        _mediaRecorder.stop();
    });
}

/**
 * Load from a File object (uploaded WAV, mp3, etc.)
 * @param {File} file
 * @returns {Promise<Float32Array>}
 */
export async function loadFile(file) {
    const ab = await file.arrayBuffer();
    return decodeToMono(ab);
}

/**
 * Load from a URL (sample WAVs bundled with the demo).
 * @param {string} url
 * @returns {Promise<Float32Array>}
 */
export async function loadUrl(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
    const ab = await res.arrayBuffer();
    return decodeToMono(ab);
}

/** Check if recording is in progress */
export function isRecording() {
    return _mediaRecorder?.state === 'recording';
}
