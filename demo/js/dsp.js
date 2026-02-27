/**
 * dsp.js — Pure-JS re-implementation of create_synthetic_temporal_graph()
 *
 * Parameters mirrored from Python:
 *   sr=16000, win_ms=50, hop_ms=100, n_mfcc=13
 *   Features per node: 13 (MFCC mean) + 13 (MFCC std) + 13 (delta) + 13 (delta2) = 52
 */

'use strict';

// ─── FFT (Cooley-Tukey, radix-2, in-place) ────────────────────────────────────
function fft(re, im) {
    const n = re.length;
    // Bit-reversal permutation
    let j = 0;
    for (let i = 1; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }
    // Butterfly
    for (let len = 2; len <= n; len <<= 1) {
        const ang = (2 * Math.PI) / len;
        const wRe = Math.cos(ang), wIm = -Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let curRe = 1, curIm = 0;
            for (let k = 0; k < len / 2; k++) {
                const uRe = re[i + k], uIm = im[i + k];
                const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
                const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
                re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
                re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm;
                const tmpRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = tmpRe;
            }
        }
    }
}

// ─── Next power of 2 ────────────────────────────────────────────────────────
function nextPow2(n) {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ─── Hz ↔ Mel conversions ──────────────────────────────────────────────────
const hzToMel = hz => 2595 * Math.log10(1 + hz / 700);
const melToHz = mel => 700 * (Math.pow(10, mel / 2595) - 1);

// ─── Build mel filterbank  (nFilt × nFft/2+1) ──────────────────────────────
function melFilterbank(sr, nFft, nFilt) {
    const fMin = 0, fMax = sr / 2;
    const melMin = hzToMel(fMin), melMax = hzToMel(fMax);
    const melPoints = Array.from({ length: nFilt + 2 }, (_, i) =>
        melMin + (i / (nFilt + 1)) * (melMax - melMin));
    const hzPoints = melPoints.map(melToHz);
    const bins = hzPoints.map(f => Math.floor((nFft + 1) * f / sr));

    const filters = [];
    for (let m = 1; m <= nFilt; m++) {
        const row = new Float32Array(Math.floor(nFft / 2) + 1);
        const left = bins[m - 1], center = bins[m], right = bins[m + 1];
        for (let k = left; k <= center; k++) {
            row[k] = (k - left) / Math.max(1, center - left);
        }
        for (let k = center; k <= right; k++) {
            row[k] = (right - k) / Math.max(1, right - center);
        }
        filters.push(row);
    }
    return filters; // nFilt arrays, each of length nFft/2+1
}

// ─── Hann window ────────────────────────────────────────────────────────────
function hannWindow(n) {
    // librosa.filters.get_window('hann', n, fftbins=True) matches periodic window
    return Float32Array.from({ length: n }, (_, i) =>
        0.5 * (1 - Math.cos((2 * Math.PI * i) / n)));
}

// ─── Compute power spectrum one frame ───────────────────────────────────────
function powerSpec(frame, nFft) {
    const N = nFft;
    const re = new Float32Array(N);
    const im = new Float32Array(N);
    const L = Math.min(frame.length, N);
    for (let i = 0; i < L; i++) re[i] = frame[i];
    fft(re, im);
    const half = Math.floor(N / 2) + 1;
    const pwr = new Float32Array(half);
    for (let i = 0; i < half; i++) {
        // Librosa power spectrogram is |STFT|^2, no 1/N scaling
        pwr[i] = (re[i] * re[i] + im[i] * im[i]);
    }
    return pwr;
}

// ─── DCT-II (for MFCC from log mel energies) ────────────────────────────────
function dct2(x, nCoeff) {
    const N = x.length;
    const out = new Float32Array(nCoeff);
    for (let k = 0; k < nCoeff; k++) {
        let sum = 0;
        for (let n = 0; n < N; n++) {
            sum += x[n] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
        }
        // Ortho-normalization (matches librosa.feature.mfcc default)
        const factor = k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
        out[k] = sum * factor;
    }
    return out;
}

// ─── Finite-difference delta (order 1 or 2) ─────────────────────────────────
function delta(matrix, order = 1) {
    // matrix: T x C
    const T = matrix.length, C = matrix[0].length;
    const out = Array.from({ length: T }, () => new Float32Array(C));
    const W = 4; // half-context width (Matches Librosa default width=9)
    for (let t = 0; t < T; t++) {
        for (let c = 0; c < C; c++) {
            let num = 0, denom = 0;
            for (let n = 1; n <= W; n++) {
                const tFwd = Math.min(t + n, T - 1);
                const tBwd = Math.max(t - n, 0);
                num += n * (matrix[tFwd][c] - matrix[tBwd][c]);
                denom += n * n;
            }
            out[t][c] = denom > 0 ? num / (2 * denom) : 0;
        }
    }
    if (order === 1) return out;
    return delta(out, 1); // delta²
}

// ─── Main: extract 52-dim features from PCM samples ─────────────────────────
/**
 * @param {Float32Array} pcm  — raw PCM float samples at `sr` Hz
 * @param {Object} opts
 * @returns {Array<Float32Array>}  T arrays of length 52
 */
export function extractFeatures(pcm, opts = {}) {
    const sr = opts.sr || 16000;
    const winMs = opts.winMs || 50;
    const hopMs = opts.hopMs || 100;
    const nMfcc = opts.nMfcc || 13;

    const winLen = Math.floor(sr * winMs / 1000);
    const hopLen = Math.floor(sr * hopMs / 1000);
    const nFft = nextPow2(winLen);          // e.g. 1024 for win=800
    const nFilt = 128;                        // Librosa default n_mels=128

    const window = hannWindow(winLen);
    const filters = melFilterbank(sr, nFft, nFilt);

    /**
     * Librosa center=True framing:
     *   The t-th frame is centered at t * hopLen.
     *   n_frames = floor(L / hopLen) + 1
     */
    const T = Math.floor(pcm.length / hopLen) + 1;
    if (T <= 0) throw new Error('Audio too short');

    // Compute log-mel spectrogram → MFCCs
    const mfccMatrix = []; // T × nMfcc
    for (let t = 0; t < T; t++) {
        const center = t * hopLen;
        const start = center - Math.floor(winLen / 2);

        const frame = new Float32Array(winLen);
        for (let i = 0; i < winLen; i++) {
            const idx = start + i;
            // Librosa uses 'reflect' padding by default, but zero-padding 
            // is a close enough and standard approximation for STFT edges.
            frame[i] = (idx >= 0 && idx < pcm.length ? pcm[idx] : 0) * window[i];
        }
        const pwr = powerSpec(frame, nFft);

        // Apply mel filters
        const melEnergy = new Float32Array(nFilt);
        for (let m = 0; m < nFilt; m++) {
            let e = 0;
            for (let k = 0; k < pwr.length; k++) e += filters[m][k] * pwr[k];
            melEnergy[m] = Math.log(Math.max(e, 1e-10));
        }
        // DCT to get MFCCs (coefficients 0..nMfcc-1)
        const mfcc = dct2(melEnergy, nMfcc);
        mfccMatrix.push(mfcc);
    }

    // ── Deltas ───────────────────────────────────────────────────────────────
    const mfccD1 = delta(mfccMatrix, 1);
    const mfccD2 = delta(mfccMatrix, 2);

    /**
     * Python constructs per-frame features as concat of:
     *   np.mean(mfcc[:, t:t+1], axis=1)   → shape (13,)  — just mfcc[t]
     *   np.std (mfcc[:, t:t+1], axis=1)   → shape (13,)  — zeros (single col)
     *   np.mean(delta[:, t:t+1], axis=1)  → shape (13,)  — delta[t]
     *   np.mean(delta2[:,t:t+1], axis=1)  → shape (13,)  — delta2[t]
     * std of a single-column slice is 0, so we replicate that.
     */
    const features = [];
    const EPS = 1e-8;
    for (let t = 0; t < T; t++) {
        const feat = new Float32Array(4 * nMfcc); // 52
        for (let c = 0; c < nMfcc; c++) {
            feat[c] = mfccMatrix[t][c];        // mean
            feat[nMfcc + c] = 0;                        // std of single col = 0
            feat[2 * nMfcc + c] = mfccD1[t][c];           // delta
            feat[3 * nMfcc + c] = mfccD2[t][c];           // delta2
        }
        features.push(feat);
    }
    return features; // T × 52
}
