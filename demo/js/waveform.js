/**
 * waveform.js — Animated canvas waveform visualizer
 *
 * • Live scrolling waveform during recording (Web Audio AnalyserNode)
 * • Static preview after load/upload
 */

'use strict';

const BAR_GAP = 2;

export class WaveformRenderer {
    /**
     * @param {HTMLCanvasElement} canvas
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this._rafId = null;
        this._mode = 'idle';    // 'idle' | 'live' | 'static'
        this._analyser = null;
        this._dataArray = null;
        this._staticPCM = null;
        this._playProgress = 0;   // 0-1 playback indicator
    }

    // ── Live recording mode ──────────────────────────────────────────────────
    startLive(analyserNode) {
        this._mode = 'live';
        this._analyser = analyserNode;
        this._analyser.fftSize = 2048;
        this._dataArray = new Uint8Array(this._analyser.frequencyBinCount);
        this._liveHistory = [];
        this._loop();
    }

    stopLive() {
        this._mode = 'idle';
        this._stop();
    }

    // ── Static preview from decoded PCM ─────────────────────────────────────
    showStatic(pcm) {
        this._mode = 'static';
        this._staticPCM = pcm;
        this._stop();
        this._drawStatic();
    }

    clearCanvas() {
        const { canvas, ctx } = this;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        this._mode = 'idle';
        this._stop();
    }

    // ── Internal helpers ─────────────────────────────────────────────────────
    _stop() {
        if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    }

    _loop() {
        this._rafId = requestAnimationFrame(() => {
            this._drawLive();
            if (this._mode === 'live') this._loop();
        });
    }

    _drawLive() {
        const { canvas, ctx, _analyser, _dataArray } = this;
        if (!_analyser) return;
        _analyser.getByteTimeDomainData(_dataArray);

        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        // Gradient fill under waveform
        const grad = ctx.createLinearGradient(0, 0, 0, H);
        grad.addColorStop(0, 'var(--wave-top, rgba(99,102,241,0.8))');
        grad.addColorStop(1, 'var(--wave-bot, rgba(99,102,241,0.1))');

        ctx.beginPath();
        ctx.moveTo(0, H / 2);
        const sliceW = W / _dataArray.length;
        for (let i = 0; i < _dataArray.length; i++) {
            const v = _dataArray[i] / 128;
            const y = (v * H) / 2;
            i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * sliceW, y);
        }
        ctx.strokeStyle = 'var(--wave-top, rgba(99,102,241,0.9))';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Mirror below center
        ctx.beginPath();
        ctx.moveTo(0, H / 2);
        for (let i = 0; i < _dataArray.length; i++) {
            const v = _dataArray[i] / 128;
            const y = H - (v * H) / 2;
            i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * sliceW, y);
        }
        ctx.strokeStyle = 'rgba(99,102,241,0.35)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Center line
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(99,102,241,0.2)';
        ctx.lineWidth = 1;
        ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2);
        ctx.stroke();
    }

    _drawStatic() {
        const { canvas, ctx, _staticPCM } = this;
        if (!_staticPCM) return;
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const N = _staticPCM.length;
        const step = Math.max(1, Math.floor(N / W));
        const midY = H / 2;

        // Build bar heights
        const bars = [];
        for (let x = 0; x < W; x++) {
            let max = 0;
            for (let s = 0; s < step; s++) {
                const idx = Math.min(x * step + s, N - 1);
                max = Math.max(max, Math.abs(_staticPCM[idx]));
            }
            bars.push(max);
        }
        const maxBar = Math.max(...bars, 0.001);

        // Draw bars
        for (let x = 0; x < W; x += BAR_GAP + 1) {
            const amp = (bars[x] / maxBar) * (H / 2 - 4);
            const prog = x / W;
            // Color gradient along time axis
            const hue = 240 + prog * 40;  // indigo → blue
            ctx.fillStyle = `hsla(${hue},70%,65%,0.85)`;
            ctx.fillRect(x, midY - amp, 1, amp * 2);
        }

        // Playhead center line
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(165,160,255,0.3)';
        ctx.lineWidth = 1;
        ctx.moveTo(0, midY); ctx.lineTo(W, midY);
        ctx.stroke();
    }
}
