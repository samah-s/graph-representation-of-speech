/**
 * charts.js — Animated Canvas charts for alpha + loss curves
 *
 * Mirrors Python plt calls:
 *   axes[0]: alpha over epochs (steelblue line + crimson dashed final)
 *   axes[1]: total / identity / compression loss curves
 *
 * All drawn pure Canvas, no libraries.
 */

'use strict';

const PAD = { top: 24, right: 24, bottom: 36, left: 52 };
const FONT = "13px 'Inter', sans-serif";
const FONT_SMALL = "11px 'Inter', sans-serif";

function lerp(a, b, t) { return a + (b - a) * t; }

// ── Smooth number counter ────────────────────────────────────────────────────
export class NumberTicker {
    constructor(el, decimals = 4) {
        this.el = el;
        this.decimals = decimals;
        this.current = 0;
        this.target = 0;
        this._start = 0;
        this._from = 0;
        this._dur = 600;
        this._rafId = null;
    }
    to(value) {
        this._from = this.current;
        this.target = value;
        this._start = performance.now();
        if (!this._rafId) this._tick();
    }
    _tick() {
        const t = Math.min(1, (performance.now() - this._start) / this._dur);
        const et = 1 - Math.pow(1 - t, 3);
        this.current = lerp(this._from, this.target, et);
        this.el.textContent = this.current.toFixed(this.decimals);
        if (t < 1) { this._rafId = requestAnimationFrame(() => this._tick()); }
        else { this._rafId = null; }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
export class LineChart {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {{ label, color, dashed? }[]} series
     * @param {string} title
     */
    constructor(canvas, series, title = '') {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.series = series.map(s => ({ ...s, data: [] }));
        this.title = title;
        this._drawn = 0;   // epochs drawn so far
        this._target = 0;   // epochs to animate to
        this._rafId = null;
    }

    /** Push one epoch's values for all series simultaneously */
    push(values) {
        values.forEach((v, i) => this.series[i].data.push(v));
        this._target = this.series[0].data.length;
        if (!this._rafId) this._scheduleRender();
    }

    /** Animate to the next epoch point */
    _scheduleRender() {
        this._rafId = requestAnimationFrame(() => {
            if (this._drawn < this._target) {
                this._drawn = Math.min(this._drawn + 1, this._target);
                this.render();
                if (this._drawn < this._target) this._scheduleRender();
                else this._rafId = null;
            } else {
                this._rafId = null;
            }
        });
    }

    render() {
        const { canvas, ctx, series, title } = this;
        const W = canvas.width, H = canvas.height;
        const { top: pT, right: pR, bottom: pB, left: pL } = PAD;
        const plotW = W - pL - pR;
        const plotH = H - pT - pB;

        // Background
        ctx.clearRect(0, 0, W, H);

        // All data up to _drawn
        const drawn = this._drawn;
        const allVals = series.flatMap(s => s.data.slice(0, drawn));
        if (!allVals.length) return;

        const minV = Math.min(0, ...allVals);
        const maxV = Math.max(...allVals) || 1;
        const range = maxV - minV || 1;

        const xScale = i => pL + (i / Math.max(1, drawn - 1)) * plotW;
        const yScale = v => pT + plotH - ((v - minV) / range) * plotH;

        // Gridlines
        ctx.strokeStyle = getComputedStyle(document.documentElement)
            .getPropertyValue('--chart-grid').trim() || 'rgba(128,128,128,0.15)';
        ctx.lineWidth = 1;
        const gridN = 4;
        for (let g = 0; g <= gridN; g++) {
            const y = pT + (g / gridN) * plotH;
            ctx.beginPath(); ctx.moveTo(pL, y); ctx.lineTo(W - pR, y); ctx.stroke();
            const val = maxV - (g / gridN) * range;
            ctx.fillStyle = getComputedStyle(document.documentElement)
                .getPropertyValue('--chart-label').trim() || '#888';
            ctx.font = FONT_SMALL;
            ctx.textAlign = 'right';
            ctx.fillText(val.toFixed(3), pL - 4, y + 4);
        }

        // Axes
        ctx.strokeStyle = getComputedStyle(document.documentElement)
            .getPropertyValue('--chart-axis').trim() || 'rgba(128,128,128,0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(pL, pT); ctx.lineTo(pL, pT + plotH);
        ctx.lineTo(pL + plotW, pT + plotH);
        ctx.stroke();

        // x-axis labels
        const nTicks = Math.min(drawn, 10);
        ctx.fillStyle = getComputedStyle(document.documentElement)
            .getPropertyValue('--chart-label').trim() || '#888';
        ctx.font = FONT_SMALL;
        ctx.textAlign = 'center';
        for (let i = 0; i <= nTicks; i++) {
            const epochIdx = Math.round((i / nTicks) * (drawn - 1));
            ctx.fillText(epochIdx + 1, xScale(epochIdx), pT + plotH + 18);
        }

        // Series lines
        for (const s of series) {
            const pts = s.data.slice(0, drawn);
            if (!pts.length) continue;
            ctx.beginPath();
            ctx.strokeStyle = s.color;
            ctx.lineWidth = s.bold ? 2.5 : 1.8;
            if (s.dashed) ctx.setLineDash([6, 4]);
            else ctx.setLineDash([]);
            pts.forEach((v, i) => {
                const x = xScale(i), y = yScale(v);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.setLineDash([]);

            // Dot at latest point
            const last = pts[pts.length - 1];
            ctx.beginPath();
            ctx.arc(xScale(pts.length - 1), yScale(last), 3.5, 0, Math.PI * 2);
            ctx.fillStyle = s.color;
            ctx.fill();
        }

        // Final alpha dashed line (alpha chart only)
        if (this._finalAlpha !== undefined) {
            const fy = yScale(this._finalAlpha);
            ctx.beginPath();
            ctx.strokeStyle = '#dc2626';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 4]);
            ctx.moveTo(pL, fy); ctx.lineTo(W - pR, fy);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#dc2626';
            ctx.font = FONT_SMALL;
            ctx.textAlign = 'left';
            ctx.fillText(`α=${this._finalAlpha.toFixed(4)}`, pL + 4, fy - 4);
        }

        // Title
        ctx.fillStyle = getComputedStyle(document.documentElement)
            .getPropertyValue('--text-primary').trim() || '#111';
        ctx.font = `600 ${FONT}`;
        ctx.textAlign = 'center';
        ctx.fillText(title, W / 2, 16);

        // Legend
        let lx = pL;
        ctx.font = FONT_SMALL;
        for (const s of series) {
            ctx.fillStyle = s.color;
            ctx.fillRect(lx, H - 10, 18, 3);
            ctx.fillStyle = getComputedStyle(document.documentElement)
                .getPropertyValue('--text-secondary').trim() || '#555';
            ctx.textAlign = 'left';
            ctx.fillText(s.label, lx + 22, H - 7);
            lx += ctx.measureText(s.label).width + 42;
        }
    }

    setFinalAlpha(a) {
        this._finalAlpha = a;
        this.render();
    }

    resize(w, h) {
        this.canvas.width = w;
        this.canvas.height = h;
        this.render();
    }
}
