/**
 * softCompress.js — JS port of SoftCompressionModule (from graph_compression_visualisation.py)
 *
 * Key features:
 *  - High-precision analytical backpropagation parity with PyTorch
 *  - Stays in t-th path (stores best_k, best_p, tau) during backward pass
 *  - Matches Librosa-style framing and features
 */

'use strict';

import { simN, simNComponents, getNeighbours, cosine, dot, norm, gradCosine, meanVec } from './similarity.js';

const EPS = 1e-8;

// ── sigmoid / inverse ─────────────────────────────────────────────────────────
export function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
export function sigmoidInverse(p) {
    const eps = 1e-10;
    const pp = Math.max(eps, Math.min(1 - eps, p));
    return Math.log(pp / (1 - pp));
}

function torchQuantile(arr, q) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (base + 1 < sorted.length) {
        return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    }
    return sorted[base];
}

// ─────────────────────────────────────────────────────────────────────────────
export class SoftCompressionModule {
    /**
     * @param {number} initialAlpha   — default 0.5
     * @param {number} initialTemp    — default 0.1
     */
    constructor(initialAlpha = 0.5, initialTemp = 0.1) {
        // a_raw_init logic
        this.aRaw = initialAlpha === 0.5 ? 0 : sigmoidInverse(initialAlpha);
        this.logTemperature = Math.log(initialTemp);
        this.mergeScale = 1.0;
    }

    getAlpha() { return sigmoid(this.aRaw); }
    get temperature() { return Math.exp(this.logTemperature); }

    /**
     * compute_merge_probability(t, k, x, edge_index, tau1, tau2)
     * Mirrors Python's sigmoid((s - tau) / temp) multiplied.
     */
    computeMergeProbability(t, k, features, edges, tau1, tau2) {
        const N = features.length;
        if (t + k >= N || t - 1 < 0 || t + k - 1 >= N || t + 1 >= N || t + k + 1 >= N) {
            return 0;
        }

        const alpha = this.getAlpha();
        const temp = this.temperature;

        const s1 = simN(t, t + k, features, edges, alpha);
        const s2 = simN(t - 1, t + k - 1, features, edges, alpha);
        const s3 = simN(t + 1, t + k + 1, features, edges, alpha);

        const p1 = sigmoid((s1 - tau1) / temp);
        const p2 = sigmoid((s2 - tau2) / temp);
        const p3 = sigmoid((s3 - tau2) / temp);

        return p1 * p2 * p3;
    }

    /**
     * soft_compress(data, tau1=None, tau2=None, window_size=10)
     */
    softCompress(features, edges) {
        const N = features.length;
        const featuresUpdated = features.map(f => new Float32Array(f));
        const allMergeProbs = [];
        const stats = []; // Store intermediate values for backprop

        for (let t = 1; t < N - 2; t++) {
            const similarities = [];
            for (let k = 1; k < N - t; k++) {
                similarities.push(simN(t, t + k, features, edges, this.getAlpha()));
            }

            if (similarities.length > 0) {
                // Match Python torch.quantile(sim_stack, 0.9) with linear interpolation
                const tau1 = torchQuantile(similarities, 0.9);
                const tau2 = torchQuantile(similarities, 0.85);

                let bestP = -1, bestK = -1, bestStats = null;
                for (let k = 1; k < N - t; k++) {
                    const N_ = features.length;
                    if (t + k >= N_ || t - 1 < 0 || t + k - 1 >= N_ || t + 1 >= N_ || t + k + 1 >= N_) continue;

                    const alpha = this.getAlpha();
                    const c1 = simNComponents(t, t + k, features, edges);
                    const c2 = simNComponents(t - 1, t + k - 1, features, edges);
                    const c3 = simNComponents(t + 1, t + k + 1, features, edges);

                    const s1 = alpha * c1.s_x + (1 - alpha) * c1.s_m;
                    const s2 = alpha * c2.s_x + (1 - alpha) * c2.s_m;
                    const s3 = alpha * c3.s_x + (1 - alpha) * c3.s_m;

                    const p1 = sigmoid((s1 - tau1) / this.temperature);
                    const p2 = sigmoid((s2 - tau2) / this.temperature);
                    const p3 = sigmoid((s3 - tau2) / this.temperature);
                    const p = p1 * p2 * p3;

                    if (p > bestP) {
                        bestP = p;
                        bestK = k;
                        bestStats = {
                            s1, s2, s3, p1, p2, p3, p,
                            tau1, tau2, t, k,
                            c1, c2, c3
                        };
                    }
                }

                if (bestK !== -1) {
                    allMergeProbs.push(bestP);
                    const strength = Math.max(0, Math.min(1, bestP * 1.2));
                    const fT = features[t];
                    const fK = features[t + bestK];
                    for (let d = 0; d < fT.length; d++) {
                        featuresUpdated[t][d] = (1 - strength) * fT[d] + strength * (fT[d] + fK[d]) / 2;
                    }
                    stats.push({ ...bestStats, strength });
                }
            }
        }

        this.lastStats = stats;
        this.lastFeatures = features;
        return {
            featuresUpdated,
            mergeProbs: allMergeProbs,
        };
    }

    backward(gradFeaturesUpdated, gradMergeProbs) {
        const alpha = this.getAlpha();
        const temp = this.temperature;
        const dAlphaD_aRaw = alpha * (1 - alpha);
        const N = this.lastFeatures.length;
        const dim = this.lastFeatures[0].length;

        this.gradARaw = 0;
        this.gradLogTemp = 0;
        const gradFeatures = Array.from({ length: N }, () => new Float32Array(dim));

        // 1. Identity path contribution
        for (let t = 0; t < N; t++) {
            const gF = gradFeaturesUpdated[t];
            const stat = this.lastStats.find(s => s.t === t);
            if (stat) {
                const strength = stat.strength;
                for (let d = 0; d < dim; d++) {
                    gradFeatures[t][d] += gF[d] * (1 - strength / 2);
                    gradFeatures[t + stat.k][d] += gF[d] * (strength / 2);
                }
            } else {
                for (let d = 0; d < dim; d++) gradFeatures[t][d] += gF[d];
            }
        }

        // 2. Probability path contribution
        for (let i = 0; i < this.lastStats.length; i++) {
            const s = this.lastStats[i];
            const t = s.t;
            const k = s.k;
            const fT = this.lastFeatures[t];
            const fK = this.lastFeatures[t + k];

            let dL_dStrength = 0;
            const gfT_upd = gradFeaturesUpdated[t];
            for (let d = 0; d < dim; d++) dL_dStrength += gfT_upd[d] * (fK[d] - fT[d]) / 2;

            const dStrength_dp = (s.p * 1.2 < 1.0) ? 1.2 : 0;
            const dL_dp = dL_dStrength * dStrength_dp + gradMergeProbs[i];

            const dz1 = 1 / temp;
            const dp_ds1 = s.p2 * s.p3 * dz1 * s.p1 * (1 - s.p1);
            const dp_ds2 = s.p1 * s.p3 * dz1 * s.p2 * (1 - s.p2);
            const dp_ds3 = s.p1 * s.p2 * dz1 * s.p3 * (1 - s.p3);

            this.gradARaw += dL_dp * (dp_ds1 * (s.c1.s_x - s.c1.s_m) + dp_ds2 * (s.c2.s_x - s.c2.s_m) + dp_ds3 * (s.c3.s_x - s.c3.s_m)) * dAlphaD_aRaw;

            const dPt = (ss, p, tau) => -(ss - tau) / (temp * temp) * p * (1 - p);
            const dp_dt = (dPt(s.s1, s.p1, s.tau1) * s.p2 * s.p3) + (s.p1 * dPt(s.s2, s.p2, s.tau2) * s.p3) + (s.p1 * s.p2 * dPt(s.s3, s.p3, s.tau2));
            this.gradLogTemp += dL_dp * dp_dt * temp;

            const backpropSim = (comp, dL_ds) => {
                const gS = dL_dp * dL_ds;
                const { gradA: gX_u, gradB: gX_v } = gradCosine(this.lastFeatures[comp.u], this.lastFeatures[comp.v], gS * alpha);
                for (let d = 0; d < dim; d++) { gradFeatures[comp.u][d] += gX_u[d]; gradFeatures[comp.v][d] += gX_v[d]; }

                if (comp.nu && comp.nv) {
                    const mu_u = meanVec(comp.nu, this.lastFeatures), mu_v = meanVec(comp.nv, this.lastFeatures);
                    const { gradA: gM_u, gradB: gM_v } = gradCosine(mu_u, mu_v, gS * (1 - alpha));
                    for (const idx of comp.nu) for (let d = 0; d < dim; d++) gradFeatures[idx][d] += gM_u[d] / comp.nu.length;
                    for (const idx of comp.nv) for (let d = 0; d < dim; d++) gradFeatures[idx][d] += gM_v[d] / comp.nv.length;
                }
            };
            backpropSim(s.c1, dp_ds1);
            backpropSim(s.c2, dp_ds2);
            backpropSim(s.c3, dp_ds3);
        }
        return gradFeatures;
    }
}
