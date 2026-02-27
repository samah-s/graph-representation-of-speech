/**
 * trainer.js — JS port of PerGraphCompressionTrainer
 * 
 * Implements SpeakerEmbeddingNet with manual backprop and Adam optimizer.
 * Uses finite differences for parameters that affect discrete graph operations (aRaw, logTemperature, mergeScale).
 */

'use strict';

import { SoftCompressionModule } from './softCompress.js';
import { dot, norm, cosine } from './similarity.js';

const EPS_FD = 1e-6;

// ─── Simple Neural Network Components ────────────────────────────────────────

class Linear {
    constructor(inDim, outDim) {
        this.inDim = inDim;
        this.outDim = outDim;
        this.weights = new Float32Array(inDim * outDim);
        this.bias = new Float32Array(outDim);
        this.gradWeights = new Float32Array(inDim * outDim);
        this.gradBias = new Float32Array(outDim);

        // PyTorch default init: U(-k, k) where k = 1/sqrt(inDim)
        const k = 1 / Math.sqrt(inDim);
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] = (Math.random() * 2 - 1) * k;
        }
        for (let j = 0; j < this.bias.length; j++) {
            this.bias[j] = (Math.random() * 2 - 1) * k;
        }
    }

    forward(input) {
        this.input = input; // Input is Array of vectors [N][inDim] or single vector [inDim]
        if (Array.isArray(input)) {
            const N = input.length;
            const output = Array.from({ length: N }, () => new Float32Array(this.outDim));
            for (let n = 0; n < N; n++) {
                for (let j = 0; j < this.outDim; j++) {
                    let s = this.bias[j];
                    for (let i = 0; i < this.inDim; i++) {
                        s += input[n][i] * this.weights[j * this.inDim + i];
                    }
                    output[n][j] = s;
                }
            }
            return output;
        } else {
            const output = new Float32Array(this.outDim);
            for (let j = 0; j < this.outDim; j++) {
                let s = this.bias[j];
                for (let i = 0; i < this.inDim; i++) {
                    s += input[i] * this.weights[j * this.inDim + i];
                }
                output[j] = s;
            }
            return output;
        }
    }

    backward(gradOutput) {
        // gradOutput is [N][outDim] or [outDim]
        if (Array.isArray(this.input)) {
            const N = this.input.length;
            const gradInput = Array.from({ length: N }, () => new Float32Array(this.inDim));
            for (let n = 0; n < N; n++) {
                for (let j = 0; j < this.outDim; j++) {
                    const go = gradOutput[n][j];
                    this.gradBias[j] += go;
                    for (let i = 0; i < this.inDim; i++) {
                        this.gradWeights[j * this.inDim + i] += go * this.input[n][i];
                        gradInput[n][i] += go * this.weights[j * this.inDim + i];
                    }
                }
            }
            return gradInput;
        } else {
            const gradInput = new Float32Array(this.inDim);
            for (let j = 0; j < this.outDim; j++) {
                const go = gradOutput[j];
                this.gradBias[j] += go;
                for (let i = 0; i < this.inDim; i++) {
                    this.gradWeights[j * this.inDim + i] += go * this.input[i];
                    gradInput[i] += go * this.weights[j * this.inDim + i];
                }
            }
            return gradInput;
        }
    }

    zeroGrad() {
        this.gradWeights.fill(0);
        this.gradBias.fill(0);
    }
}

class ReLU {
    forward(input) {
        this.input = input;
        if (Array.isArray(input)) {
            return input.map(row => row.map(v => Math.max(0, v)));
        } else {
            return input.map(v => Math.max(0, v));
        }
    }
    backward(gradOutput) {
        if (Array.isArray(this.input)) {
            return gradOutput.map((row, n) => row.map((go, j) => this.input[n][j] > 0 ? go : 0));
        } else {
            return gradOutput.map((go, j) => this.input[j] > 0 ? go : 0);
        }
    }
}

class Adam {
    constructor(params, lr = 0.14, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
        this.params = params; // params is object with { weights: Float32Array, grads: Float32Array }
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.m = params.map(p => new Float32Array(p.weights.length));
        this.v = params.map(p => new Float32Array(p.weights.length));
        this.t = 0;
    }

    step() {
        this.t++;
        for (let i = 0; i < this.params.length; i++) {
            const p = this.params[i];
            const m = this.m[i];
            const v = this.v[i];
            for (let j = 0; j < p.weights.length; j++) {
                const g = p.grads[j];
                m[j] = this.beta1 * m[j] + (1 - this.beta1) * g;
                v[j] = this.beta2 * v[j] + (1 - this.beta2) * g * g;
                const mHat = m[j] / (1 - Math.pow(this.beta1, this.t));
                const vHat = v[j] / (1 - Math.pow(this.beta2, this.t));
                p.weights[j] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
            }
        }
    }
}

// ─── SpeakerEmbeddingNet ─────────────────────────────────────────────────────

class SpeakerEmbeddingNet {
    constructor(inputDim = 52, hiddenDim = 64, outputDim = 32) {
        this.nodeEncoder = [
            new Linear(inputDim, hiddenDim), new ReLU(),
            new Linear(hiddenDim, hiddenDim), new ReLU()
        ];
        this.attention = new Linear(hiddenDim, 1);
        this.outputLayer = new Linear(hiddenDim, outputDim);
    }

    forward(graphFeatures) {
        this.graphFeatures = graphFeatures;
        let x = graphFeatures;
        for (const layer of this.nodeEncoder) x = layer.forward(x);
        this.nf = x; // [N][hiddenDim]

        const attnScores = this.attention.forward(x); // [N][1]

        // --- Numerically Stable Softmax ---
        const scores = attnScores.map(s => s[0]);
        const maxS = Math.max(...scores);
        const expScores = scores.map(s => Math.exp(s - maxS));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        this.aw = expScores.map(s => s / (sumExp + 1e-8)); // [N]

        const sumWeighted = new Float32Array(this.nf[0].length);
        for (let n = 0; n < this.nf.length; n++) {
            const w = this.aw[n];
            const row = this.nf[n];
            for (let d = 0; d < sumWeighted.length; d++) {
                sumWeighted[d] += w * row[d];
            }
        }
        this.sumWeighted = sumWeighted;
        return this.outputLayer.forward(sumWeighted);
    }

    backward(gradOutput) {
        // gradOutput is [outputDim]
        const gradSumWeighted = this.outputLayer.backward(gradOutput); // [hiddenDim]

        const gradNF = Array.from({ length: this.nf.length }, () => new Float32Array(this.nf[0].length));
        const gradAW = new Float32Array(this.nf.length);

        for (let n = 0; n < this.nf.length; n++) {
            gradAW[n] = dot(gradSumWeighted, this.nf[n]);
            for (let d = 0; d < this.nf[0].length; d++) {
                gradNF[n][d] = gradSumWeighted[d] * this.aw[n];
            }
        }

        // Softmax backward
        const gradAttnScores = Array.from({ length: this.nf.length }, () => new Float32Array(1));
        for (let i = 0; i < this.nf.length; i++) {
            let s = 0;
            for (let j = 0; j < this.nf.length; j++) {
                s += gradAW[j] * this.aw[j] * ((i === j ? 1 : 0) - this.aw[i]);
            }
            gradAttnScores[i][0] = s;
        }

        const gradNFFromAttn = this.attention.backward(gradAttnScores);
        for (let n = 0; n < this.nf.length; n++) {
            for (let d = 0; d < this.nf[0].length; d++) {
                gradNF[n][d] += gradNFFromAttn[n][d];
            }
        }

        let gradX = gradNF;
        for (let i = this.nodeEncoder.length - 1; i >= 0; i--) {
            gradX = this.nodeEncoder[i].backward(gradX);
        }
        return gradX;
    }

    zeroGrad() {
        for (const layer of this.nodeEncoder) if (layer.zeroGrad) layer.zeroGrad();
        this.attention.zeroGrad();
        this.outputLayer.zeroGrad();
    }

    getParams() {
        const p = [];
        const add = (l) => p.push({ weights: l.weights, grads: l.gradWeights }, { weights: l.bias, grads: l.gradBias });
        add(this.nodeEncoder[0]);
        add(this.nodeEncoder[2]);
        add(this.attention);
        add(this.outputLayer);
        return p;
    }
}

// ─── PerGraphCompressionTrainer ─────────────────────────────────────────────

export class PerGraphCompressionTrainer {
    constructor(features, edges, opts = {}) {
        this.features = features;
        this.edges = edges;

        // --- Sanity Checks for Features ---
        let min = Infinity, max = -Infinity, hasNaN = false;
        for (const f of features) {
            for (let d = 0; d < f.length; d++) {
                const v = f[d];
                if (isNaN(v)) hasNaN = true;
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        console.log(`[Trainer] Features: min=${min.toFixed(4)}, max=${max.toFixed(4)}, NaN=${hasNaN}, N=${features.length}`);
        if (hasNaN) console.error('[Trainer] ERROR: Input features contain NaNs!');

        this.lambdaId = opts.lambdaId ?? 0.6;
        this.lambdaComp = opts.lambdaComp ?? 0.4;
        this.module = new SoftCompressionModule(opts.initialAlpha ?? 0.5);
        this.net = new SpeakerEmbeddingNet();
        this.optimizer = new Adam([
            ...this.net.getParams(),
            { weights: new Float32Array([this.module.aRaw]), grads: new Float32Array(1) },
            { weights: new Float32Array([this.module.logTemperature]), grads: new Float32Array(1) }
        ], opts.lr ?? 0.14);

        this.history = { total: [], identity: [], compression: [], alpha: [], gradNorm: [] };
    }

    computeIdentityLoss(eOrig, eComp) {
        return 1 - cosine(eOrig, eComp);
    }

    computeCompressionLoss(mergeProbs) {
        if (!mergeProbs.length) return 0;
        return mergeProbs.reduce((s, p) => s + (1 - p), 0) / mergeProbs.length;
    }

    trainStep() {
        this.net.zeroGrad();
        const params = this.optimizer.params;

        // Zero gradients for module parameters in optimizer
        params[params.length - 2].grads.fill(0); // aRaw
        params[params.length - 1].grads.fill(0); // logTemperature

        // 1. Forward Pass
        // Pass 1: original features
        const eOrig = this.net.forward(this.features);
        // Pass 2: soft compress logic
        const { featuresUpdated, mergeProbs } = this.module.softCompress(this.features, this.edges);
        // Pass 3: compressed features
        const eComp = this.net.forward(featuresUpdated);

        // 2. Compute Losses
        const lId = this.computeIdentityLoss(eOrig, eComp);
        const lComp = this.computeCompressionLoss(mergeProbs);
        const lTot = this.lambdaId * lId + this.lambdaComp * lComp;

        // 3. Backprop for Net
        const getGradCosine = (A, B) => {
            const nA = norm(A), nB = norm(B);
            const dotVal = dot(A, B);
            const grad = new Float32Array(B.length);
            for (let i = 0; i < B.length; i++) {
                // d(cos(A,B))/dB_i = (A_i * nB^2 - dot(A,B) * B_i) / (nA * nB^3)
                grad[i] = (A[i] * nB * nB - dotVal * B[i]) / (nA * nB * nB * nB + 1e-8);
            }
            return grad;
        };

        // dL_tot / d_eComp
        const gradEComp = getGradCosine(eOrig, eComp).map(v => -this.lambdaId * v);
        // dL_tot / d_featuresUpdated (this will be used by SoftCompress.backward too)
        const gradFeaturesUpdated = this.net.backward(gradEComp);

        // dL_tot / d_eOrig
        const gradEOrig = getGradCosine(eComp, eOrig).map(v => -this.lambdaId * v);
        // Restore net state to original for eOrig backward
        this.net.forward(this.features);
        this.net.backward(gradEOrig);

        // 4. Backprop for Soft Compression Module
        // gradMergeProbs: dL_tot / dp_t = -lambdaComp / M
        const M = mergeProbs.length;
        const gradMergeProbs = new Float32Array(M).fill(M > 0 ? -this.lambdaComp / M : 0);

        this.module.backward(gradFeaturesUpdated, gradMergeProbs);

        // 5. Update Optimizer Params
        params[params.length - 2].grads[0] = this.module.gradARaw;
        params[params.length - 1].grads[0] = this.module.gradLogTemp;

        this.optimizer.step();

        // Sync parameters back to module
        this.module.aRaw = params[params.length - 2].weights[0];
        this.module.logTemperature = params[params.length - 1].weights[0];

        if (Math.random() < 0.1) {
            console.log(`[Epoch Update] Total: ${lTot.toFixed(4)} | Id: ${lId.toFixed(4)} | Comp: ${lComp.toFixed(4)}`);
            console.log(`[Params] Alpha: ${this.module.getAlpha().toFixed(4)} | Temp: ${this.module.temperature.toFixed(4)} | gA: ${this.module.gradARaw.toFixed(6)}`);
        }

        return {
            totalLoss: lTot,
            identityLoss: lId,
            compressionLoss: lComp,
            alpha: this.module.getAlpha(),
            gradNorm: Math.abs(this.module.gradARaw)
        };
    }

    async train(epochs = 20, onEpoch = null) {
        console.log(`%c Starting Alpha Learning (${epochs} epochs) `, 'background: #222; color: #bada55; padding: 2px;');
        for (let e = 0; e < epochs; e++) {
            const m = this.trainStep();
            this.history.total.push(m.totalLoss);
            this.history.identity.push(m.identityLoss);
            this.history.compression.push(m.compressionLoss);
            this.history.alpha.push(m.alpha);
            this.history.gradNorm.push(m.gradNorm);

            if (e % 5 === 0) {
                console.group(`Epoch ${e}`);
                console.table({
                    loss: { total: m.totalLoss.toFixed(4), identity: m.identityLoss.toFixed(4), compression: m.compressionLoss.toFixed(4) },
                    params: { alpha: m.alpha.toFixed(4), temp: this.module.temperature.toFixed(4), scale: this.module.mergeScale.toFixed(4) }
                });
                console.groupEnd();
            }

            if (onEpoch) onEpoch(e, m);
            await new Promise(r => setTimeout(r, 0));
        }
        console.log(`%c Alpha Learning Complete. Final Alpha: ${this.module.getAlpha().toFixed(6)} `, 'background: #222; color: #bada55; padding: 2px;');
        return this.module.getAlpha();
    }
}
