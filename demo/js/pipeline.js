/**
 * pipeline.js — Orchestration of the full processing pipeline
 *
 * Steps (mirrors Python main pipeline):
 *   1. Build temporal graph from WAV features
 *   2. Learn alpha via soft compression training (20 epochs)
 *   3. Apply soft compression (feature update, same topology)
 *   4. Hard compression — step-by-step node removal + edge rerouting
 */

'use strict';

import { extractFeatures } from './dsp.js';
import { buildTemporalGraph } from './graph.js';
import { PerGraphCompressionTrainer } from './trainer.js';
import { SoftCompressionModule } from './softCompress.js';
import { hardCompress } from './hardCompress.js';
import { GraphRenderer } from './graphRenderer.js';
import { LineChart, NumberTicker } from './charts.js';

const EPOCHS = 20;

// ─────────────────────────────────────────────────────────────────────────────
export class Pipeline {
    constructor(ui) {
        this.ui = ui;
        this._running = false;
        this._renderer = null;
    }

    cancel() { this._running = false; }

    async run(pcm) {
        if (this._running) return;
        this._running = true;
        const { ui } = this;

        console.log(`\n%c============================================================`, 'color: #888');
        console.log(`%c  STEP 1/4 — Build temporal graph from WAV`, 'font-weight: bold; color: #fff');
        console.log(`%c============================================================`, 'color: #888');

        try {
            // ── Step 1: Extract features & build temporal graph ─────────────────
            this._setStep(1, 'Building temporal graph…');
            ui.statusText.textContent = 'Extracting MFCC features…';

            const features = extractFeatures(pcm, { sr: 16000, winMs: 50, hopMs: 100, nMfcc: 13 });
            if (!this._running) return;

            const { nodes, edges, numNodes } = buildTemporalGraph(features);
            console.log(`  Nodes     : ${numNodes}`);
            console.log(`  Edges     : ${edges.length}`);
            console.log(`  Feature dim: ${features[0].length}`);
            console.log(`\n  [INFO] Graph has ${numNodes} nodes — visualising all nodes.`);

            this._stat('nodes', numNodes);
            this._stat('edges', edges.length);
            this._stat('remaining', numNodes);
            this._stat('removed', 0);
            this._stat('ratio', '—');

            // Initialise graph renderer (mirrors Python step-0 visualise_compressed_graph call)
            GraphRenderer.injectDefs(ui.graphSvg);
            if (this._renderer) this._renderer.destroy();
            this._renderer = new GraphRenderer(ui.graphSvg);
            this._renderer.init(numNodes, edges, new Set());

            ui.statusText.textContent = `Graph built — ${numNodes} nodes, ${edges.length} edges`;
            await sleep(500);
            if (!this._running) return;

            // ── Step 2: Soft-compression training (learn alpha) ─────────────────
            console.log(`\n%c============================================================`, 'color: #888');
            console.log(`%c  STEP 2/4 — Learn alpha via soft compression`, 'font-weight: bold; color: #fff');
            console.log(`%c============================================================`, 'color: #888');
            this._setStep(2, 'Learning α via soft compression…');

            // Set up charts
            const alphaChart = new LineChart(
                ui.alphaCanvas,
                [{ label: 'α (alpha)', color: '#3b82f6', bold: true }],
                'α over soft-compression epochs'
            );
            const lossChart = new LineChart(
                ui.lossCanvas,
                [
                    { label: 'Total', color: '#6366f1', bold: true },
                    { label: 'Identity', color: '#06b6d4' },
                    { label: 'Compression', color: '#f59e0b' },
                ],
                'Loss curves (soft compression)'
            );
            this._resizeCharts(alphaChart, lossChart);

            const alphaTicker = new NumberTicker(ui.alphaValue, 4);
            const totalTicker = new NumberTicker(ui.totalLoss, 4);

            const trainer = new PerGraphCompressionTrainer(features, edges, {
                lambdaId: 0.6, lambdaComp: 0.4, lr: 0.14, initialAlpha: 0.5,
            });

            const learnedAlpha = await trainer.train(EPOCHS, (epoch, m) => {
                if (!this._running) return;
                console.log(`  [Epoch ${String(epoch + 1).padStart(3, ' ')}/${EPOCHS}] Total=${m.totalLoss.toFixed(4)}  ID=${m.identityLoss.toFixed(4)}  Comp=${m.compressionLoss.toFixed(4)}  α=${m.alpha.toFixed(4)}  ∇=${m.gradNorm.toFixed(6)}`);
                alphaChart.push([m.alpha]);
                lossChart.push([m.totalLoss, m.identityLoss, m.compressionLoss]);
                alphaTicker.to(m.alpha);
                totalTicker.to(m.totalLoss);
                ui.epochBadge.textContent = `Epoch ${epoch + 1} / ${EPOCHS}`;
            });

            if (!this._running) return;
            console.log(`\n  ✓ Learned α = ${learnedAlpha.toFixed(6)}`);
            alphaChart.setFinalAlpha(learnedAlpha);
            ui.statusText.textContent = `Training complete — learned α = ${learnedAlpha.toFixed(6)}`;
            await sleep(600);

            // ── Step 3: Soft compression (feature update, same topology) ─────────
            console.log(`\n%c============================================================`, 'color: #888');
            console.log(`%c  STEP 3/4 — Apply soft compression  (feature update, same graph topology)`, 'font-weight: bold; color: #fff');
            console.log(`%c============================================================`, 'color: #888');
            this._setStep(3, 'Applying soft compression…');
            const { featuresUpdated } = trainer.module.softCompress(features, edges);
            if (!this._running) return;

            // Re-draw graph with same structure but softened features (Python Step 3)
            this._renderer.init(numNodes, edges, new Set());
            ui.statusText.textContent = `Soft compression applied — α=${learnedAlpha.toFixed(4)} (same topology, updated features)`;
            await sleep(700);
            if (!this._running) return;

            // ── Step 4: Hard compression — step-by-step animation ───────────────
            console.log(`\n%c============================================================`, 'color: #888');
            console.log(`%c  STEP 4/4 — Hard compression  (node removal + edge merging, step by step)`, 'font-weight: bold; color: #fff');
            console.log(`%c============================================================`, 'color: #888');
            this._setStep(4, 'Hard compression (node removal…)');

            // Re-init renderer for the hard-compression phase
            // Step-0 will fire immediately from hardCompress via onStep
            this._renderer.init(numNodes, edges, new Set());

            const finalGraph = await hardCompress(
                featuresUpdated,
                edges,
                learnedAlpha,
                { T1: 0.9, T2: 0.85 },
                async (stepData) => {
                    if (!this._running) return;

                    if (stepData.step === 0) {
                        console.log(`\n%c============================================================`, 'color: #888');
                        console.log(`  STEP 0 — Original graph  (${stepData.numNodes} nodes)`);
                        console.log(`%c============================================================`, 'color: #888');
                        ui.statusText.textContent =
                            `Step 0 — Original graph (${stepData.numNodes} nodes)`;
                        return;
                    }

                    // Animate this removal step
                    await this._renderer.applyStep(stepData);

                    const { simUV, simPrev, simNext } = stepData.sims;
                    console.log(`\n%c============================================================`, 'color: #888');
                    console.log(`  STEP ${stepData.step} — Removed node ${stepData.removedU}, merged edges → node ${stepData.mergedToV}`);
                    if (stepData.addedEdges && stepData.addedEdges.length > 0) {
                        const addedStr = stepData.addedEdges.map(([a, b]) => `${a}→${b}`).join(', ');
                        console.log(`  [Edges Added] : ${addedStr}`);
                    }
                    console.log(`  sim(u,v)=${simUV.toFixed(3)}  sim(prev)=${simPrev.toFixed(3)}  sim(next)=${simNext.toFixed(3)}`);
                    console.log(`  Nodes remaining: ${stepData.remaining} / ${stepData.numNodes}`);
                    console.log(`%c============================================================`, 'color: #888');

                    this._stat('remaining', stepData.remaining);
                    this._stat('removed', stepData.numNodes - stepData.remaining);

                    ui.statusText.textContent =
                        `Step ${stepData.step}: removed node ${stepData.removedU} → ` +
                        `merged to node ${stepData.mergedToV} ` +
                        `(${stepData.remaining}/${stepData.numNodes} left) ` +
                        `| sim=${simUV.toFixed(3)} prev=${simPrev.toFixed(3)} next=${simNext.toFixed(3)}`;

                    await sleep(120); // pacing between steps
                }
            );

            if (!this._running) return;

            // ── Final result ─────────────────────────────────────────────────────
            console.log(`\n%c============================================================`, 'color: #888');
            console.log(`  FINAL — Hard compression complete`);
            console.log(`  Original nodes : ${numNodes}`);
            console.log(`  Remaining nodes: ${finalGraph.numKept}`);
            console.log(`  Compression    : ${(finalGraph.compressionRatio * 100).toFixed(1)}%`);
            console.log(`%c============================================================`, 'color: #888');

            console.log(`\n✅ Pipeline complete.`);
            console.log(`   Original nodes : ${numNodes}`);
            console.log(`   Final nodes    : ${finalGraph.numKept}`);
            console.log(`   Final edges    : ${finalGraph.edges.length}`);

            this._setStep(4, '✓ Hard compression complete');
            this._stat('remaining', finalGraph.numKept);
            this._stat('removed', numNodes - finalGraph.numKept);
            this._stat('ratio', (finalGraph.compressionRatio * 100).toFixed(1) + '%');

            ui.statusText.textContent =
                `✅ Pipeline complete — ${finalGraph.numKept}/${numNodes} nodes kept ` +
                `(${(finalGraph.compressionRatio * 100).toFixed(1)}% compression, ` +
                `α=${learnedAlpha.toFixed(4)})`;
            ui.completeBanner.classList.add('visible');

        } catch (err) {
            console.error('[Pipeline]', err);
            ui.statusText.textContent = `⚠ Error: ${err.message}`;
        } finally {
            this._running = false;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    _setStep(n, label) {
        this.ui.stepItems.forEach((el, i) => {
            el.classList.toggle('active', i === n - 1);
            el.classList.toggle('done', i < n - 1);
        });
        const lbl = this.ui.stepItems[n - 1]?.querySelector('.step-label');
        if (lbl) lbl.textContent = label;
    }

    _stat(key, val) {
        const el = this.ui.stats?.[key];
        if (el) el.textContent = val;
    }

    _resizeCharts(alphaChart, lossChart) {
        const pW = el => Math.floor((el.parentElement?.clientWidth || 420) * 0.96);
        const H = 200;
        alphaChart.resize(pW(this.ui.alphaCanvas), H);
        lossChart.resize(pW(this.ui.lossCanvas), H);
    }
}

export const sleep = ms => new Promise(r => setTimeout(r, ms));
