/**
 * hardCompress.js — Exact JS port of compress_graph_hard_with_vis()
 */

'use strict';

import { simN, predecessors, successors } from './similarity.js';

/**
 * @param {Array<Float32Array>} features   — node features (after soft-compress)
 * @param {Array<[number,number]>} edges0  — initial directed edge list
 * @param {number} learnedAlpha
 * @param {{ T1?, T2? }} opts
 * @param {function} onStep — async callback(stepData) on each visualisation step
 * @returns {Promise<{features, edges, numKept, numTotal, compressionRatio, keptNodes}>}
 */
export async function hardCompress(features, edges0, learnedAlpha, opts = {}, onStep = null) {
    const T1 = opts.T1 ?? 0.9;
    const T2 = opts.T2 ?? 0.85;
    const N = features.length;

    let edges = edges0.map(e => [e[0], e[1]]); // mutable copy of edges
    const toRemove = new Set();
    let step = 0;

    // ── Yield a step snapshot to the visualiser ──────────────────────────────
    async function fireStep(removedU, mergedToV, sims, addedEdges = []) {
        step++;
        const remaining = N - toRemove.size;
        const validEdges = edges.filter(([a, b]) => !toRemove.has(a) && !toRemove.has(b));
        if (onStep) {
            await onStep({
                step,
                removedU,
                mergedToV,
                remaining,
                numNodes: N,
                toRemove: new Set(toRemove),
                edges: validEdges.map(e => [e[0], e[1]]),
                sims,
                addedEdges,
            });
        }
        await new Promise(r => setTimeout(r, 0));
    }

    // ── Step 0 — original graph ───────────────────────────────────────────────
    if (onStep) {
        await onStep({
            step: 0,
            removedU: null,
            mergedToV: null,
            remaining: N,
            numNodes: N,
            toRemove: new Set(),
            edges: edges.map(e => [e[0], e[1]]),
            sims: null,
        });
        await new Promise(r => setTimeout(r, 0));
    }

    // ── Main compression loop — mirrors Python exactly ────────────────────────
    for (let u = 1; u <= N - 3; u++) {
        // Inner: v from N-1 down to u+1 (range(num_nodes-1, u, -1))
        for (let v = N - 1; v > u; v--) {
            // Python bounds guard: if v+1>=N or u-1<0: continue
            if (v + 1 >= N || u - 1 < 0) continue;

            // Python: if any(n in to_remove for n in [u,v,u-1,v-1,u+1,v+1]): continue
            if ([u, v, u - 1, v - 1, u + 1, v + 1].some(n => toRemove.has(n))) continue;

            // Python uses fixed edge_index for similarity calculation
            const simUV = simN(u, v, features, edges0, learnedAlpha);
            const simPrev = simN(u - 1, v - 1, features, edges0, learnedAlpha);
            const simNext = simN(u + 1, v + 1, features, edges0, learnedAlpha);

            if (simUV >= T1 && simPrev >= T2 && simNext >= T2) {
                toRemove.add(u);

                // ── Edge rerouting (uses evolving edges list) ────────────────────
                const preds = predecessors(edges, u);
                const succs = successors(edges, u);
                const addedForStep = [];

                // Remove all edges touching u
                edges = edges.filter(([a, b]) => a !== u && b !== u);

                // Redirect predecessors of u → v
                for (const p of preds) {
                    if (p !== v && !toRemove.has(p)) {
                        edges.push([p, v]);
                        addedForStep.push([p, v]);
                    }
                }
                // Redirect successors of u → from v
                for (const s of succs) {
                    if (s !== v && !toRemove.has(s)) {
                        edges.push([v, s]);
                        addedForStep.push([v, s]);
                    }
                }

                // Deduplicate + remove self-loops
                const seen = new Set();
                edges = edges.filter(([a, b]) => {
                    if (a === b) return false;
                    const key = `${a},${b}`;
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                });

                // Yield visualisation step
                await fireStep(u, v, { simUV, simPrev, simNext }, addedForStep);
                // Note: v-loop continues, but all future v for this u will hit the toRemove.has(u) guard
            }
        }
    }

    // ── Final re-indexing (mirrors Python exactly) ────────────────────────────
    const keptNodes = Array.from({ length: N }, (_, i) => i)
        .filter(i => !toRemove.has(i));

    const nodeMap = new Map(keptNodes.map((old, newIdx) => [old, newIdx]));
    const finalFeats = keptNodes.map(i => features[i]);
    const finalEdges = edges
        .filter(([a, b]) => nodeMap.has(a) && nodeMap.has(b))
        .map(([a, b]) => [nodeMap.get(a), nodeMap.get(b)]);

    return {
        features: finalFeats,
        edges: finalEdges,
        numKept: keptNodes.length,
        numTotal: N,
        compressionRatio: 1 - keptNodes.length / N,
        keptNodes,
    };
}
