/**
 * graph.js — Temporal graph construction
 * Mirrors create_synthetic_temporal_graph() exactly:
 *   - One node per frame (t = 0 … T-1)
 *   - Directed edges: t → t+1
 */

'use strict';

/**
 * Build a temporal graph from feature matrix.
 * @param {Array<Float32Array>} features — T × 52 feature vectors
 * @returns {{ nodes: Array<Float32Array>, edges: Array<[number,number]> }}
 */
export function buildTemporalGraph(features) {
    const T = features.length;
    const edges = [];
    for (let i = 0; i < T - 1; i++) {
        edges.push([i, i + 1]);
    }
    return {
        nodes: features,  // each is Float32Array(52)
        edges,            // [u, v] directed
        numNodes: T,
    };
}

/**
 * Clone graph state (deep copy of edges, shallow copy of nodes — features immutable).
 */
export function cloneGraph({ nodes, edges, numNodes }) {
    return {
        nodes,
        edges: edges.map(e => [e[0], e[1]]),
        numNodes,
    };
}
