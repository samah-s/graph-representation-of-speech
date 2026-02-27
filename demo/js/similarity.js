/**
 * similarity.js — Exact JS port of Python sim_n()
 *
 * Python:
 *   def sim_n(u, v, x, edge_index, alpha, beta=0.5):
 *       s_x = dot(x[u],x[v]) / (norm(x[u])*norm(x[v]) + 1e-8)
 *       nu = get_neighbors(u, edge_index, num_nodes)
 *       nv = get_neighbors(v, edge_index, num_nodes)
 *       if not nu or not nv: s_m = 0
 *       else:
 *           mu_u = mean(x[nu]), mu_v = mean(x[nv])
 *           s_m = dot(mu_u,mu_v)/(norm(mu_u)*norm(mu_v)+1e-8)
 *       return alpha * s_x + (1 - alpha) * s_m
 */

'use strict';

const EPS = 1e-8;

/** Dot product of two Float32Arrays */
export function dot(a, b) {
    const n = a.length;
    let s = 0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/** L2 norm */
export function norm(a) {
    return Math.sqrt(dot(a, a));
}

/** Cosine similarity in [−1, 1] */
export function cosine(a, b) {
    return dot(a, b) / (norm(a) * norm(b) + EPS);
}

/** 
 * Gradient of cosine similarity cos(A, B) with respect to A and B.
 * Returns { gradA, gradB } where each is a Float32Array.
 * If gCos is provided, it's a scalar dL/dCos to multiply results by.
 */
export function gradCosine(A, B, gCos = 1.0) {
    const nA = norm(A);
    const nB = norm(B);
    const dotVal = dot(A, B);
    const nAnB = nA * nB + EPS;

    const gradA = new Float32Array(A.length);
    const gradB = new Float32Array(B.length);

    // d(cos)/dA = (B * (nA*nB) - (A*B) * (nB * A/nA)) / (nA*nB)^2
    //           = (B * nA - cos * A) / (nA^2 * nB)
    for (let i = 0; i < A.length; i++) {
        gradA[i] = gCos * (B[i] * nA * nA - dotVal * A[i]) / (nA * nA * nA * nB + EPS);
        gradB[i] = gCos * (A[i] * nB * nB - dotVal * B[i]) / (nA * nB * nB * nB + EPS);
    }
    return { gradA, gradB };
}

/**
 * Get all neighbours (in + out) of `node` from edge list.
 * Mirrors Python get_neighbors() — both predecessors and successors.
 */
export function getNeighbours(node, edges) {
    const nbrs = new Set();
    for (const [u, v] of edges) {
        if (u === node) nbrs.add(v);
        if (v === node) nbrs.add(u);
    }
    return [...nbrs];
}

/**
 * Mean of a set of feature vectors.
 * @param {Array<number>} indices
 * @param {Array<Float32Array>} features
 */
export function meanVec(indices, features) {
    const dim = features[0].length;
    const out = new Float32Array(dim);
    for (const idx of indices) {
        const f = features[idx];
        for (let d = 0; d < dim; d++) out[d] += f[d];
    }
    const n = indices.length;
    for (let d = 0; d < dim; d++) out[d] /= n;
    return out;
}

export function simNComponents(u, v, features, edges) {
    const s_x = cosine(features[u], features[v]);
    const nu = getNeighbours(u, edges);
    const nv = getNeighbours(v, edges);
    let s_m = 0;
    if (nu.length > 0 && nv.length > 0) {
        const mu_u = meanVec(nu, features);
        const mu_v = meanVec(nv, features);
        s_m = cosine(mu_u, mu_v);
    }
    return { s_x, s_m, u, v, nu, nv };
}

/**
 * sim_n(u, v, features, edges, alpha) — exact port of Python sim_n()
 * @param {number} u
 * @param {number} v
 * @param {Array<Float32Array>} features
 * @param {Array<[number,number]>} edges
 * @param {number} alpha  — feature weight in [0,1]
 * @returns {number}
 */
export function simN(u, v, features, edges, alpha) {
    const { s_x, s_m } = simNComponents(u, v, features, edges);
    return alpha * s_x + (1 - alpha) * s_m;
}

/**
 * Predecessors of `node` in directed edge list.
 */
export function predecessors(edges, node) {
    return edges.filter(([a, b]) => b === node).map(([a]) => a);
}

/**
 * Successors of `node` in directed edge list.
 */
export function successors(edges, node) {
    return edges.filter(([a, b]) => a === node).map(([, b]) => b);
}
