/**
 * graphRenderer.js — Animated SVG graph visualiser
 *
 * Faithfully mirrors Python visualize_compressed_graph():
 *
 *  pos = {i: (i, 0) for i in range(num_total_nodes)}
 *  → Node i is ALWAYS at x = MARGIN + i * SPACING. This never changes.
 *    Removed nodes simply aren't drawn — gaps are left in place.
 *
 *  Edge classification (on original node IDs, unchanged after merges):
 *    abs(u - v) == 1  →  gray straight arrow (original t→t+1)
 *    abs(u - v) >  1  →  crimson quadratic arc
 *
 *  Arc formula (mirrors Python FancyArrowPatch connectionstyle arc3):
 *    span = abs(v - u)      (node index distance, same as Python pos diff)
 *    rad  = -(0.3 + 0.015 * span)  if v > u
 *           +(0.3 + 0.015 * span)  if v < u
 *    Higher span → higher arc above the node row.
 *
 *  Node removal: fade + shrink IN PLACE (400ms). No re-layout.
 *  requestAnimationFrame render loop for smooth tweens.
 *  SVG only, no D3.
 */

'use strict';

const SVG_NS = 'http://www.w3.org/2000/svg';
const NODE_R = 18;
const MARGIN = 40;       // left offset
const NODE_Y = 110;      // fixed y for all nodes
const ANIM_MS = 420;      // node removal duration (ms)

function svgEl(tag, attrs = {}) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
}

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

// ─────────────────────────────────────────────────────────────────────────────
export class GraphRenderer {
    constructor(svg) {
        this.svg = svg;
        this.nodeEls = {};  // originalId → { g, circle, label }
        this.edgeEls = {};  // `${u},${v}` → SVGElement
        this._spacing = 60;
        this._numNodes = 0;
        this._removedSet = new Set();
        this._animations = [];
        this._rafId = null;
        this._startLoop();
    }

    // ── rAF animation loop ───────────────────────────────────────────────────
    _startLoop() {
        const tick = () => {
            const now = performance.now();
            this._animations = this._animations.filter(a => {
                const t = Math.min(1, (now - a.start) / a.dur);
                a.tick(easeOutCubic(t));
                if (t >= 1 && a.done) a.done();
                return t < 1;
            });
            this._rafId = requestAnimationFrame(tick);
        };
        this._rafId = requestAnimationFrame(tick);
    }

    _tween(dur, tick, done = null) {
        this._animations.push({ start: performance.now(), dur, tick, done });
    }

    // ── Fixed x-position for a node (never changes) ──────────────────────────
    _xOf(id) { return MARGIN + id * this._spacing; }

    // ── Build / re-build the entire graph view ───────────────────────────────
    /**
     * Call once at the start, or to reset to a new snapshot.
     * @param {number} numNodes   — total original node count
     * @param {Array<[number,number]>} edges  — current valid edges (original IDs)
     * @param {Set<number>} toRemove — nodes already removed in this snapshot
     */
    init(numNodes, edges, toRemove = new Set()) {
        this._numNodes = numNodes;
        this._removedSet = new Set(toRemove);
        this._animations = [];

        // Compute spacing so all nodes fit in ~90% of container width
        const containerW = this.svg.parentElement?.clientWidth || 800;
        this._spacing = Math.max(44, Math.min(72, (containerW * 0.92) / Math.max(numNodes, 1)));
        const totalW = MARGIN + numNodes * this._spacing + MARGIN;
        const totalH = NODE_Y * 2 + 20;

        this.svg.setAttribute('width', String(Math.max(300, totalW)));
        this.svg.setAttribute('height', String(totalH));
        this.svg.innerHTML = '';

        // SVG defs (arrowhead markers)
        GraphRenderer.injectDefs(this.svg);

        // Layers
        this._edgeLayer = svgEl('g', { id: 'edgeLayer' });
        this._nodeLayer = svgEl('g', { id: 'nodeLayer' });
        this.svg.appendChild(this._edgeLayer);
        this.svg.appendChild(this._nodeLayer);

        this.nodeEls = {};
        this.edgeEls = {};

        // Draw visible nodes
        for (let i = 0; i < numNodes; i++) {
            if (!toRemove.has(i)) this._createNode(i);
        }

        // Draw edges
        const seen = new Set();
        for (const [u, v] of edges) {
            if (toRemove.has(u) || toRemove.has(v)) continue;
            const key = `${u},${v}`;
            if (!seen.has(key)) { seen.add(key); this._drawEdge(u, v, false); }
        }
    }

    // ── Create a node at its fixed position ─────────────────────────────────
    _createNode(id, animated = false) {
        const x = this._xOf(id);
        const r = Math.max(9, Math.min(NODE_R, Math.floor(400 / this._numNodes)));
        const fs = Math.max(7, Math.min(13, Math.floor(260 / this._numNodes)));

        const g = svgEl('g', {
            transform: `translate(${x},${NODE_Y}) scale(${animated ? 0 : 1})`,
            opacity: animated ? 0 : 1,
        });

        const circle = svgEl('circle', {
            r, cx: 0, cy: 0,
            fill: 'var(--node-fill)',
            stroke: 'var(--node-stroke)',
            'stroke-width': 2,
        });
        const label = svgEl('text', {
            x: 0, y: fs * 0.38,
            'text-anchor': 'middle',
            'font-size': fs,
            fill: 'var(--node-label)',
            'font-family': "JetBrains Mono, monospace",
            'font-weight': 500,
        });
        label.textContent = String(id);

        g.appendChild(circle);
        g.appendChild(label);
        this._nodeLayer.appendChild(g);
        this.nodeEls[id] = { g, circle, label };
    }

    // ── Draw one edge ────────────────────────────────────────────────────────
    /**
     * @param {number} u
     * @param {number} v
     * @param {boolean} animated — fade in if true
     */
    _drawEdge(u, v, animated = false) {
        const key = `${u},${v}`;
        if (this.edgeEls[key]) return;  // already exists

        const x1 = this._xOf(u), x2 = this._xOf(v);
        const isOrig = Math.abs(u - v) === 1;   // original t→t+1 edge?

        let el;
        if (isOrig) {
            // Gray straight arrow
            const nodeR = Math.max(9, Math.min(NODE_R, Math.floor(400 / this._numNodes)));
            const dx = x2 > x1 ? nodeR : -nodeR;
            el = svgEl('line', {
                x1: x1 + dx, y1: NODE_Y,
                x2: x2 - dx, y2: NODE_Y,
                stroke: 'var(--edge-gray)',
                'stroke-width': 1.5,
                'marker-end': 'url(#arrowGray)',
            });
        } else {
            // Crimson curved arc
            // Python: span = abs(x2-x1) where pos[i]=(i,0) → span = abs(v-u)
            const span = Math.abs(v - u);
            // Convert Python rad logic: higher span -> larger arc height.
            const arcH = (0.3 + 0.015 * span) * this._spacing * 2.8;

            const isForward = v > u;
            const clampedH = Math.min(arcH, isForward ? NODE_Y - 10 : 80);
            const cx = (x1 + x2) / 2;
            const cy = isForward ? (NODE_Y - clampedH) : (NODE_Y + clampedH);

            el = svgEl('path', {
                d: `M${x1},${NODE_Y} Q${cx},${cy} ${x2},${NODE_Y}`,
                stroke: 'var(--edge-crimson)',
                'stroke-width': Math.max(1, Math.min(2, 18 / this._numNodes)),
                fill: 'none',
                'marker-end': 'url(#arrowCrimson)',
                opacity: animated ? 0 : 1,
            });

            if (animated) {
                // Fade in over 400ms
                this._tween(400, t => el.setAttribute('opacity', String(t)));
            }
        }

        this._edgeLayer.appendChild(el);
        this.edgeEls[key] = el;
    }

    // ── Apply one hard-compression step ─────────────────────────────────────
    /**
     * Mirrors one iteration of compress_graph_hard_with_vis():
     *  1. Fade + shrink removed node IN PLACE (no layout shift)
     *  2. Remove edges touching removed node
     *  3. Draw new merge edge(s) as animated crimson arcs
     *
     * @param {{ step, removedU, mergedToV, toRemove, edges }} stepData
     * @returns {Promise<void>}
     */
    async applyStep(stepData) {
        const { removedU, toRemove, edges } = stepData;
        if (removedU === null) return; // step 0

        // 1. Animate node removal (fade + shrink in-place)
        await this._animateRemoveNode(removedU);

        // 2. Sync edges: remove those touching removed nodes, add new ones
        this._syncEdges(edges, toRemove);
    }

    // ── Animate a node away (stays at original position, just disappears) ────
    _animateRemoveNode(id) {
        return new Promise(resolve => {
            const el = this.nodeEls[id];
            if (!el) { resolve(); return; }

            this._removedSet.add(id);
            const x = this._xOf(id);
            this._tween(ANIM_MS, t => {
                const s = 1 - t;
                el.g.setAttribute('transform', `translate(${x},${NODE_Y}) scale(${s})`);
                el.g.setAttribute('opacity', String(1 - t));
            }, () => {
                el.g.setAttribute('opacity', '0');
                el.g.setAttribute('transform', `translate(${x},${NODE_Y}) scale(0)`);
                resolve();
            });
            // Safety resolve
            setTimeout(resolve, ANIM_MS + 60);
        });
    }

    // ── Sync edges to match the current hard-compress state ─────────────────
    _syncEdges(currentEdges, toRemove) {
        // Remove SVG elements for edges that no longer exist or involve removed nodes
        const validKeys = new Set(
            currentEdges
                .filter(([a, b]) => !toRemove.has(a) && !toRemove.has(b))
                .map(([a, b]) => `${a},${b}`)
        );
        for (const key of Object.keys(this.edgeEls)) {
            if (!validKeys.has(key)) {
                this.edgeEls[key]?.remove();
                delete this.edgeEls[key];
            }
        }
        // Add new edges (animate crimson arcs in)
        for (const [u, v] of currentEdges) {
            if (toRemove.has(u) || toRemove.has(v)) continue;
            const key = `${u},${v}`;
            if (!this.edgeEls[key]) {
                this._drawEdge(u, v, true);  // animated = true for new merge edges
            }
        }
    }

    // ── Inject SVG arrowhead marker defs ────────────────────────────────────
    static injectDefs(svg) {
        if (svg.querySelector('defs')) return; // already injected
        const defs = svgEl('defs');

        const mkMarker = (id, colorVar) => {
            const m = svgEl('marker', {
                id, markerWidth: 7, markerHeight: 7,
                refX: 6, refY: 3, orient: 'auto', markerUnits: 'strokeWidth',
            });
            const poly = svgEl('polygon', {
                points: '0 0, 7 3, 0 6',
                fill: colorVar,
            });
            m.appendChild(poly);
            return m;
        };

        defs.appendChild(mkMarker('arrowGray', 'var(--edge-gray)'));
        defs.appendChild(mkMarker('arrowCrimson', 'var(--edge-crimson)'));
        svg.prepend(defs);
    }

    destroy() {
        if (this._rafId) cancelAnimationFrame(this._rafId);
        this._animations = [];
    }
}
