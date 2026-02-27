# ============================================================
# FULL PIPELINE: WAV → Temporal Graph → Learn Alpha (Soft) →
#                Hard Compression with step-by-step visualisation
# ============================================================
# Drop your .wav file path into WAV_PATH below, then run.
# ============================================================



!pip install torch torchvision torchaudio
!pip install torch-geometric
!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
!pip install librosa
!pip install networkx
!pip install matplotlib
!pip install numpy
!pip install soundfile


WAV_PATH = "arctic_a0591.wav"   # <-- change this

# ──────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ──────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch



# ──────────────────────────────────────────────────────────────
# 1.  VISUALISATION HELPER  (unchanged from your specification)
# ──────────────────────────────────────────────────────────────
def visualize_compressed_graph(data, title="Compressed Graph",
                                num_total_nodes=None, to_remove=None):
    G_nx = to_networkx(data, to_undirected=False)
    to_remove = to_remove or set()

    if num_total_nodes is None:
        num_total_nodes = data.num_nodes

    # ── Dynamic sizing based on node count ──
    n = num_total_nodes
    fig_w      = max(16, n * 0.55)        # wider for more nodes
    fig_h      = max(3,  min(5, n * 0.15))
    node_size  = max(500,  min(500,  4000 // n))
    font_size  = max(9,   min(10,   80   // n))
    linewidth  = max(0.5, min(2.0,  20   // n))
    arrow_mut  = max(6,   min(15,   120  // n))

    pos = {i: (i, 0) for i in range(num_total_nodes)}

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    visible_nodes = [nd for nd in G_nx.nodes if nd not in to_remove]
    nx.draw_networkx_nodes(G_nx, pos, nodelist=visible_nodes,
                           node_color='skyblue', node_size=node_size, ax=ax)
    nx.draw_networkx_labels(G_nx, pos,
                            labels={nd: nd for nd in visible_nodes},
                            font_size=font_size, ax=ax)

    original_edges, added_edges = [], []
    for u, v in G_nx.edges:
        if u in to_remove or v in to_remove:
            continue
        if abs(u - v) == 1:
            original_edges.append((u, v))
        else:
            added_edges.append((u, v))

    nx.draw_networkx_edges(G_nx, pos, edgelist=original_edges,
                           edge_color='gray', arrows=True,
                           width=linewidth,
                           arrowsize=max(8, arrow_mut),
                           ax=ax)

    for u, v in added_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # Scale arc radius by distance so long-range edges arc more
        span = abs(x2 - x1)
        rad  = -(0.3 + 0.015 * span) if x2 > x1 else (0.3 + 0.015 * span)
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle='->',
            color='crimson',
            linewidth=linewidth,
            mutation_scale=arrow_mut
        )
        ax.add_patch(arrow)

    ax.set_title(title, fontsize=max(7, min(13, font_size + 3)), pad=8)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────
# 2.  GRAPH CONSTRUCTION  (from notebook)
# ──────────────────────────────────────────────────────────────
def create_synthetic_temporal_graph(file_path, sr=16000,
                                    win_ms=50, hop_ms=100, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    win_length = int(sr * win_ms / 1000)
    hop_length  = int(sr * hop_ms / 1000)

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                   n_fft=win_length,
                                   hop_length=hop_length,
                                   win_length=win_length)
    delta  = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    T = mfcc.shape[1]

    node_features = []
    for t in range(T):
        x_t = np.concatenate([
            np.mean(mfcc[:,  t:t+1], axis=1),
            np.std( mfcc[:,  t:t+1], axis=1),
            np.mean(delta[:, t:t+1], axis=1),
            np.mean(delta2[:,t:t+1], axis=1),
        ])
        node_features.append(x_t)

    x          = torch.tensor(np.stack(node_features), dtype=torch.float)
    edge_index = torch.tensor([[i, i+1] for i in range(T-1)],
                               dtype=torch.long).T
    return Data(x=x, edge_index=edge_index)


# ──────────────────────────────────────────────────────────────
# 3.  HELPER FUNCTIONS  (from notebook)
# ──────────────────────────────────────────────────────────────
def get_neighbors(node, edge_index, num_nodes):
    neighbors = []
    mask = edge_index[0] == node
    neighbors.extend(edge_index[1][mask].tolist())
    mask = edge_index[1] == node
    neighbors.extend(edge_index[0][mask].tolist())
    return list(set(neighbors))


def sim_n(u, v, x, edge_index, alpha, beta=0.5):
    num_nodes = x.size(0)
    x_u, x_v = x[u], x[v]
    s_x = torch.dot(x_u, x_v) / (torch.norm(x_u) * torch.norm(x_v) + 1e-8)
    nu = get_neighbors(u, edge_index, num_nodes)
    nv = get_neighbors(v, edge_index, num_nodes)
    if not nu or not nv:
        s_m = torch.tensor(0.0)
    else:
        mu_u = torch.mean(x[nu], dim=0)
        mu_v = torch.mean(x[nv], dim=0)
        s_m  = torch.dot(mu_u, mu_v) / (torch.norm(mu_u)*torch.norm(mu_v)+1e-8)
    return alpha * s_x + (1 - alpha) * s_m


def predecessors_from_edges(edges, node):
    return {a for (a, b) in edges if b == node}

def successors_from_edges(edges, node):
    return {b for (a, b) in edges if a == node}


# ──────────────────────────────────────────────────────────────
# 4.  SOFT COMPRESSION MODULE  (from notebook, unchanged)
# ──────────────────────────────────────────────────────────────
class SoftCompressionModule(nn.Module):
    def __init__(self, initial_alpha=0.5, initial_temperature=0.1):
        super().__init__()
        a_raw_init = 0.0 if initial_alpha == 0.5 \
                     else -np.log(1.0 / initial_alpha - 1.0)
        self.a_raw           = nn.Parameter(torch.tensor(a_raw_init, dtype=torch.float32))
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature)))
        self.merge_scale     = nn.Parameter(torch.tensor(1.0))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def get_alpha(self):
        return torch.sigmoid(self.a_raw)

    def compute_similarity(self, u, v, x, edge_index):
        alpha     = self.get_alpha()
        num_nodes = x.size(0)
        x_u, x_v = x[u], x[v]
        s_x = torch.dot(x_u, x_v) / (torch.norm(x_u)*torch.norm(x_v)+1e-8)
        nu = get_neighbors(u, edge_index, num_nodes)
        nv = get_neighbors(v, edge_index, num_nodes)
        if not nu or not nv:
            s_m = torch.tensor(0.0, device=x.device)
        else:
            mu_u = torch.mean(x[nu], dim=0)
            mu_v = torch.mean(x[nv], dim=0)
            s_m  = torch.dot(mu_u, mu_v)/(torch.norm(mu_u)*torch.norm(mu_v)+1e-8)
        return alpha * s_x + (1 - alpha) * s_m

    def compute_merge_probability(self, t, k, x, edge_index, tau1, tau2):
        num_nodes = x.size(0)
        if (t+k >= num_nodes or t-1 < 0 or t+k-1 >= num_nodes or
                t+1 >= num_nodes or t+k+1 >= num_nodes):
            return torch.tensor(0.0, device=x.device)
        s1 = self.compute_similarity(t,   t+k,   x, edge_index)
        s2 = self.compute_similarity(t-1, t+k-1, x, edge_index)
        s3 = self.compute_similarity(t+1, t+k+1, x, edge_index)
        p1 = torch.sigmoid((s1 - tau1) / self.temperature)
        p2 = torch.sigmoid((s2 - tau2) / self.temperature)
        p3 = torch.sigmoid((s3 - tau2) / self.temperature)
        return p1 * p2 * p3

    def soft_compress(self, data, tau1=None, tau2=None, window_size=10):
        x          = data.x.clone()
        edge_index = data.edge_index
        num_nodes  = x.size(0)
        x_updated  = x.clone()
        all_merge_probs = []

        for t in range(1, num_nodes - 2):
            similarities = []
            for k in range(1, num_nodes - t):
                if t + k >= num_nodes:
                    break
                similarities.append(self.compute_similarity(t, t+k, x, edge_index))

            if similarities:
                sim_stack   = torch.stack(similarities)
                tau1_local  = torch.quantile(sim_stack, 0.9)
                tau2_local  = torch.quantile(sim_stack, 0.85)
                best_p, best_k = None, None

                for k_idx, k in enumerate(range(1, num_nodes - t)):
                    if t + k >= num_nodes:
                        break
                    p = self.compute_merge_probability(t, k, x, edge_index,
                                                       tau1_local, tau2_local)
                    if best_p is None or p > best_p:
                        best_p, best_k = p, k

                if best_k is not None:
                    all_merge_probs.append(best_p)
                    strength = torch.clamp(best_p * 1.2, 0.0, 1.0)
                    x_updated[t] = ((1 - strength) * x[t] +
                                    strength * (x[t] + x[t + best_k]) / 2)

        data_compressed = Data(x=x_updated, edge_index=edge_index)
        data_compressed.merge_probs = (torch.stack(all_merge_probs)
                                       if all_merge_probs
                                       else torch.tensor([], device=x.device))
        return data_compressed


# ──────────────────────────────────────────────────────────────
# 5.  SPEAKER EMBEDDING NET  (from notebook, unchanged)
# ──────────────────────────────────────────────────────────────
class SpeakerEmbeddingNet(nn.Module):
    def __init__(self, input_dim=52, hidden_dim=64, output_dim=32):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.attention   = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=0))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_features):
        nf = self.node_encoder(graph_features)
        aw = self.attention(nf)
        return self.output_layer(torch.sum(aw * nf, dim=0))


# ──────────────────────────────────────────────────────────────
# 6.  PER-GRAPH COMPRESSION TRAINER  (from notebook, unchanged)
# ──────────────────────────────────────────────────────────────
class PerGraphCompressionTrainer:
    def __init__(self, data, feature_dim=52, embedding_dim=32,
                 lambda_id=0.6, lambda_comp=0.4,
                 learning_rate=0.14, initial_alpha=0.5):
        self.data              = data
        self.soft_compression  = SoftCompressionModule(initial_alpha=initial_alpha)
        self.speaker_embedding = SpeakerEmbeddingNet(input_dim=feature_dim,
                                                      output_dim=embedding_dim)
        self.lambda_id   = lambda_id
        self.lambda_comp = lambda_comp
        self.optimizer   = optim.Adam(
            list(self.soft_compression.parameters()) +
            list(self.speaker_embedding.parameters()),
            lr=learning_rate
        )
        self.loss_history = {'total': [], 'identity': [], 'compression': [],
                             'alpha': [], 'gradient_norm': []}

    def compute_identity_loss(self, e_orig, e_comp):
        cos = torch.dot(e_orig, e_comp) / (torch.norm(e_orig)*torch.norm(e_comp)+1e-8)
        return 1 - cos

    def compute_compression_loss(self, compressed_data):
        if not hasattr(compressed_data, 'merge_probs') or \
                len(compressed_data.merge_probs) == 0:
            return torch.tensor(0.0, device=self.data.x.device)
        return torch.mean(1.0 - compressed_data.merge_probs)

    def train_step(self, tau1=0.9, tau2=0.85, window_size=10):
        self.optimizer.zero_grad()
        e_orig       = self.speaker_embedding(self.data.x)
        data_comp    = self.soft_compression.soft_compress(self.data, tau1, tau2, window_size)
        e_comp       = self.speaker_embedding(data_comp.x)
        loss_id      = self.compute_identity_loss(e_orig, e_comp)
        loss_comp    = self.compute_compression_loss(data_comp)
        loss_total   = self.lambda_id * loss_id + self.lambda_comp * loss_comp
        loss_total.backward()
        grad_norm    = (self.soft_compression.a_raw.grad.norm().item()
                        if self.soft_compression.a_raw.grad is not None else 0.0)
        self.optimizer.step()
        alpha = self.soft_compression.get_alpha().item()
        return {'total_loss': loss_total.item(), 'identity_loss': loss_id.item(),
                'compression_loss': loss_comp.item(), 'alpha': alpha,
                'gradient_norm': grad_norm}

    def train(self, epochs=100, tau1=0.9, tau2=0.85, window_size=10, verbose=True):
        for epoch in range(epochs):
            m = self.train_step(tau1, tau2, window_size)
            for k in ('total','identity','compression','alpha','gradient_norm'):
                self.loss_history[k].append(
                    m[k if k != 'gradient_norm' else 'gradient_norm']
                    if k in m else m.get(f'{k}_loss', m.get(k)))
            if verbose:
                print(f"  [Epoch {epoch+1:3d}/{epochs}] "
                      f"Total={m['total_loss']:.4f}  "
                      f"ID={m['identity_loss']:.4f}  "
                      f"Comp={m['compression_loss']:.4f}  "
                      f"α={m['alpha']:.4f}  "
                      f"∇={m['gradient_norm']:.6f}")
        return self.soft_compression.get_alpha().item()

    def get_learned_alpha(self):
        return self.soft_compression.get_alpha().item()


# ──────────────────────────────────────────────────────────────
# 7.  HARD COMPRESSION  (from notebook) + step-by-step vis
# ──────────────────────────────────────────────────────────────
def compress_graph_hard_with_vis(data, learned_alpha,
                                  T1=0.9, T2=0.85, window_size=10):
    """
    Runs hard compression exactly as in the notebook and calls
    visualize_compressed_graph after EVERY node removal + edge merge step.
    Returns the final compressed Data object.
    """
    x          = data.x
    edge_index = data.edge_index.clone()
    num_nodes  = x.size(0)
    window_size = num_nodes - 1          # match notebook

    edges     = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    to_remove = set()
    step      = 0

    # ── helper: build a Data object reflecting the *current* state ──
    def current_data():
        all_nodes = list(range(num_nodes))
        kept      = [n for n in all_nodes if n not in to_remove]
        # Re-index edges using original ids (some may point to removed nodes)
        valid_edges = [(a, b) for (a, b) in edges
                       if a not in to_remove and b not in to_remove]
        if valid_edges:
            ei = torch.tensor(valid_edges, dtype=torch.long).T
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
        return Data(x=x[kept], edge_index=ei), kept, valid_edges

    # Show original graph
    print(f"\n{'='*60}")
    print(f"  STEP 0 — Original graph  ({num_nodes} nodes)")
    print(f"{'='*60}")
    visualize_compressed_graph(data, title=f"Step 0 — Original graph ({num_nodes} nodes)",
                                num_total_nodes=num_nodes, to_remove=set())

    for u in range(1, num_nodes - 2):
        for v in range(num_nodes - 1, u, -1):

            if v + 1 >= num_nodes or u - 1 < 0:
                continue
            if any(n in to_remove for n in [u, v, u-1, v-1, u+1, v+1]):
                continue

            sim_uv   = sim_n(u,   v,   x, edge_index, learned_alpha, beta=0.5)
            sim_prev = sim_n(u-1, v-1, x, edge_index, learned_alpha, beta=0.5)
            sim_next = sim_n(u+1, v+1, x, edge_index, learned_alpha, beta=0.5)

            if sim_uv >= T1 and sim_prev >= T2 and sim_next >= T2:
                step += 1
                to_remove.add(u)

                preds = predecessors_from_edges(edges, u)
                succs = successors_from_edges(edges, u)

                edges = [(a, b) for (a, b) in edges if a != u and b != u]

                for p in preds:
                    if p != v and p not in to_remove:
                        edges.append((p, v))
                for s in succs:
                    if s != v and s not in to_remove:
                        edges.append((v, s))

                edges = list(set((a, b) for (a, b) in edges if a != b))

                # ── Visualise this step ──
                remaining = num_nodes - len(to_remove)
                print(f"\n{'='*60}")
                print(f"  STEP {step} — Removed node {u}, merged edges → node {v}")
                print(f"  sim(u,v)={sim_uv:.3f}  sim(prev)={sim_prev:.3f}  "
                      f"sim(next)={sim_next:.3f}")
                print(f"  Nodes remaining: {remaining} / {num_nodes}")
                print(f"{'='*60}")

                # Build a view of the CURRENT state for visualisation
                valid_edges = [(a, b) for (a, b) in edges
                               if a not in to_remove and b not in to_remove]
                ei_vis = (torch.tensor(valid_edges, dtype=torch.long).T
                          if valid_edges
                          else torch.empty((2, 0), dtype=torch.long))
                # Use the full original node set but mark removed nodes
                data_vis = Data(x=x, edge_index=ei_vis)
                visualize_compressed_graph(
                    data_vis,
                    title=(f"Step {step} — Removed node {u}, merged to {v}  "
                           f"({remaining} nodes left)"),
                    num_total_nodes=num_nodes,
                    to_remove=to_remove.copy()
                )

    # ── Final compressed graph (re-indexed) ──
    kept_nodes  = sorted([i for i in range(num_nodes) if i not in to_remove])
    node_mapping = {old: new for new, old in enumerate(kept_nodes)}
    x_new        = x[kept_nodes]
    filtered     = [(node_mapping[a], node_mapping[b])
                    for (a, b) in edges
                    if a in node_mapping and b in node_mapping]
    ei_new = (torch.tensor(filtered, dtype=torch.long).T
              if filtered else torch.empty((2, 0), dtype=torch.long))
    final_data = Data(x=x_new, edge_index=ei_new)

    print(f"\n{'='*60}")
    print(f"  FINAL — Hard compression complete")
    print(f"  Original nodes : {num_nodes}")
    print(f"  Remaining nodes: {len(kept_nodes)}")
    print(f"  Compression    : {1 - len(kept_nodes)/num_nodes:.1%}")
    print(f"{'='*60}")
    visualize_compressed_graph(final_data,
                                title=(f"Final compressed graph — "
                                       f"{len(kept_nodes)}/{num_nodes} nodes kept "
                                       f"({1-len(kept_nodes)/num_nodes:.0%} removed)"))
    return final_data


# ──────────────────────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1/4 — Build temporal graph from WAV")
print("=" * 60)
graph = create_synthetic_temporal_graph(WAV_PATH)
feature_dim = graph.x.shape[1]
print(f"  Nodes     : {graph.num_nodes}")
print(f"  Edges     : {graph.edge_index.shape[1]}")
print(f"  Feature dim: {feature_dim}")

# Visualise the original graph before any compression
visualize_compressed_graph(graph, title="Original Temporal Graph (pre-compression)")

# ── If the graph is huge, cap it for visualisation clarity ──
# MAX_VIS_NODES = 30
# if graph.num_nodes > MAX_VIS_NODES:
#     print(f"\n  [INFO] Graph has {graph.num_nodes} nodes — "
#           f"capping at {MAX_VIS_NODES} nodes for visualisation clarity.")
#     graph = Data(x=graph.x[:MAX_VIS_NODES],
#                  edge_index=torch.tensor(
#                      [[i, i+1] for i in range(MAX_VIS_NODES-1)],
#                      dtype=torch.long).T)
#     print(f"  Capped graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

# No capping — visualisation scales automatically to all nodes
print(f"\n  [INFO] Graph has {graph.num_nodes} nodes — visualising all nodes.")

print("\n" + "=" * 60)
print("  STEP 2/4 — Learn alpha via soft compression")
print("=" * 60)

trainer = PerGraphCompressionTrainer(
    data=graph,
    feature_dim=feature_dim,
    lambda_id=0.6,
    lambda_comp=0.4,
    learning_rate=0.14,
    initial_alpha=0.5
)

EPOCHS = 20   # increase for a real run; 20 is fine for a demo
learned_alpha = trainer.train(epochs=EPOCHS, verbose=True)
print(f"\n  ✓ Learned α = {learned_alpha:.6f}")

# Plot alpha + loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(trainer.loss_history['alpha'], color='steelblue', linewidth=2)
axes[0].set_title("α (alpha) over soft-compression epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("α")
axes[0].axhline(learned_alpha, color='crimson', linestyle='--',
                label=f"Final α={learned_alpha:.4f}")
axes[0].legend()

axes[1].plot(trainer.loss_history['total'],       label="Total",       linewidth=2)
axes[1].plot(trainer.loss_history['identity'],    label="Identity",    linewidth=1.5)
axes[1].plot(trainer.loss_history['compression'], label="Compression", linewidth=1.5)
axes[1].set_title("Loss curves (soft compression training)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend()
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("  STEP 3/4 — Apply soft compression  (feature update, same graph topology)")
print("=" * 60)
soft_module = trainer.soft_compression
with torch.no_grad():
    data_soft = soft_module.soft_compress(graph)
visualize_compressed_graph(data_soft,
                            title=f"After Soft Compression (α={learned_alpha:.4f})")

print("\n" + "=" * 60)
print("  STEP 4/4 — Hard compression  (node removal + edge merging, step by step)")
print("=" * 60)
final_graph = compress_graph_hard_with_vis(
    data_soft,          # apply hard compression on the soft-compressed features
    learned_alpha=learned_alpha,
    T1=0.9,
    T2=0.85,
    window_size=10
)

print("\n✅ Pipeline complete.")
print(f"   Original nodes : {graph.num_nodes}")
print(f"   Final nodes    : {final_graph.num_nodes}")
print(f"   Final edges    : {final_graph.edge_index.shape[1]}")