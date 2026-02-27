import streamlit as st
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
import time
import io
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1.  VISUALISATION HELPER
# ============================================================
def visualize_compressed_graph(data, title="Compressed Graph",
                                num_total_nodes=None, to_remove=None):
    G_nx = to_networkx(data, to_undirected=False)
    to_remove = to_remove or set()

    if num_total_nodes is None:
        num_total_nodes = data.num_nodes

    n = num_total_nodes
    fig_w      = max(12, n * 0.45) if n > 0 else 10
    fig_h      = max(3,  min(5, n * 0.12)) if n > 0 else 3
    node_size  = max(300,  min(800,  4000 // (n if n > 0 else 1)))
    font_size  = max(8,   min(10,   80   // (n if n > 0 else 1)))
    linewidth  = max(0.5, min(1.5,  20   // (n if n > 0 else 1)))
    arrow_mut  = max(6,   min(12,   120  // (n if n > 0 else 1)))

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
    return fig

# ============================================================
# 2.  GRAPH CONSTRUCTION
# ============================================================
@st.cache_data
def create_temporal_graph(audio_bytes, sr=16000, win_ms=50, hop_ms=100, n_mfcc=13):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
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

# ============================================================
# 3.  HELPER FUNCTIONS
# ============================================================
def get_neighbors(node, edge_index, num_nodes):
    mask = edge_index[0] == node
    neighbors = edge_index[1][mask].tolist()
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

# ============================================================
# 4.  SOFT COMPRESSION MODULE
# ============================================================
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

# ============================================================
# 5.  SPEAKER EMBEDDING NET
# ============================================================
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

# ============================================================
# 6.  STREAMLIT APP UI
# ============================================================
st.set_page_config(page_title="Graph Compression Visualisation", layout="wide")

st.title("🔊 Temporal Graph Compression Pipeline")
st.markdown("""
This app replicates the graph compression pipeline:
1. **Build Temporal Graph** from WAV (MFCC features).
2. **Learn Alpha** via Soft Compression.
3. **Hard Compression** with step-by-step visualisation.
""")

# --- Sidebar Configuration ---
st.sidebar.header("1. Data & Preprocessing")
uploaded_file = st.sidebar.file_uploader("Upload WAV file", type=["wav"])

win_ms = st.sidebar.slider("Window Size (ms)", 10, 200, 50)
hop_ms = st.sidebar.slider("Hop Size (ms)", 10, 200, 100)
n_mfcc = st.sidebar.slider("N MFCC", 5, 40, 13)

st.sidebar.header("2. Soft Compression Training")
epochs = st.sidebar.number_input("Epochs", 5, 200, 20)
lr     = st.sidebar.number_input("Learning Rate", 0.001, 1.0, 0.14, format="%.3f")
lambda_id   = st.sidebar.slider("Lambda Identity", 0.0, 1.0, 0.6)
lambda_comp = st.sidebar.slider("Lambda Compression", 0.0, 1.0, 0.4)

st.sidebar.header("3. Hard Compression")
tau1 = st.sidebar.slider("T1 (Primary Sim)", 0.5, 1.0, 0.9)
tau2 = st.sidebar.slider("T2 (Context Sim)", 0.5, 1.0, 0.85)

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    
    # --- Step 1: Build Graph ---
    st.header("Step 1/3: Build Temporal Graph")
    with st.spinner("Building graph from audio..."):
        graph = create_temporal_graph(audio_bytes, win_ms=win_ms, hop_ms=hop_ms, n_mfcc=n_mfcc)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", graph.num_nodes)
    col2.metric("Edges", graph.edge_index.shape[1])
    col3.metric("Feature Dim", graph.x.shape[1])
    
    with st.expander("Original Graph Visualisation"):
        fig_orig = visualize_compressed_graph(graph, title="Original Temporal Graph")
        st.pyplot(fig_orig)

    # --- Step 2: Training ---
    st.header("Step 2/3: Learn Alpha (Soft Compression)")
    if st.button("🚀 Start Training"):
        soft_compression  = SoftCompressionModule(initial_alpha=0.5)
        speaker_embedding = SpeakerEmbeddingNet(input_dim=graph.x.shape[1], output_dim=32)
        optimizer = optim.Adam(list(soft_compression.parameters()) + list(speaker_embedding.parameters()), lr=lr)

        loss_progress = st.empty()
        chart_col1, chart_col2 = st.columns(2)
        alpha_chart = chart_col1.empty()
        loss_chart  = chart_col2.empty()

        history = {'alpha': [], 'total': [], 'id': [], 'comp': []}

        prog_bar = st.progress(0)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            e_orig    = speaker_embedding(graph.x)
            data_comp = soft_compression.soft_compress(graph)
            e_comp    = speaker_embedding(data_comp.x)
            
            # Identity Loss
            cos = torch.dot(e_orig, e_comp) / (torch.norm(e_orig)*torch.norm(e_comp)+1e-8)
            loss_id = 1 - cos
            
            # Compression Loss
            if hasattr(data_comp, 'merge_probs') and len(data_comp.merge_probs) > 0:
                loss_comp = torch.mean(1.0 - data_comp.merge_probs)
            else:
                loss_comp = torch.tensor(0.0)
            
            loss_total = lambda_id * loss_id + lambda_comp * loss_comp
            loss_total.backward()
            optimizer.step()
            
            # Update history
            alpha_val = soft_compression.get_alpha().item()
            history['alpha'].append(alpha_val)
            history['total'].append(loss_total.item())
            history['id'].append(loss_id.item())
            history['comp'].append(loss_comp.item())

            # Update UI
            prog_bar.progress((epoch + 1) / epochs)
            loss_progress.write(f"**Epoch {epoch+1}/{epochs}** | Alpha: {alpha_val:.4f} | Loss: {loss_total.item():.4f}")
            
            alpha_chart.line_chart(history['alpha'], use_container_width=True)
            loss_chart.line_chart({k: history[k] for k in ['total', 'id', 'comp']}, use_container_width=True)

        st.session_state.learned_alpha = soft_compression.get_alpha().item()
        st.session_state.soft_module = soft_compression
        st.success(f"Training Complete! Learned α = {st.session_state.learned_alpha:.4f}")

    # --- Step 3: Hard Compression ---
    if 'learned_alpha' in st.session_state:
        st.header("Step 3/3: Hard Compression (Step-by-Step)")
        if st.button("🔥 Run Hard Compression"):
            with st.spinner("Preparing soft-compressed features..."):
                data_soft = st.session_state.soft_module.soft_compress(graph)
            
            x = data_soft.x
            edge_index = data_soft.edge_index.clone()
            num_nodes  = x.size(0)
            learned_alpha = st.session_state.learned_alpha

            edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
            to_remove = set()
            step_count = 0

            status_area = st.empty()
            plot_area   = st.empty()

            for u in range(1, num_nodes - 2):
                for v in range(num_nodes - 1, u, -1):
                    if v + 1 >= num_nodes or u - 1 < 0: continue
                    if any(n in to_remove for n in [u, v, u-1, v-1, u+1, v+1]): continue

                    sim_uv   = sim_n(u,   v,   x, edge_index, learned_alpha)
                    sim_prev = sim_n(u-1, v-1, x, edge_index, learned_alpha)
                    sim_next = sim_n(u+1, v+1, x, edge_index, learned_alpha)

                    if sim_uv >= tau1 and sim_prev >= tau2 and sim_next >= tau2:
                        step_count += 1
                        to_remove.add(u)

                        preds = predecessors_from_edges(edges, u)
                        succs = successors_from_edges(edges, u)
                        edges = [(a, b) for (a, b) in edges if a != u and b != u]
                        for p in preds:
                            if p != v and p not in to_remove: edges.append((p, v))
                        for s in succs:
                            if s != v and s not in to_remove: edges.append((v, s))
                        edges = list(set((a, b) for (a, b) in edges if a != b))

                        # Update Visualisation
                        status_area.write(f"**Step {step_count}**: Removed node {u}, merged to {v} (Nodes left: {num_nodes - len(to_remove)}/{num_nodes})")
                        
                        valid_edges = [(a, b) for (a, b) in edges if a not in to_remove and b not in to_remove]
                        ei_vis = (torch.tensor(valid_edges, dtype=torch.long).T if valid_edges else torch.empty((2, 0), dtype=torch.long))
                        data_vis = Data(x=x, edge_index=ei_vis)
                        
                        fig_step = visualize_compressed_graph(data_vis, title=f"Step {step_count} | Nodes: {num_nodes - len(to_remove)}", num_total_nodes=num_nodes, to_remove=to_remove.copy())
                        plot_area.pyplot(fig_step)
                        time.sleep(0.1)

            # Final Results
            st.divider()
            st.subheader("Final Compression Results")
            kept_nodes = sorted([i for i in range(num_nodes) if i not in to_remove])
            comp_rate = (1 - len(kept_nodes)/num_nodes)
            
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Final Nodes", len(kept_nodes))
            res_col2.metric("Compression %", f"{comp_rate:.1%}")
            res_col3.metric("Reduction", f"{num_nodes} → {len(kept_nodes)}")

else:
    st.info("Please upload a WAV file in the sidebar to begin.")
