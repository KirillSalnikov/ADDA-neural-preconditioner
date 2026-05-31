"""NeuralPrecond — GNN-based callable preconditioner M(v) for BiCGStab.

Two-phase architecture:
  Phase 1: encode_geometry(data) — one-time encoding of the geometry graph.
    Identical encoder+processor as NeuralInitGuess, plus lightweight projections
    for the apply phase (node_proj, edge_gate).

  Phase 2: apply_precond(cache, v) — called ~400 times per solve.
    Takes cached geometry info + vector v, applies a single lightweight MP step,
    returns M(v) = v + correction (residual learning).

The apply phase is designed to be fast (~0.04ms on GPU) with minimal parameters
(~5K in apply path) since it runs inside the Krylov loop.

Key design:
  - encode_geometry() is expensive (encoder + 6 GraphNet layers) but runs once
  - apply_precond() is cheap (1 MLP + 1 MP step + 1 linear) and runs per iteration
  - Residual learning: M(v) = v + correction ensures M starts as identity
  - edge_gate (sigmoid) learns which edges are important for preconditioning

Also includes:
  - PolyPrecondMLP — MLP-based polynomial preconditioner
  - ConvSAI_MLP — MLP-based convolutional SAI preconditioner (FFT apply, 1 matvec cost)
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch_geometric.nn import aggr

from core.models import GraphNet, MLP


class NeuralPrecond(nn.Module):
    """Neural preconditioner M(v) for iterative DDA solver.

    Architecture:
      - Encoder: MLPs for node (9), edge (8), global (4) features -> latent L
      - Processor: K GraphNet layers with edge skip connections
      - Apply-phase projections: node_proj (L -> L_apply), edge_gate (L -> L_apply)
      - Apply-phase network: input MLP + 1 MP step + output linear

    encode_geometry() returns a cache dict — run once per geometry.
    apply_precond() uses cache + vector v — run per Krylov iteration.
    """

    def __init__(self, latent_size=64, latent_size_apply=32,
                 message_passing_steps=6, apply_mp_steps=1,
                 node_features_in=9, edge_features_in=8, global_features_in=4,
                 activation="relu", use_checkpoint=False):
        super().__init__()

        L = latent_size
        La = latent_size_apply
        self.latent_size = L
        self.latent_size_apply = La
        self.message_passing_steps = message_passing_steps
        self.apply_mp_steps = apply_mp_steps
        self.use_checkpoint = use_checkpoint
        self.node_features_in = node_features_in
        self.edge_features_in = edge_features_in
        self.global_features_in = global_features_in

        # ── Encoder (same as NeuralInitGuess) ──────────────
        self.node_enc = MLP([node_features_in, L, L], activation=activation)
        self.edge_enc = MLP([edge_features_in, L, L], activation=activation)
        self.global_enc = MLP([global_features_in, L, L], activation=activation)

        # ── Processor: K GraphNet layers with skip connections ─
        self.processor = nn.ModuleList([
            GraphNet(node_features=L, edge_features=L, global_features=L,
                     hidden_size=L, aggregate="mean", activation=activation)
            for _ in range(message_passing_steps)
        ])
        self.edge_skip_proj = nn.ModuleList([
            nn.Linear(2 * L, L)
            for _ in range(message_passing_steps)
        ])

        # ── Apply-phase projections (cached after encode) ──
        self.node_proj = nn.Linear(L, La)
        self.edge_gate_proj = nn.Sequential(
            nn.Linear(L, La),
            nn.Sigmoid(),
        )

        # ── Apply-phase network (runs per Krylov iteration) ──
        # Input: [node_cache (La) || v_re (1) || v_im (1)] -> La
        self.apply_input = MLP([La + 2, La, La], activation=activation)

        # Lightweight MP steps for the apply phase
        self.apply_aggregate = aggr.MeanAggregation()
        self.apply_mp = nn.ModuleList()
        for _ in range(apply_mp_steps):
            self.apply_mp.append(nn.ModuleDict({
                'msg_proj': nn.Linear(La, La),
                'upd': MLP([La + La, La, La], activation=activation),
            }))

        # Output: La -> 2 (correction re, im)
        self.apply_output = nn.Linear(La, 2)

        # Initialize output to near-zero so M(v) ≈ v initially
        nn.init.zeros_(self.apply_output.weight)
        nn.init.zeros_(self.apply_output.bias)

    def encode_geometry(self, data):
        """Encode geometry graph into cached representations for apply_precond.

        This is the expensive part (full message passing). Call once per geometry.

        Args:
            data: torch_geometric.Data with x (n, 9), edge_index, edge_attr (E, 8),
                  global_features (1, 4)

        Returns:
            cache: dict with:
                node_cache: (n, La) projected node embeddings
                edge_gate: (E, La) sigmoid-gated edge embeddings
                edge_index: (2, E) graph connectivity
        """
        x_nodes = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Global features
        if hasattr(data, 'global_features') and data.global_features is not None:
            g = data.global_features.squeeze(0)  # [4]
        else:
            g = torch.zeros(self.global_features_in, device=x_nodes.device,
                            dtype=x_nodes.dtype)

        # Encode
        node_latent = self.node_enc(x_nodes)
        edge_latent = self.edge_enc(edge_attr)
        global_latent = self.global_enc(g.unsqueeze(0))  # [1, L]

        # Save encoder edge embeddings for skip connections
        edge_enc_saved = edge_latent.clone()

        # Process
        for gn_layer, skip_proj in zip(self.processor, self.edge_skip_proj):
            if self.use_checkpoint and edge_latent.requires_grad:
                # skip_proj + gn_layer inside checkpoint so cat/proj intermediates
                # are freed and recomputed during backward
                def _make_fn(layer, sproj):
                    def fn(edge_lat, edge_enc, node, eidx, g):
                        edge_input = sproj(torch.cat([edge_lat, edge_enc], dim=1))
                        return layer(node, eidx, edge_input, g=g)
                    return fn
                edge_latent, node_latent, global_latent = torch_checkpoint(
                    _make_fn(gn_layer, skip_proj),
                    edge_latent, edge_enc_saved, node_latent, edge_index,
                    global_latent,
                    use_reentrant=False,
                )
            else:
                edge_input = skip_proj(torch.cat([edge_latent, edge_enc_saved], dim=1))
                edge_latent, node_latent, global_latent = gn_layer(
                    node_latent, edge_index, edge_input, g=global_latent
                )

        # Project to apply-phase dimensions
        node_cache = self.node_proj(node_latent)       # (n, La)
        edge_gate = self.edge_gate_proj(edge_latent)   # (E, La)

        return {
            'node_cache': node_cache,
            'edge_gate': edge_gate,
            'edge_index': edge_index,
        }

    def apply_precond(self, cache, v_re, v_im):
        """Apply preconditioner: M(v) = v + correction.

        Lightweight forward pass designed to run ~400 times per solve.

        Args:
            cache: dict from encode_geometry()
            v_re: (n,) float32 — real part of input vector (per DOF node)
            v_im: (n,) float32 — imaginary part of input vector (per DOF node)

        Returns:
            out_re: (n,) float32 — real part of M(v)
            out_im: (n,) float32 — imaginary part of M(v)
        """
        node_cache = cache['node_cache']   # (n, La)
        edge_gate = cache['edge_gate']     # (E, La)
        edge_index = cache['edge_index']   # (2, E)
        n = node_cache.shape[0]

        # Input projection: combine cached geometry with vector v
        v_input = torch.stack([v_re, v_im], dim=1)  # (n, 2)
        h = self.apply_input(torch.cat([node_cache, v_input], dim=1))  # (n, La)

        # Lightweight message passing
        row, col = edge_index  # row = target, col = source
        for layer in self.apply_mp:
            msg = layer['msg_proj'](h[col]) * edge_gate   # (E, La)
            agg = self.apply_aggregate(msg, row, dim_size=n)  # (n, La)
            h = h + layer['upd'](torch.cat([h, agg], dim=1))  # (n, La)

        # Output correction
        correction = self.apply_output(h)  # (n, 2)

        # Residual learning: M(v) = v + correction
        out_re = v_re + correction[:, 0]
        out_im = v_im + correction[:, 1]

        return out_re, out_im

    def make_precond_fn(self, cache):
        """Create a callable preconditioner for BiCGStab.

        Returns a function M(v) that takes a complex vector v (3N,) and returns
        the preconditioned vector M(v) (3N,) in the same dtype.

        BiCGStab operates in complex128, so we run apply_precond in float64
        to avoid precision loss from float32 roundtrip (which can add ~30%
        more iterations due to Krylov subspace orthogonality loss).

        Args:
            cache: dict from encode_geometry()

        Returns:
            precond_fn: callable(v) -> M(v), compatible with bicgstab(M=precond_fn)
        """
        # Pre-cast cache to float64 for solver use
        cache_f64 = {
            'node_cache': cache['node_cache'].double(),
            'edge_gate': cache['edge_gate'].double(),
            'edge_index': cache['edge_index'],
        }

        def precond_fn(v):
            # v is (3N,) complex128 from BiCGStab
            # Run in float64 to preserve precision
            # no_grad: precond is called inside Krylov loop, no backprop needed
            with torch.no_grad():
                out_re, out_im = self.apply_precond(cache_f64, v.real, v.imag)
            return torch.complex(out_re, out_im)

        return precond_fn

    def forward(self, data, v_re, v_im):
        """Full forward: encode + apply (for testing/single use).

        Args:
            data: torch_geometric.Data
            v_re: (n,) float32
            v_im: (n,) float32

        Returns:
            out_re: (n,) float32
            out_im: (n,) float32
        """
        cache = self.encode_geometry(data)
        return self.apply_precond(cache, v_re, v_im)


class PolyPrecond(nn.Module):
    """Polynomial neural preconditioner M(v) = Σ c_k · A^k · v.

    GNN predicts K+1 complex polynomial coefficients once per geometry.
    Apply-phase uses Horner's method through FFTMatVec — exact arithmetic,
    zero approximation errors.

    Architecture:
      - Encoder: identical to NeuralPrecond (node 9→L, edge 8→L, global 4→L)
      - Processor: K GraphNet layers with edge skip connections
      - Decoder: global mean pooling → MLP → K+1 complex coefficients
      - Apply: Horner evaluation p(A)·v via FFTMatVec (no learnable params)

    Spectral guarantee: eigenvalues of p(A) are p(λ_i) for polynomial p.
    """

    def __init__(self, poly_degree=3, latent_size=64,
                 message_passing_steps=6,
                 node_features_in=9, edge_features_in=8, global_features_in=4,
                 activation="relu", use_checkpoint=False):
        super().__init__()

        K = poly_degree
        L = latent_size
        self.poly_degree = K
        self.latent_size = L
        self.message_passing_steps = message_passing_steps
        self.use_checkpoint = use_checkpoint
        self.node_features_in = node_features_in
        self.edge_features_in = edge_features_in
        self.global_features_in = global_features_in

        # ── Encoder (same as NeuralPrecond) ──────────────
        self.node_enc = MLP([node_features_in, L, L], activation=activation)
        self.edge_enc = MLP([edge_features_in, L, L], activation=activation)
        self.global_enc = MLP([global_features_in, L, L], activation=activation)

        # ── Processor: GraphNet layers with edge skip connections ─
        self.processor = nn.ModuleList([
            GraphNet(node_features=L, edge_features=L, global_features=L,
                     hidden_size=L, aggregate="mean", activation=activation)
            for _ in range(message_passing_steps)
        ])
        self.edge_skip_proj = nn.ModuleList([
            nn.Linear(2 * L, L)
            for _ in range(message_passing_steps)
        ])

        # ── Decoder: global pooling → MLP → K+1 complex coefficients ─
        # Global mean pooling of node embeddings (1, L) concat with global_latent (1, L) → (1, 2L)
        # Output: 2*(K+1) floats → K+1 complex coefficients (re, im pairs)
        self.coeff_decoder = nn.Sequential(
            nn.Linear(2 * L, L),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(L, L),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(L, 2 * (K + 1)),
        )

        # Initialize decoder output to c_0=1+0j, c_{k>0}=0+0j → p(A) = I
        self._init_identity()

    def _init_identity(self):
        """Initialize so that p(A) = I at the start (c_0=1, rest=0)."""
        last_linear = self.coeff_decoder[-1]
        nn.init.zeros_(last_linear.weight)
        # bias: [re_0, im_0, re_1, im_1, ..., re_K, im_K]
        # Set re_0 = 1, everything else = 0
        bias = torch.zeros(2 * (self.poly_degree + 1))
        bias[0] = 1.0  # c_0 real part = 1
        last_linear.bias.data.copy_(bias)

    def encode_geometry(self, data):
        """Encode geometry graph into K+1 complex polynomial coefficients.

        This is the expensive part (full message passing + decoder). Call once per geometry.

        Args:
            data: torch_geometric.Data with x (n, 9), edge_index, edge_attr (E, 8),
                  global_features (1, 4)

        Returns:
            coefficients: (K+1,) complex64 — polynomial coefficients c_0...c_K
        """
        x_nodes = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Global features
        if hasattr(data, 'global_features') and data.global_features is not None:
            g = data.global_features.squeeze(0)  # [4]
        else:
            g = torch.zeros(self.global_features_in, device=x_nodes.device,
                            dtype=x_nodes.dtype)

        # Encode
        node_latent = self.node_enc(x_nodes)
        edge_latent = self.edge_enc(edge_attr)
        global_latent = self.global_enc(g.unsqueeze(0))  # [1, L]

        # Save encoder edge embeddings for skip connections
        edge_enc_saved = edge_latent.clone()

        # Process
        for gn_layer, skip_proj in zip(self.processor, self.edge_skip_proj):
            if self.use_checkpoint and edge_latent.requires_grad:
                # skip_proj + gn_layer inside checkpoint so cat/proj intermediates
                # are freed and recomputed during backward
                def _make_fn(layer, sproj):
                    def fn(edge_lat, edge_enc, node, eidx, g):
                        edge_input = sproj(torch.cat([edge_lat, edge_enc], dim=1))
                        return layer(node, eidx, edge_input, g=g)
                    return fn
                edge_latent, node_latent, global_latent = torch_checkpoint(
                    _make_fn(gn_layer, skip_proj),
                    edge_latent, edge_enc_saved, node_latent, edge_index,
                    global_latent,
                    use_reentrant=False,
                )
            else:
                edge_input = skip_proj(torch.cat([edge_latent, edge_enc_saved], dim=1))
                edge_latent, node_latent, global_latent = gn_layer(
                    node_latent, edge_index, edge_input, g=global_latent
                )

        # Decode: global mean pooling + MLP → coefficients
        node_mean = node_latent.mean(dim=0, keepdim=True)  # (1, L)
        decoder_input = torch.cat([node_mean, global_latent], dim=1)  # (1, 2L)
        raw = self.coeff_decoder(decoder_input).squeeze(0).float()  # (2*(K+1),) float32 for torch.complex

        # Reshape to complex coefficients: [re_0, im_0, re_1, im_1, ...]
        raw = raw.view(self.poly_degree + 1, 2)
        coefficients = torch.complex(raw[:, 0], raw[:, 1])  # (K+1,)

        return coefficients

    @staticmethod
    def apply_poly(coefficients, fft_matvec, v):
        """Apply polynomial preconditioner p(A)·v via Horner's method.

        Computes h = c_K·v, then for k=K-1,...,0: h = A·h + c_k·v.
        Total cost: K calls to fft_matvec.

        No learnable parameters — exact arithmetic in whatever precision
        fft_matvec operates (typically complex128).

        Args:
            coefficients: (K+1,) complex — polynomial coefficients c_0...c_K
            fft_matvec: callable, A·v operation (handles (n,) or (n, P))
            v: (n,) or (n, P) complex — input vector(s)

        Returns:
            h: same shape as v — p(A)·v
        """
        K = len(coefficients) - 1

        # Cast coefficients to match v dtype for exact arithmetic
        c = coefficients.to(dtype=v.dtype, device=v.device)

        # Horner: h = c_K * v
        if v.dim() == 1:
            h = c[K] * v
        else:
            h = c[K].unsqueeze(0) * v  # broadcast (1,) * (n, P)

        # h = A·h + c_k * v for k = K-1, ..., 0
        for k in range(K - 1, -1, -1):
            # A·h via FFT matvec
            h_2d = h.unsqueeze(1) if h.dim() == 1 else h
            Ah = fft_matvec(h_2d)
            Ah = Ah.squeeze(1) if v.dim() == 1 else Ah

            if v.dim() == 1:
                h = Ah + c[k] * v
            else:
                h = Ah + c[k].unsqueeze(0) * v

        return h

    def make_precond_fn(self, coefficients, fft_matvec):
        """Create a callable preconditioner for BiCGStab.

        Returns a function M(v) = p(A)·v that takes a complex vector v (3N,)
        and returns p(A)·v (3N,) in the same dtype.

        All computation is in exact arithmetic via FFTMatVec — no NN inference
        in the Krylov loop.

        Args:
            coefficients: (K+1,) complex — from encode_geometry()
            fft_matvec: FFTMatVec instance

        Returns:
            precond_fn: callable(v) -> p(A)·v, compatible with bicgstab(M=precond_fn)
        """
        # Detach and move to complex128 for solver precision
        c = coefficients.detach().to(dtype=torch.complex128)

        def precond_fn(v):
            # v is (3N,) complex128 from BiCGStab — no grad needed
            with torch.no_grad():
                return PolyPrecond.apply_poly(c, fft_matvec, v)

        return precond_fn


class PolyPrecondMLP(nn.Module):
    """MLP-based polynomial preconditioner.

    Predicts K+1 complex polynomial coefficients from physical parameters
    (m_re, m_im, kd, grid, shape_id) without needing a graph.

    No graph construction, no message passing — works on any grid size.
    Apply-phase reuses PolyPrecond.apply_poly (Horner via FFTMatVec).

    Architecture:
      - Shape embedding: shape_id → learnable vector (embed_dim)
      - Input: (m_re, m_im, kd, log(grid), shape_embed) → MLP → 2*(K+1) floats
      - Output: K+1 complex polynomial coefficients

    Grid is passed as log(grid) so the MLP sees a normalized scale.
    """

    # Canonical shape name → integer id mapping
    SHAPE_IDS = {'sphere': 0, 'cube': 1, 'ellipsoid': 2, 'cylinder': 3, 'capsule': 4}

    def __init__(self, poly_degree=3, hidden_size=128, num_layers=4,
                 num_shapes=3, shape_embed_dim=8, activation='relu'):
        super().__init__()

        K = poly_degree
        self.poly_degree = K
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_shapes = num_shapes
        self.shape_embed_dim = shape_embed_dim

        # Shape embedding: shape_id → (embed_dim,) vector
        self.shape_embed = nn.Embedding(num_shapes, shape_embed_dim)

        # Activation
        if activation == 'gelu':
            act = nn.GELU()
        else:
            act = nn.ReLU()

        # Build MLP: (4 + embed_dim) → hidden → ... → 2*(K+1)
        # Input: m_re, m_im, kd, log(grid), shape_embed
        input_dim = 4 + shape_embed_dim
        layers = [nn.Linear(input_dim, hidden_size), act]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), act])
        layers.append(nn.Linear(hidden_size, 2 * (K + 1)))
        self.mlp = nn.Sequential(*layers)

        # Initialize output so p(A) = I at start: c_0=1+0j, rest=0
        self._init_identity()

    def _init_identity(self):
        """Initialize so that p(A) = I at the start (c_0=1, rest=0)."""
        last_linear = self.mlp[-1]
        nn.init.zeros_(last_linear.weight)
        bias = torch.zeros(2 * (self.poly_degree + 1))
        bias[0] = 1.0  # c_0 real part = 1
        last_linear.bias.data.copy_(bias)

    def forward(self, m_re, m_im, kd, shape_id, grid):
        """Predict polynomial coefficients from physical parameters.

        Args:
            m_re: scalar or (B,) float — real part of refractive index
            m_im: scalar or (B,) float — imaginary part
            kd: scalar or (B,) float — size parameter
            shape_id: scalar or (B,) int — shape index (0=sphere, 1=cube, 2=ellipsoid)
            grid: scalar or (B,) float/int — grid size

        Returns:
            coefficients: (K+1,) complex tensor (unbatched) or (B, K+1) complex (batched)
        """
        import math
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Handle scalar inputs
        is_scalar = not isinstance(m_re, torch.Tensor) or m_re.dim() == 0
        if is_scalar:
            m_re_t = torch.tensor([m_re], device=device, dtype=dtype)
            m_im_t = torch.tensor([m_im], device=device, dtype=dtype)
            kd_t = torch.tensor([kd], device=device, dtype=dtype)
            log_grid_t = torch.tensor([math.log(grid)], device=device, dtype=dtype)
            shape_id_t = torch.tensor([shape_id], device=device, dtype=torch.long)
        else:
            m_re_t = m_re.to(device=device, dtype=dtype)
            m_im_t = m_im.to(device=device, dtype=dtype)
            kd_t = kd.to(device=device, dtype=dtype)
            log_grid_t = grid.float().to(device=device, dtype=dtype).log()
            shape_id_t = shape_id.to(device=device)

        # Shape embedding
        s_embed = self.shape_embed(shape_id_t)  # (B, embed_dim)

        # Concatenate input features: (m_re, m_im, kd, log_grid, shape_embed)
        x = torch.cat([m_re_t.unsqueeze(-1) if m_re_t.dim() == 1 else m_re_t,
                        m_im_t.unsqueeze(-1) if m_im_t.dim() == 1 else m_im_t,
                        kd_t.unsqueeze(-1) if kd_t.dim() == 1 else kd_t,
                        log_grid_t.unsqueeze(-1) if log_grid_t.dim() == 1 else log_grid_t,
                        s_embed], dim=-1)  # (B, 4 + embed_dim)

        # MLP forward
        raw = self.mlp(x)  # (B, 2*(K+1))

        # Reshape to complex coefficients
        raw = raw.view(-1, self.poly_degree + 1, 2)  # (B, K+1, 2)
        coefficients = torch.complex(raw[..., 0], raw[..., 1])  # (B, K+1)

        if is_scalar:
            coefficients = coefficients.squeeze(0)  # (K+1,)

        return coefficients

    @staticmethod
    def apply_poly(coefficients, fft_matvec, v):
        """Apply polynomial preconditioner p(A)·v via Horner's method.

        Delegates to PolyPrecond.apply_poly — identical implementation.
        """
        return PolyPrecond.apply_poly(coefficients, fft_matvec, v)

    def make_precond_fn(self, coefficients, fft_matvec):
        """Create a callable preconditioner for BiCGStab.

        Delegates to PolyPrecond.make_precond_fn logic.
        """
        c = coefficients.detach().to(dtype=torch.complex128)

        def precond_fn(v):
            with torch.no_grad():
                return PolyPrecond.apply_poly(c, fft_matvec, v)

        return precond_fn


def compute_stencil(r_cut=3):
    """Compute all integer displacement vectors within radius r_cut.

    Returns sorted list of (dx, dy, dz) tuples where dx²+dy²+dz² ≤ r_cut².
    For r_cut=3: 123 displacements. For r_cut=2: 33. For r_cut=1: 7.
    """
    r_cut_sq = r_cut * r_cut
    rc = int(r_cut)
    offsets = []
    for dx in range(-rc, rc + 1):
        for dy in range(-rc, rc + 1):
            for dz in range(-rc, rc + 1):
                if dx * dx + dy * dy + dz * dz <= r_cut_sq:
                    offsets.append((dx, dy, dz))
    offsets.sort()
    return offsets


class ConvSAI_MLP(nn.Module):
    """MLP-based convolutional SAI preconditioner.

    Predicts a translation-invariant 3D convolution kernel M ≈ A⁻¹.
    For each integer displacement (dx, dy, dz) within r_cut, predicts a 3×3
    complex block. On a regular grid this defines a sparse preconditioner.

    Apply via FFT convolution: cost = ONE A·v operation (vs K for polynomial).
    In BiCGStab: 2 A·v + 2 M·v = 4 matvec-equivalents per iteration.
    Need >2x iteration reduction for wall-clock speedup (easy vs poly's >K+1 threshold).

    Architecture:
      - Shape embedding: shape_id → learnable vector (embed_dim)
      - Input: (m_re, m_im, kd, log(grid), shape_embed) → MLP → n_stencil × 3 × 3 × 2
      - Output: convolution kernel as complex tensor
      - Identity init: diagonal (0,0,0) block = I₃, off-diagonal = 0
    """

    SHAPE_IDS = {'sphere': 0, 'cube': 1, 'ellipsoid': 2, 'cylinder': 3, 'capsule': 4}

    def __init__(self, r_cut=3, hidden_size=256, num_layers=4,
                 num_shapes=3, shape_embed_dim=8, activation='relu',
                 scale_by_stencil=True):
        super().__init__()

        self.r_cut = r_cut
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_shapes = num_shapes
        self.shape_embed_dim = shape_embed_dim
        self.scale_by_stencil = scale_by_stencil

        # Precompute stencil offsets
        stencil_list = compute_stencil(r_cut)
        self.register_buffer('stencil', torch.tensor(stencil_list, dtype=torch.long))
        self.n_stencil = len(stencil_list)

        # Find index of (0,0,0) for identity init
        self._diag_idx = stencil_list.index((0, 0, 0))

        # Precompute identity kernel as buffer
        identity = torch.zeros(self.n_stencil, 3, 3, 2)  # last dim: re, im
        identity[self._diag_idx, 0, 0, 0] = 1.0
        identity[self._diag_idx, 1, 1, 0] = 1.0
        identity[self._diag_idx, 2, 2, 0] = 1.0
        self.register_buffer('_identity_kernel', identity)

        # Shape embedding
        self.shape_embed = nn.Embedding(num_shapes, shape_embed_dim)

        # Activation
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()

        # Output: n_stencil * 3 * 3 * 2 (real + imag for each 3×3 block)
        output_dim = self.n_stencil * 18
        input_dim = 4 + shape_embed_dim  # m_re, m_im, kd, log(grid), shape_embed

        layers = [nn.Linear(input_dim, hidden_size), act]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), act])
        layers.append(nn.Linear(hidden_size, output_dim))
        self.mlp = nn.Sequential(*layers)

        # Initialize MLP output to zero (residual starts at 0)
        self._init_zero_residual()

    def _init_zero_residual(self):
        """Initialize MLP to output zero residual (kernel starts as identity)."""
        last_linear = self.mlp[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, m_re, m_im, kd, shape_id, grid):
        """Predict SAI convolution kernel from physical parameters.

        Uses residual architecture: kernel = identity + MLP_residual.
        At initialization MLP outputs zero -> kernel = identity (M = I).

        Args:
            m_re, m_im, kd: scalar or (B,) — physical parameters
            shape_id: scalar or (B,) int — shape index
            grid: scalar or (B,) — grid size

        Returns:
            kernel: (n_stencil, 3, 3) complex (unbatched) or (B, n_stencil, 3, 3) complex
        """
        import math
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        is_scalar = not isinstance(m_re, torch.Tensor) or m_re.dim() == 0
        if is_scalar:
            m_re_t = torch.tensor([m_re], device=device, dtype=dtype)
            m_im_t = torch.tensor([m_im], device=device, dtype=dtype)
            kd_t = torch.tensor([kd], device=device, dtype=dtype)
            log_grid_t = torch.tensor([math.log(grid)], device=device, dtype=dtype)
            shape_id_t = torch.tensor([shape_id], device=device, dtype=torch.long)
        else:
            m_re_t = m_re.to(device=device, dtype=dtype)
            m_im_t = m_im.to(device=device, dtype=dtype)
            kd_t = kd.to(device=device, dtype=dtype)
            log_grid_t = grid.float().to(device=device, dtype=dtype).log()
            shape_id_t = shape_id.to(device=device)

        s_embed = self.shape_embed(shape_id_t)

        x = torch.cat([m_re_t.unsqueeze(-1) if m_re_t.dim() == 1 else m_re_t,
                        m_im_t.unsqueeze(-1) if m_im_t.dim() == 1 else m_im_t,
                        kd_t.unsqueeze(-1) if kd_t.dim() == 1 else kd_t,
                        log_grid_t.unsqueeze(-1) if log_grid_t.dim() == 1 else log_grid_t,
                        s_embed], dim=-1)

        residual = self.mlp(x)  # (B, n_stencil * 18)
        # Scale residual to keep initial updates small
        if self.scale_by_stencil:
            residual = residual * (1.0 / math.sqrt(self.n_stencil * 18))
        else:
            residual = residual * (1.0 / math.sqrt(18))
        residual = residual.view(-1, self.n_stencil, 3, 3, 2)  # (B, S, 3, 3, 2)

        # Add identity kernel (broadcast over batch)
        raw = residual + self._identity_kernel  # (B, S, 3, 3, 2)
        kernel = torch.complex(raw[..., 0], raw[..., 1])  # (B, S, 3, 3)

        if is_scalar:
            kernel = kernel.squeeze(0)  # (S, 3, 3)

        return kernel

    def build_M_hat(self, kernel, fft_matvec):
        """Build FFT of kernel on doubled grid for convolution apply.

        Args:
            kernel: (n_stencil, 3, 3) complex — from forward()
            fft_matvec: FFTMatVec instance (provides grid dimensions)

        Returns:
            M_hat: (3, 3, gx, gy, gz) complex — FFT of kernel on doubled grid
        """
        device = kernel.device
        box = fft_matvec.box
        gx = 2 * box[0].item()
        gy = 2 * box[1].item()
        gz = 2 * box[2].item()

        stencil = self.stencil.to(device)
        wi = stencil[:, 0] % gx
        wj = stencil[:, 1] % gy
        wk = stencil[:, 2] % gz

        # Flatten kernel for out-of-place scatter (autograd-safe)
        kernel_9 = kernel.reshape(self.n_stencil, 9).T  # (9, S)
        flat_size = gx * gy * gz
        flat_idx = (wi * gy * gz + wj * gz + wk).long()  # (S,)
        idx_expanded = flat_idx.unsqueeze(0).expand(9, -1)  # (9, S)

        M_flat = torch.zeros(9, flat_size, dtype=kernel.dtype, device=device)
        M_flat = M_flat.scatter(1, idx_expanded, kernel_9)
        M_grid = M_flat.reshape(3, 3, gx, gy, gz)

        M_hat = torch.fft.fftn(M_grid, dim=(2, 3, 4))
        return M_hat

    def make_precond_fn(self, kernel, fft_matvec):
        """Create callable preconditioner for BiCGStab.

        Builds FFT of kernel on doubled grid, then applies M·v via FFT convolution.
        Cost per application: one FFT convolution ≈ one A·v.

        Args:
            kernel: (n_stencil, 3, 3) complex — from forward()
            fft_matvec: FFTMatVec instance

        Returns:
            precond_fn: callable(v) -> M·v, compatible with bicgstab(M=precond_fn)
        """
        kernel_c128 = kernel.detach().to(dtype=torch.complex128, device='cpu')

        box = fft_matvec.box
        gx = 2 * box[0].item()
        gy = 2 * box[1].item()
        gz = 2 * box[2].item()

        stencil = self.stencil.cpu()
        wi = stencil[:, 0] % gx
        wj = stencil[:, 1] % gy
        wk = stencil[:, 2] % gz

        # Build M_hat (no autograd needed in eval)
        M_grid = torch.zeros(3, 3, gx, gy, gz, dtype=torch.complex128)
        M_grid[:, :, wi, wj, wk] = kernel_c128.permute(1, 2, 0)
        M_hat = torch.fft.fftn(M_grid, dim=(2, 3, 4))

        pos = fft_matvec.pos_shifted.cpu()
        pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
        N = fft_matvec.N
        n = fft_matvec.n

        def precond_fn(v):
            # v: (3N,) complex128 from BiCGStab
            with torch.no_grad():
                v_grid = torch.zeros(3, gx, gy, gz, dtype=torch.complex128)
                v_grid[:, pi, pj, pk] = v.reshape(N, 3).T  # (3, N)

                v_hat = torch.fft.fftn(v_grid, dim=(1, 2, 3))
                result_hat = torch.einsum('ijxyz,jxyz->ixyz', M_hat, v_hat)
                result_grid = torch.fft.ifftn(result_hat, dim=(1, 2, 3))

                result = result_grid[:, pi, pj, pk]  # (3, N)
                return result.T.reshape(n)

        return precond_fn


# ---------------------------------------------------------------------------
# Universal model: 3D CNN shape encoder replaces discrete shape_id
# ---------------------------------------------------------------------------

def positions_to_occupancy(positions, grid_size=None, device='cpu'):
    """Convert dipole positions to 3D binary occupancy grid.

    Args:
        positions: (N, 3) numpy array or tensor — integer dipole positions
        grid_size: int — grid dimension (if None, inferred from positions)
        device: target device

    Returns:
        grid: (1, 1, G, G, G) float tensor — binary occupancy grid
    """
    import numpy as np
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions).long()
    else:
        positions = positions.long()

    # Shift to start from (0, 0, 0)
    pos = positions - positions.min(dim=0).values

    if grid_size is None:
        grid_size = int(pos.max().item()) + 1

    grid = torch.zeros(1, 1, grid_size, grid_size, grid_size, device=device)
    grid[0, 0, pos[:, 0], pos[:, 1], pos[:, 2]] = 1.0
    return grid


class ShapeEncoder3D(nn.Module):
    """3D CNN encoder: binary occupancy grid -> fixed-size shape embedding.

    Converts an arbitrary 3D binary shape into a continuous embedding vector.
    Handles variable input sizes via trilinear interpolation to fixed resolution.

    Architecture:
      - 3 x (Conv3d + ReLU + MaxPool3d) downsample by 8x
      - AdaptiveAvgPool3d(1) -> Linear -> embed_dim
    """

    def __init__(self, embed_dim=8, encoder_resolution=32, channels=(16, 32, 64)):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder_resolution = encoder_resolution

        layers = []
        in_c = 1
        for c in channels:
            layers.extend([
                nn.Conv3d(in_c, c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
            ])
            in_c = c
        layers.append(nn.AdaptiveAvgPool3d(1))

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 1, G, G, G) float tensor — binary occupancy grid

        Returns:
            (B, embed_dim) shape embedding
        """
        R = self.encoder_resolution
        if x.shape[2] != R or x.shape[3] != R or x.shape[4] != R:
            x = torch.nn.functional.interpolate(
                x, size=(R, R, R), mode='trilinear', align_corners=False,
            )
        h = self.conv(x)          # (B, C, 1, 1, 1)
        return self.fc(h.flatten(1))  # (B, embed_dim)


class ConvSAI_Universal(nn.Module):
    """Universal ConvSAI preconditioner with 3D geometry encoder.

    Instead of discrete shape_id -> nn.Embedding, uses a 3D CNN encoder
    that takes binary occupancy grids of arbitrary shapes and produces
    a continuous shape embedding. One model for ALL shapes.

    Architecture identical to ConvSAI_MLP except:
      - shape_embed replaced by ShapeEncoder3D
      - forward() takes occupancy_grid instead of shape_id
    """

    def __init__(self, r_cut=3, hidden_size=256, num_layers=4,
                 shape_embed_dim=16, activation='relu',
                 scale_by_stencil=True,
                 encoder_resolution=32, encoder_channels=(16, 32, 64)):
        super().__init__()

        self.r_cut = r_cut
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.shape_embed_dim = shape_embed_dim
        self.scale_by_stencil = scale_by_stencil
        self.encoder_resolution = encoder_resolution

        # Precompute stencil offsets (same as ConvSAI_MLP)
        stencil_list = compute_stencil(r_cut)
        self.register_buffer('stencil', torch.tensor(stencil_list, dtype=torch.long))
        self.n_stencil = len(stencil_list)

        self._diag_idx = stencil_list.index((0, 0, 0))

        # Identity kernel buffer
        identity = torch.zeros(self.n_stencil, 3, 3, 2)
        identity[self._diag_idx, 0, 0, 0] = 1.0
        identity[self._diag_idx, 1, 1, 0] = 1.0
        identity[self._diag_idx, 2, 2, 0] = 1.0
        self.register_buffer('_identity_kernel', identity)

        # 3D CNN shape encoder (replaces nn.Embedding)
        self.shape_encoder = ShapeEncoder3D(
            embed_dim=shape_embed_dim,
            encoder_resolution=encoder_resolution,
            channels=encoder_channels,
        )

        # Activation
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()

        # MLP: (4 + shape_embed_dim) -> hidden -> ... -> n_stencil * 18
        output_dim = self.n_stencil * 18
        input_dim = 4 + shape_embed_dim

        layers = [nn.Linear(input_dim, hidden_size), act]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), act])
        layers.append(nn.Linear(hidden_size, output_dim))
        self.mlp = nn.Sequential(*layers)

        # Initialize MLP output to zero residual (kernel starts as identity)
        last_linear = self.mlp[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, m_re, m_im, kd, occupancy_grid, grid):
        """Predict SAI convolution kernel from physical parameters + geometry.

        Args:
            m_re, m_im, kd: scalar or (B,) — physical parameters
            occupancy_grid: (B, 1, G, G, G) or (1, G, G, G) — binary occupancy
            grid: scalar or (B,) — grid size

        Returns:
            kernel: (n_stencil, 3, 3) complex or (B, n_stencil, 3, 3) complex
        """
        import math
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        is_scalar = not isinstance(m_re, torch.Tensor) or m_re.dim() == 0
        if is_scalar:
            m_re_t = torch.tensor([m_re], device=device, dtype=dtype)
            m_im_t = torch.tensor([m_im], device=device, dtype=dtype)
            kd_t = torch.tensor([kd], device=device, dtype=dtype)
            log_grid_t = torch.tensor([math.log(grid)], device=device, dtype=dtype)
        else:
            m_re_t = m_re.to(device=device, dtype=dtype)
            m_im_t = m_im.to(device=device, dtype=dtype)
            kd_t = kd.to(device=device, dtype=dtype)
            log_grid_t = grid.float().to(device=device, dtype=dtype).log()

        if occupancy_grid.dim() == 4:
            occupancy_grid = occupancy_grid.unsqueeze(0)
        occupancy_grid = occupancy_grid.to(device=device, dtype=dtype)

        # Shape embedding from 3D CNN encoder
        s_embed = self.shape_encoder(occupancy_grid)  # (B, embed_dim)

        x = torch.cat([m_re_t.unsqueeze(-1) if m_re_t.dim() == 1 else m_re_t,
                        m_im_t.unsqueeze(-1) if m_im_t.dim() == 1 else m_im_t,
                        kd_t.unsqueeze(-1) if kd_t.dim() == 1 else kd_t,
                        log_grid_t.unsqueeze(-1) if log_grid_t.dim() == 1 else log_grid_t,
                        s_embed], dim=-1)

        residual = self.mlp(x)  # (B, n_stencil * 18)
        if self.scale_by_stencil:
            residual = residual * (1.0 / math.sqrt(self.n_stencil * 18))
        else:
            residual = residual * (1.0 / math.sqrt(18))
        residual = residual.view(-1, self.n_stencil, 3, 3, 2)

        raw = residual + self._identity_kernel
        kernel = torch.complex(raw[..., 0], raw[..., 1])

        if is_scalar:
            kernel = kernel.squeeze(0)

        return kernel

    # Reuse build_M_hat and make_precond_fn from ConvSAI_MLP
    build_M_hat = ConvSAI_MLP.build_M_hat
    make_precond_fn = ConvSAI_MLP.make_precond_fn


class ConvSAI_Multigrid(nn.Module):
    """Multigrid ConvSAI: multi-scale stencils for large effective r_cut.

    Places the same r_cut stencil at stride 1, 2, 4, ..., giving
    exponentially growing receptive field with linear parameter cost.

    For r_cut=7 with 3 levels (stride 1, 2, 4):
      - Level 0 (fine):   stride=1, r_cut_eff = 7   (1419 entries)
      - Level 1 (medium): stride=2, r_cut_eff = 14  (1419 entries)
      - Level 2 (coarse): stride=4, r_cut_eff = 28  (1419 entries)
      - Combined: ~4257 unique entries (vs 10649 for K² at r_cut=14)

    Level 0 uses the base model's output head (identity + residual).
    Coarse levels have separate heads, zero-initialized (start as zero correction).
    Backbone (all MLP layers except last) is shared across levels.

    forward() returns a list of kernels; build_M_hat() and make_precond_fn()
    accept this list and compose all levels into a single preconditioner.
    This makes it transparent to existing loss functions.
    """

    def __init__(self, base, num_levels=3, bottleneck=0, squared=False):
        super().__init__()
        self.base = base
        self.num_levels = num_levels
        self.squared = squared
        self.r_cut = base.r_cut
        self.n_stencil = base.n_stencil
        self.stencil = base.stencil
        self.shape_encoder = base.shape_encoder

        # Coarse-level output heads (level 0 reuses base's last layer)
        hidden = base.hidden_size
        n_out = base.n_stencil * 18
        self.coarse_heads = nn.ModuleList()
        for _ in range(1, num_levels):
            if bottleneck > 0:
                head = nn.Sequential(
                    nn.Linear(hidden, bottleneck),
                    nn.ReLU(),
                    nn.Linear(bottleneck, n_out),
                )
                # Zero-init last layer
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
            else:
                head = nn.Linear(hidden, n_out)
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            self.coarse_heads.append(head)

    def _prepare_input(self, m_re, m_im, kd, occupancy_grid, grid):
        """Prepare MLP input tensor (same logic as base.forward)."""
        import math
        device = next(self.base.parameters()).device
        dtype = next(self.base.parameters()).dtype

        is_scalar = not isinstance(m_re, torch.Tensor) or m_re.dim() == 0
        if is_scalar:
            m_re_t = torch.tensor([m_re], device=device, dtype=dtype)
            m_im_t = torch.tensor([m_im], device=device, dtype=dtype)
            kd_t = torch.tensor([kd], device=device, dtype=dtype)
            log_grid_t = torch.tensor([math.log(grid)], device=device, dtype=dtype)
        else:
            m_re_t = m_re.to(device=device, dtype=dtype)
            m_im_t = m_im.to(device=device, dtype=dtype)
            kd_t = kd.to(device=device, dtype=dtype)
            log_grid_t = grid.float().to(device=device, dtype=dtype).log()

        if occupancy_grid.dim() == 4:
            occupancy_grid = occupancy_grid.unsqueeze(0)
        occupancy_grid = occupancy_grid.to(device=device, dtype=dtype)

        s_embed = self.base.shape_encoder(occupancy_grid)

        x = torch.cat([m_re_t.unsqueeze(-1) if m_re_t.dim() == 1 else m_re_t,
                        m_im_t.unsqueeze(-1) if m_im_t.dim() == 1 else m_im_t,
                        kd_t.unsqueeze(-1) if kd_t.dim() == 1 else kd_t,
                        log_grid_t.unsqueeze(-1) if log_grid_t.dim() == 1 else log_grid_t,
                        s_embed], dim=-1)

        return x, is_scalar

    def forward(self, m_re, m_im, kd, occupancy_grid, grid):
        """Predict multi-level kernels from physical parameters + geometry.

        Returns:
            kernels: list of num_levels tensors, each (n_stencil, 3, 3) complex
        """
        import math
        x, is_scalar = self._prepare_input(m_re, m_im, kd, occupancy_grid, grid)

        # Run backbone (all layers except the last Linear)
        # Use _modules.values() instead of children() to preserve duplicate activations
        mlp_layers = list(self.base.mlp._modules.values())
        features = x
        for layer in mlp_layers[:-1]:
            features = layer(features)

        # Level 0: base model's last layer (identity + residual)
        fine_head = mlp_layers[-1]
        residual_0 = fine_head(features)

        # Coarse levels: separate heads (pure correction, no identity)
        coarse_residuals = [head(features) for head in self.coarse_heads]

        # Build kernels
        scale_factor = 1.0 / math.sqrt(self.n_stencil * 18) if self.base.scale_by_stencil \
            else 1.0 / math.sqrt(18)

        kernels = []
        for level, residual in enumerate([residual_0] + coarse_residuals):
            residual = residual * scale_factor
            residual = residual.view(-1, self.n_stencil, 3, 3, 2)

            if level == 0:
                raw = residual + self.base._identity_kernel
            else:
                raw = residual  # coarse levels: pure correction, zero-init

            kernel = torch.complex(raw[..., 0], raw[..., 1])
            if is_scalar:
                kernel = kernel.squeeze(0)
            kernels.append(kernel)

        return kernels

    def build_M_hat(self, kernels, fft_matvec):
        """Build M_hat by composing all levels at their respective strides.

        Each level's stencil entries are placed at stride 2^level on the
        doubled grid, then a single FFT produces M_hat.
        """
        device = kernels[0].device
        box = fft_matvec.box
        gx = 2 * box[0].item()
        gy = 2 * box[1].item()
        gz = 2 * box[2].item()

        stencil = self.stencil.to(device)
        flat_size = gx * gy * gz

        # Accumulate all levels via out-of-place scatter (autograd-safe)
        M_flat = torch.zeros(9, flat_size, dtype=kernels[0].dtype, device=device)

        for level, kernel in enumerate(kernels):
            scale = 2 ** level

            # Skip coarse levels where stencil would wrap around
            max_disp = self.r_cut * scale
            if max_disp > min(gx, gy, gz) // 2:
                continue

            wi = (stencil[:, 0] * scale) % gx
            wj = (stencil[:, 1] * scale) % gy
            wk = (stencil[:, 2] * scale) % gz

            kernel_9 = kernel.reshape(self.n_stencil, 9).T  # (9, S)
            flat_idx = (wi * gy * gz + wj * gz + wk).long()
            idx_expanded = flat_idx.unsqueeze(0).expand(9, -1)

            level_flat = torch.zeros(9, flat_size, dtype=kernel.dtype, device=device)
            level_flat = level_flat.scatter(1, idx_expanded, kernel_9)
            M_flat = M_flat + level_flat

        M_grid = M_flat.reshape(3, 3, gx, gy, gz)
        M_hat = torch.fft.fftn(M_grid, dim=(2, 3, 4))

        if self.squared:
            # K²: M_hat = K_hat @ K_hat (3x3 matrix multiply per frequency)
            M_hat = torch.einsum('ijxyz,jkxyz->ikxyz', M_hat, M_hat)

        return M_hat

    def make_precond_fn(self, kernels, fft_matvec):
        """Create callable preconditioner for BiCGStab."""
        kernels_c128 = [k.detach().to(dtype=torch.complex128, device='cpu')
                        for k in kernels]

        box = fft_matvec.box
        gx = 2 * box[0].item()
        gy = 2 * box[1].item()
        gz = 2 * box[2].item()

        stencil = self.stencil.cpu()

        # Build combined M_grid from all levels
        M_grid = torch.zeros(3, 3, gx, gy, gz, dtype=torch.complex128)
        for level, kernel in enumerate(kernels_c128):
            scale = 2 ** level
            max_disp = self.r_cut * scale
            if max_disp > min(gx, gy, gz) // 2:
                continue

            wi = (stencil[:, 0] * scale) % gx
            wj = (stencil[:, 1] * scale) % gy
            wk = (stencil[:, 2] * scale) % gz
            M_grid[:, :, wi, wj, wk] += kernel.permute(1, 2, 0)

        K_hat = torch.fft.fftn(M_grid, dim=(2, 3, 4))
        if self.squared:
            M_hat = torch.einsum('ijxyz,jkxyz->ikxyz', K_hat, K_hat)
        else:
            M_hat = K_hat

        pos = fft_matvec.pos_shifted.cpu()
        pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
        N = fft_matvec.N
        n = fft_matvec.n

        def precond_fn(v):
            with torch.no_grad():
                v_grid = torch.zeros(3, gx, gy, gz, dtype=torch.complex128)
                v_grid[:, pi, pj, pk] = v.reshape(N, 3).T
                v_hat = torch.fft.fftn(v_grid, dim=(1, 2, 3))
                result_hat = torch.einsum('ijxyz,jxyz->ixyz', M_hat, v_hat)
                result_grid = torch.fft.ifftn(result_hat, dim=(1, 2, 3))
                return result_grid[:, pi, pj, pk].T.reshape(n)

        return precond_fn


class ConvSAI_Separable(nn.Module):
    """Separable 1D kernel preconditioner for large grids.

    Three independent 1D convolution kernels (along x, y, z axes) composed
    multiplicatively in frequency domain:

        M_hat(fx,fy,fz) = Kx_hat(fx) @ Ky_hat(fy) @ Kz_hat(fz)

    Each 1D kernel has range [-L, L] with (2L+1) entries, each a 3x3 complex block.
    Total output: 3 * (2L+1) * 18 values.

    With axis_range=50: covers [-50,50]^3 cube with only 5.5K values
    (vs 25.5K for r_cut=7 spherical stencil).

    K^2 option: square M_hat after composition for doubled effective range.
    """

    def __init__(self, axis_range=50, hidden_size=512, num_layers=4,
                 shape_embed_dim=16, activation='relu',
                 scale_by_stencil=True, squared=False,
                 encoder_resolution=32, encoder_channels=(16, 32, 64)):
        super().__init__()

        self.axis_range = axis_range
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.shape_embed_dim = shape_embed_dim
        self.scale_by_stencil = scale_by_stencil
        self.squared = squared
        self.encoder_resolution = encoder_resolution
        self.n_axis = 2 * axis_range + 1

        # Identity: (3, n_axis, 3, 3, 2) — I_3 at offset 0 for each axis
        identity = torch.zeros(3, self.n_axis, 3, 3, 2)
        for ax in range(3):
            identity[ax, axis_range, 0, 0, 0] = 1.0
            identity[ax, axis_range, 1, 1, 0] = 1.0
            identity[ax, axis_range, 2, 2, 0] = 1.0
        self.register_buffer('_identity_kernel', identity)

        self.register_buffer('_offsets',
                             torch.arange(-axis_range, axis_range + 1, dtype=torch.long))

        self.shape_encoder = ShapeEncoder3D(
            embed_dim=shape_embed_dim,
            encoder_resolution=encoder_resolution,
            channels=encoder_channels,
        )

        act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        output_dim = 3 * self.n_axis * 18
        input_dim = 4 + shape_embed_dim

        layers = [nn.Linear(input_dim, hidden_size), act]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), act])
        layers.append(nn.Linear(hidden_size, output_dim))
        self.mlp = nn.Sequential(*layers)

        last_linear = self.mlp[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, m_re, m_im, kd, occupancy_grid, grid):
        """Predict 3 axis kernels.

        Returns:
            kernels: (3, n_axis, 3, 3) complex
        """
        import math
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        is_scalar = not isinstance(m_re, torch.Tensor) or m_re.dim() == 0
        if is_scalar:
            m_re_t = torch.tensor([m_re], device=device, dtype=dtype)
            m_im_t = torch.tensor([m_im], device=device, dtype=dtype)
            kd_t = torch.tensor([kd], device=device, dtype=dtype)
            log_grid_t = torch.tensor([math.log(grid)], device=device, dtype=dtype)
        else:
            m_re_t = m_re.to(device=device, dtype=dtype)
            m_im_t = m_im.to(device=device, dtype=dtype)
            kd_t = kd.to(device=device, dtype=dtype)
            log_grid_t = grid.float().to(device=device, dtype=dtype).log()

        if occupancy_grid.dim() == 4:
            occupancy_grid = occupancy_grid.unsqueeze(0)
        occupancy_grid = occupancy_grid.to(device=device, dtype=dtype)

        s_embed = self.shape_encoder(occupancy_grid)
        x = torch.cat([m_re_t.unsqueeze(-1) if m_re_t.dim() == 1 else m_re_t,
                        m_im_t.unsqueeze(-1) if m_im_t.dim() == 1 else m_im_t,
                        kd_t.unsqueeze(-1) if kd_t.dim() == 1 else kd_t,
                        log_grid_t.unsqueeze(-1) if log_grid_t.dim() == 1 else log_grid_t,
                        s_embed], dim=-1)

        residual = self.mlp(x)
        if self.scale_by_stencil:
            import math as _m
            residual = residual * (1.0 / _m.sqrt(3 * self.n_axis * 18))
        else:
            import math as _m
            residual = residual * (1.0 / _m.sqrt(18))
        residual = residual.view(-1, 3, self.n_axis, 3, 3, 2)

        raw = residual + self._identity_kernel
        kernels = torch.complex(raw[..., 0], raw[..., 1])

        if is_scalar:
            kernels = kernels.squeeze(0)
        return kernels

    def build_M_hat(self, kernels, fft_matvec):
        """Build M_hat via multiplicative composition of 1D axis FFTs.

        M_hat[i,l,x,y,z] = sum_{j,k} Kx[i,j,x] * Ky[j,k,y] * Kz[k,l,z]
        """
        device = kernels.device
        box = fft_matvec.box
        gx = 2 * box[0].item()
        gy = 2 * box[1].item()
        gz = 2 * box[2].item()

        offsets = self._offsets.to(device)
        sizes = [gx, gy, gz]
        K_hats = []

        for ax in range(3):
            g = sizes[ax]
            valid = offsets.abs() < g // 2
            idx = (offsets[valid] % g).long()
            k_vals = kernels[ax, valid]  # (n_valid, 3, 3)

            # Autograd-safe scatter
            k_flat = k_vals.reshape(-1, 9).T  # (9, n_valid)
            idx_flat = idx.unsqueeze(0).expand(9, -1)
            K_flat = torch.zeros(9, g, dtype=kernels.dtype, device=device)
            K_flat = K_flat.scatter(1, idx_flat, k_flat)
            K_1d = K_flat.reshape(3, 3, g)

            K_hats.append(torch.fft.fft(K_1d, dim=2))

        M_hat = torch.einsum('ijx,jky,klz->ilxyz', *K_hats)

        if self.squared:
            M_hat = torch.einsum('ijxyz,jkxyz->ikxyz', M_hat, M_hat)

        return M_hat

    def make_precond_fn(self, kernels, fft_matvec):
        """Create callable preconditioner for BiCGStab."""
        kernels_c128 = kernels.detach().to(dtype=torch.complex128, device='cpu')
        box = fft_matvec.box
        gx, gy, gz = 2 * box[0].item(), 2 * box[1].item(), 2 * box[2].item()

        offsets = self._offsets.cpu()
        sizes = [gx, gy, gz]
        K_hats = []
        for ax in range(3):
            g = sizes[ax]
            valid = offsets.abs() < g // 2
            idx = offsets[valid] % g
            K_1d = torch.zeros(3, 3, g, dtype=torch.complex128)
            K_1d[:, :, idx] = kernels_c128[ax, valid].permute(1, 2, 0)
            K_hats.append(torch.fft.fft(K_1d, dim=2))

        M_hat = torch.einsum('ijx,jky,klz->ilxyz', *K_hats)
        if self.squared:
            M_hat = torch.einsum('ijxyz,jkxyz->ikxyz', M_hat, M_hat)

        pos = fft_matvec.pos_shifted.cpu()
        pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
        N = fft_matvec.N
        n = fft_matvec.n

        def precond_fn(v):
            with torch.no_grad():
                v_grid = torch.zeros(3, gx, gy, gz, dtype=torch.complex128)
                v_grid[:, pi, pj, pk] = v.reshape(N, 3).T
                v_hat = torch.fft.fftn(v_grid, dim=(1, 2, 3))
                result_hat = torch.einsum('ijxyz,jxyz->ixyz', M_hat, v_hat)
                result_grid = torch.fft.ifftn(result_hat, dim=(1, 2, 3))
                return result_grid[:, pi, pj, pk].T.reshape(n)

        return precond_fn


class ConvSAI_Hybrid(nn.Module):
    """Hybrid preconditioner: 3D stencil (near) x separable 1D (far).

    M_hat = M_near_hat @ M_far_hat  (multiplicative in frequency domain)

    Near-field: ConvSAI_Universal with r_cut=7 (pre-trained, local 3D)
    Far-field:  ConvSAI_Separable (learns long-range axis corrections)

    At init: M_far = I -> M = M_near (no regression from pre-trained).
    """

    def __init__(self, near_model, far_model, squared=False):
        super().__init__()
        self.near = near_model
        self.far = far_model
        self.squared = squared
        self.shape_encoder = near_model.shape_encoder

    def forward(self, m_re, m_im, kd, occupancy_grid, grid):
        near_kernel = self.near(m_re, m_im, kd, occupancy_grid, grid)
        far_kernels = self.far(m_re, m_im, kd, occupancy_grid, grid)
        return (near_kernel, far_kernels)

    def build_M_hat(self, kernels, fft_matvec):
        near_kernel, far_kernels = kernels
        M_near = self.near.build_M_hat(near_kernel, fft_matvec)
        M_far = self.far.build_M_hat(far_kernels, fft_matvec)
        M_hat = torch.einsum('ijxyz,jkxyz->ikxyz', M_near, M_far)
        if self.squared:
            M_hat = torch.einsum('ijxyz,jkxyz->ikxyz', M_hat, M_hat)
        return M_hat

    def make_precond_fn(self, kernels, fft_matvec):
        near_kernel, far_kernels = kernels
        near_c128 = near_kernel.detach().to(dtype=torch.complex128, device='cpu')
        far_c128 = far_kernels.detach().to(dtype=torch.complex128, device='cpu')

        box = fft_matvec.box
        gx, gy, gz = 2*box[0].item(), 2*box[1].item(), 2*box[2].item()

        # Near M_hat
        stencil = self.near.stencil.cpu()
        M_grid = torch.zeros(3, 3, gx, gy, gz, dtype=torch.complex128)
        M_grid[:, :, stencil[:,0]%gx, stencil[:,1]%gy, stencil[:,2]%gz] = \
            near_c128.permute(1, 2, 0)
        M_near = torch.fft.fftn(M_grid, dim=(2, 3, 4))

        # Far M_hat (separable)
        offsets = self.far._offsets.cpu()
        sizes = [gx, gy, gz]
        K_hats = []
        for ax in range(3):
            g = sizes[ax]
            valid = offsets.abs() < g // 2
            idx = offsets[valid] % g
            K_1d = torch.zeros(3, 3, g, dtype=torch.complex128)
            K_1d[:, :, idx] = far_c128[ax, valid].permute(1, 2, 0)
            K_hats.append(torch.fft.fft(K_1d, dim=2))
        M_far = torch.einsum('ijx,jky,klz->ilxyz', *K_hats)

        M_hat = torch.einsum('ijxyz,jkxyz->ikxyz', M_near, M_far)
        if self.squared:
            M_hat = torch.einsum('ijxyz,jkxyz->ikxyz', M_hat, M_hat)

        pos = fft_matvec.pos_shifted.cpu()
        pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
        N = fft_matvec.N
        n = fft_matvec.n

        def precond_fn(v):
            with torch.no_grad():
                v_grid = torch.zeros(3, gx, gy, gz, dtype=torch.complex128)
                v_grid[:, pi, pj, pk] = v.reshape(N, 3).T
                v_hat = torch.fft.fftn(v_grid, dim=(1, 2, 3))
                result_hat = torch.einsum('ijxyz,jxyz->ixyz', M_hat, v_hat)
                result_grid = torch.fft.ifftn(result_hat, dim=(1, 2, 3))
                return result_grid[:, pi, pj, pk].T.reshape(n)

        return precond_fn


class ResidualFrequencyMLP(nn.Module):
    """Frequency MLP variant used by older spectral checkpoints."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks, activation='relu'):
        super().__init__()
        act = nn.GELU if activation == 'gelu' else nn.ReLU
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.act_in = act()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_blocks)
        ])
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.act_in(self.proj_in(x))
        for block in self.blocks:
            x = x + block(x)
        return self.proj_out(x)


class SpectralCoarseTransformer(nn.Module):
    """Cheap global frequency-context branch for ConvSAI_Spectral.

    The main Spectral ConvSAI MLP is pointwise in frequency. This branch samples
    a small coarse grid of frequency tokens, lets them communicate with a
    transformer, and returns an additive update to the global conditioning
    vector. It is intentionally zero-initialized so existing resumed checkpoints
    start with identical behavior until the branch is trained.
    """

    def __init__(self, input_dim, cond_dim, model_dim=128, num_layers=2,
                 num_heads=4, mlp_ratio=2.0, scale=1.0):
        super().__init__()
        self.scale = float(scale)
        self.token_proj = nn.Linear(input_dim, model_dim)
        self.cond_proj = nn.Linear(cond_dim, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=int(model_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.out = nn.Linear(model_dim, cond_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, token_features, cond):
        tokens = self.token_proj(token_features).unsqueeze(0)
        cls = self.cond_proj(cond).view(1, 1, -1)
        h = torch.cat([cls, tokens], dim=1)
        h = self.encoder(h)
        delta = self.out(self.norm(h[:, 0, :])).squeeze(0)
        return delta * self.scale


class ConvSAI_Spectral(nn.Module):
    """Pointwise spectral preconditioner — no stencil, no range limit.

    For each frequency point k, a small MLP predicts M_hat(k) from:
      - D_hat(k): full 3x3 complex interaction matrix (18 reals)
      - Normalized frequency coordinates (kx/gx, ky/gy, kz/gz) — 3 reals
      - Global conditioning: (m_re, m_im, kd, log_grid, shape_embed)

    The same MLP is applied independently to every frequency point.
    Works on ANY grid size. No stencil range limitation.

    Options:
      - squared: apply K² after building M_hat (M = M @ M)
      - freq_coords: add normalized frequency coordinates to input

    forward() returns a global conditioning vector.
    build_M_hat(cond, fft_matvec) applies the per-frequency MLP using D_hat.

    Identity init: MLP outputs zero residual -> M_hat = I for all frequencies.
    """

    def __init__(self, freq_hidden=64, freq_layers=3,
                 global_hidden=256, global_layers=3,
                 shape_embed_dim=16, activation='relu',
                 squared=False, freq_coords=True,
                 encoder_resolution=32, encoder_channels=(16, 32, 64),
                 normalize_inputs=False,
                 freq_chunk_size=0,
                 freq_checkpoint_chunks=False,
                 freq_residual_blocks=0,
                 correction_hidden=0,
                 correction_layers=2,
                 correction_scale=1.0,
                 transformer_tokens=0,
                 transformer_dim=128,
                 transformer_layers=2,
                 transformer_heads=4,
                 transformer_scale=1.0,
                 m_re_min=1.5, m_re_max=4.0,
                 m_im_min=0.0, m_im_max=0.5,
                 kd_min=0.2, kd_max=0.8,
                 log_grid_center=2.0, log_grid_scale=2.0):
        super().__init__()

        self.shape_embed_dim = shape_embed_dim
        self.encoder_resolution = encoder_resolution
        self.squared = squared
        self.freq_coords = freq_coords
        self.normalize_inputs = normalize_inputs
        self.freq_chunk_size = int(freq_chunk_size or 0)
        self.freq_checkpoint_chunks = bool(freq_checkpoint_chunks)
        self.freq_residual_blocks = freq_residual_blocks
        self.correction_hidden = int(correction_hidden or 0)
        self.correction_layers = int(correction_layers or 2)
        self.correction_scale = float(correction_scale)
        self.transformer_tokens = int(transformer_tokens or 0)
        self.transformer_dim = int(transformer_dim or 128)
        self.transformer_layers = int(transformer_layers or 2)
        self.transformer_heads = int(transformer_heads or 4)
        self.transformer_scale = float(transformer_scale)
        self._last_correction_penalty = None
        self.m_re_min = m_re_min
        self.m_re_max = m_re_max
        self.m_im_min = m_im_min
        self.m_im_max = m_im_max
        self.kd_min = kd_min
        self.kd_max = kd_max
        self.log_grid_center = log_grid_center
        self.log_grid_scale = log_grid_scale

        # 3D CNN shape encoder
        self.shape_encoder = ShapeEncoder3D(
            embed_dim=shape_embed_dim,
            encoder_resolution=encoder_resolution,
            channels=encoder_channels,
        )

        # Global encoder: physics + shape -> conditioning vector
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        global_input = 4 + shape_embed_dim
        cond_dim = freq_hidden
        g_layers = [nn.Linear(global_input, global_hidden), act]
        for _ in range(global_layers - 2):
            g_layers.extend([nn.Linear(global_hidden, global_hidden), act])
        g_layers.append(nn.Linear(global_hidden, cond_dim))
        self.global_enc = nn.Sequential(*g_layers)

        # Per-frequency MLP input: full D_hat (18 real) + freq_coords (3) + cond
        d_hat_input = 18  # full 3x3 complex = 9 re + 9 im
        coord_input = 3 if freq_coords else 0
        freq_input = d_hat_input + coord_input + cond_dim

        if freq_residual_blocks:
            self.freq_mlp = ResidualFrequencyMLP(
                freq_input, freq_hidden, 18, int(freq_residual_blocks), activation
            )
        else:
            f_layers = [nn.Linear(freq_input, freq_hidden), act]
            for _ in range(freq_layers - 2):
                f_layers.extend([nn.Linear(freq_hidden, freq_hidden), act])
            f_layers.append(nn.Linear(freq_hidden, 18))
            self.freq_mlp = nn.Sequential(*f_layers)

        # Zero-init last layer -> M_hat = I at start
        last = self.freq_mlp.proj_out if freq_residual_blocks else self.freq_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        self.correction_mlp = None
        if self.correction_hidden > 0:
            # Zero-initialized additive residual. This preserves old checkpoints
            # exactly while allowing a small large-grid correction to be trained.
            corr_input = d_hat_input + 7 + cond_dim
            corr_layers = [nn.Linear(corr_input, self.correction_hidden), act]
            for _ in range(max(self.correction_layers, 2) - 2):
                corr_layers.extend([nn.Linear(self.correction_hidden, self.correction_hidden), act])
            corr_layers.append(nn.Linear(self.correction_hidden, 18))
            self.correction_mlp = nn.Sequential(*corr_layers)
            nn.init.zeros_(self.correction_mlp[-1].weight)
            nn.init.zeros_(self.correction_mlp[-1].bias)
            self.register_buffer(
                "_correction_scale",
                torch.tensor(self.correction_scale, dtype=torch.float32),
            )

        self.cond_dim = cond_dim
        self.coarse_transformer = None
        if self.transformer_tokens > 0:
            self.coarse_transformer = SpectralCoarseTransformer(
                input_dim=25,
                cond_dim=cond_dim,
                model_dim=self.transformer_dim,
                num_layers=self.transformer_layers,
                num_heads=self.transformer_heads,
                scale=self.transformer_scale,
            )

    @staticmethod
    def _signed_frequency_features(gx, gy, gz, device):
        fx = torch.fft.fftfreq(gx, d=1.0).to(device=device, dtype=torch.float32)
        fy = torch.fft.fftfreq(gy, d=1.0).to(device=device, dtype=torch.float32)
        fz = torch.fft.fftfreq(gz, d=1.0).to(device=device, dtype=torch.float32)
        sx, sy, sz = torch.meshgrid(fx, fy, fz, indexing='ij')
        sx = sx.reshape(-1)
        sy = sy.reshape(-1)
        sz = sz.reshape(-1)
        sx2 = sx.square()
        sy2 = sy.square()
        sz2 = sz.square()
        r2 = sx2 + sy2 + sz2
        return torch.stack([sx, sy, sz, sx2, sy2, sz2, r2], dim=1)

    @staticmethod
    def _coarse_frequency_features(D_real, gx, gy, gz, token_grid, device):
        nx = min(int(token_grid), int(gx))
        ny = min(int(token_grid), int(gy))
        nz = min(int(token_grid), int(gz))
        ix = torch.linspace(0, gx - 1, nx, device=device).round().long().unique()
        iy = torch.linspace(0, gy - 1, ny, device=device).round().long().unique()
        iz = torch.linspace(0, gz - 1, nz, device=device).round().long().unique()
        mx, my, mz = torch.meshgrid(ix, iy, iz, indexing='ij')
        flat = ((mx.reshape(-1) * gy) + my.reshape(-1)) * gz + mz.reshape(-1)

        fx = torch.fft.fftfreq(gx, d=1.0).to(device=device, dtype=torch.float32)[ix]
        fy = torch.fft.fftfreq(gy, d=1.0).to(device=device, dtype=torch.float32)[iy]
        fz = torch.fft.fftfreq(gz, d=1.0).to(device=device, dtype=torch.float32)[iz]
        sx, sy, sz = torch.meshgrid(fx, fy, fz, indexing='ij')
        sx = sx.reshape(-1)
        sy = sy.reshape(-1)
        sz = sz.reshape(-1)
        sx2 = sx.square()
        sy2 = sy.square()
        sz2 = sz.square()
        r2 = sx2 + sy2 + sz2
        coord = torch.stack([sx, sy, sz, sx2, sy2, sz2, r2], dim=1)
        return torch.cat([D_real[flat], coord], dim=1)

    def forward(self, m_re, m_im, kd, occupancy_grid, grid):
        """Compute global conditioning vector from physics + shape."""
        import math
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        m_re_t = torch.tensor([m_re], device=device, dtype=dtype)
        m_im_t = torch.tensor([m_im], device=device, dtype=dtype)
        kd_t = torch.tensor([kd], device=device, dtype=dtype)
        log_grid_t = torch.tensor([math.log(grid)], device=device, dtype=dtype)

        if self.normalize_inputs:
            eps = 1e-12
            m_re_t = (m_re_t - self.m_re_min) / max(self.m_re_max - self.m_re_min, eps)
            m_im_t = (m_im_t - self.m_im_min) / max(self.m_im_max - self.m_im_min, eps)
            kd_t = (kd_t - self.kd_min) / max(self.kd_max - self.kd_min, eps)
            log_grid_t = (log_grid_t - self.log_grid_center) / max(self.log_grid_scale, eps)

        if occupancy_grid.dim() == 4:
            occupancy_grid = occupancy_grid.unsqueeze(0)
        occupancy_grid = occupancy_grid.to(device=device, dtype=dtype)

        s_embed = self.shape_encoder(occupancy_grid)

        x = torch.cat([m_re_t.unsqueeze(-1), m_im_t.unsqueeze(-1),
                        kd_t.unsqueeze(-1), log_grid_t.unsqueeze(-1),
                        s_embed], dim=-1)

        cond = self.global_enc(x).squeeze(0)
        return cond

    def build_M_hat(self, cond, fft_matvec):
        """Build M_hat by applying per-frequency MLP to D_hat.

        D_hat is (6, gx, gy, gz) complex — upper triangle of symmetric 3x3.
        Reconstructed to full 3x3: 9 complex = 18 reals.
        Optionally adds normalized frequency coordinates (kx/gx, ky/gy, kz/gz).
        """
        D_hat_raw = fft_matvec.D_hat  # (6, gx, gy, gz) complex
        device = cond.device
        gx, gy, gz = D_hat_raw.shape[1], D_hat_raw.shape[2], D_hat_raw.shape[3]
        G = gx * gy * gz

        # Reconstruct full 3x3 from upper triangle: 0=xx,1=xy,2=xz,3=yy,4=yz,5=zz
        # Full: [[0,1,2],[1,3,4],[2,4,5]]
        D_full = torch.stack([
            D_hat_raw[0], D_hat_raw[1], D_hat_raw[2],  # row 0
            D_hat_raw[1], D_hat_raw[3], D_hat_raw[4],  # row 1
            D_hat_raw[2], D_hat_raw[4], D_hat_raw[5],  # row 2
        ], dim=0)  # (9, gx, gy, gz) complex

        D_flat = D_full.reshape(9, G).T  # (G, 9) complex
        D_real = torch.cat([D_flat.real.float(), D_flat.imag.float()], dim=1)  # (G, 18)
        D_real = D_real.to(device)

        if self.coarse_transformer is not None:
            token_features = self._coarse_frequency_features(
                D_real, gx, gy, gz, self.transformer_tokens, device)
            cond = cond + self.coarse_transformer(token_features, cond)

        # Build input features
        input_parts = [D_real]

        # Add normalized frequency coordinates
        if self.freq_coords:
            kx = torch.arange(gx, device=device, dtype=torch.float32) / gx  # [0, 1)
            ky = torch.arange(gy, device=device, dtype=torch.float32) / gy
            kz = torch.arange(gz, device=device, dtype=torch.float32) / gz
            # Meshgrid -> (G, 3)
            cx, cy, cz = torch.meshgrid(kx, ky, kz, indexing='ij')
            coords = torch.stack([cx.reshape(-1), cy.reshape(-1), cz.reshape(-1)], dim=1)
            input_parts.append(coords)

        chunk_size = int(getattr(self, "freq_chunk_size", 0) or 0)
        use_chunks = chunk_size > 0 and chunk_size < G
        self._last_correction_penalty = None

        if use_chunks:
            from torch.utils.checkpoint import checkpoint

            coord_full = input_parts[1] if self.freq_coords else None
            corr_features_full = None
            correction_penalty = None
            if self.correction_mlp is not None:
                corr_features_full = self._signed_frequency_features(gx, gy, gz, device)
            residual_chunks = []

            def run_mlp(x):
                return self.freq_mlp(x)

            def run_correction_mlp(x):
                return self.correction_mlp(x)

            for start in range(0, G, chunk_size):
                end = min(start + chunk_size, G)
                chunk_parts = [D_real[start:end]]
                if coord_full is not None:
                    chunk_parts.append(coord_full[start:end])
                chunk_parts.append(cond.unsqueeze(0).expand(end - start, -1))
                freq_input = torch.cat(chunk_parts, dim=1)
                if self.freq_checkpoint_chunks and self.training and torch.is_grad_enabled():
                    residual = checkpoint(run_mlp, freq_input, use_reentrant=False)
                else:
                    residual = self.freq_mlp(freq_input)
                if self.correction_mlp is not None:
                    corr_input = torch.cat([
                        D_real[start:end],
                        corr_features_full[start:end],
                        cond.unsqueeze(0).expand(end - start, -1),
                    ], dim=1)
                    scale = self._correction_scale.to(device=device, dtype=residual.dtype)
                    if self.freq_checkpoint_chunks and self.training and torch.is_grad_enabled():
                        correction = checkpoint(
                            run_correction_mlp, corr_input, use_reentrant=False
                        ) * scale
                    else:
                        correction = self.correction_mlp(corr_input) * scale
                    residual = residual + correction
                    chunk_weight = float(end - start) / float(G)
                    chunk_penalty = correction.float().square().mean() * chunk_weight
                    correction_penalty = (
                        chunk_penalty if correction_penalty is None
                        else correction_penalty + chunk_penalty
                    )
                residual_chunks.append(residual)
            residual = torch.cat(residual_chunks, dim=0)
            self._last_correction_penalty = correction_penalty
        else:
            # Broadcast conditioning
            cond_expanded = cond.unsqueeze(0).expand(G, -1)
            input_parts.append(cond_expanded)

            freq_input = torch.cat(input_parts, dim=1)
            residual = self.freq_mlp(freq_input)  # (G, 18)
            if self.correction_mlp is not None:
                corr_features = self._signed_frequency_features(gx, gy, gz, device)
                corr_input = torch.cat([D_real, corr_features, cond_expanded], dim=1)
                correction = self.correction_mlp(corr_input) * self._correction_scale.to(
                    device=device, dtype=residual.dtype)
                residual = residual + correction
                self._last_correction_penalty = correction.float().square().mean()

        import math
        residual = residual * (1.0 / math.sqrt(18))

        res_re = residual[:, :9].reshape(G, 3, 3)
        res_im = residual[:, 9:].reshape(G, 3, 3)
        M_residual = torch.complex(res_re, res_im)

        eye = torch.eye(3, dtype=M_residual.dtype, device=device)
        M_flat = M_residual + eye

        M_hat = M_flat.permute(1, 2, 0).reshape(3, 3, gx, gy, gz)

        if self.squared:
            M_hat = torch.einsum('ijxyz,jkxyz->ikxyz', M_hat, M_hat)

        return M_hat

    def make_precond_fn(self, cond, fft_matvec):
        """Create callable preconditioner for BiCGStab."""
        with torch.no_grad():
            M_hat = self.build_M_hat(cond, fft_matvec)
        M_hat = M_hat.to(dtype=torch.complex128, device='cpu')

        box = fft_matvec.box
        gx, gy, gz = 2*box[0].item(), 2*box[1].item(), 2*box[2].item()
        pos = fft_matvec.pos_shifted.cpu()
        pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
        N = fft_matvec.N
        n = fft_matvec.n

        def precond_fn(v):
            with torch.no_grad():
                v_grid = torch.zeros(3, gx, gy, gz, dtype=torch.complex128)
                v_grid[:, pi, pj, pk] = v.reshape(N, 3).T
                v_hat = torch.fft.fftn(v_grid, dim=(1, 2, 3))
                result_hat = torch.einsum('ijxyz,jxyz->ixyz', M_hat, v_hat)
                result_grid = torch.fft.ifftn(result_hat, dim=(1, 2, 3))
                return result_grid[:, pi, pj, pk].T.reshape(n)

        return precond_fn
