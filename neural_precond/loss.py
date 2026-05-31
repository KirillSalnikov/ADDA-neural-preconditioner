"""Loss functions for neural preconditioner training.

Three families of loss:

1. **Probe loss** (original): ||M·A·z - z||² / ||z||²
   Measures average quality of M ≈ A⁻¹ on random vectors.
   Fast but doesn't correlate well with BiCGStab convergence.

2. **Differentiable BiCGStab loss** (new): unroll K iterations of BiCGStab,
   minimize log(||r_hat_K||² / ||r_hat_0||²).
   Directly optimizes what we care about — convergence speed.

3. **Spectral loss** (new): minimize spectral radius of (I - M·A) via
   power iteration. Lower spectral radius = faster Krylov convergence.
"""
import torch


def _truncate_M_hat_spatial(M_hat, max_radius=None, threshold_rel=0.0):
    """Apply differentiable spatial truncation to frequency-domain M_hat.

    The mask is selected from detached magnitudes, so gradients flow through
    kept spatial coefficients only. This matches ADDA export more closely than
    training a dense spectral inverse and pruning it afterwards.
    """
    threshold_rel = 0.0 if threshold_rel is None else float(threshold_rel)
    if max_radius is None and threshold_rel <= 0.0:
        return M_hat

    M_spatial = torch.fft.ifftn(M_hat, dim=(2, 3, 4))
    gx, gy, gz = M_spatial.shape[2:]
    device = M_spatial.device
    mask = torch.ones((gx, gy, gz), dtype=torch.bool, device=device)

    if max_radius is not None:
        r = int(max_radius)

        def signed_axis(size):
            idx = torch.arange(size, device=device)
            return torch.where(idx <= size // 2, idx, idx - size)

        x = signed_axis(gx).abs() <= r
        y = signed_axis(gy).abs() <= r
        z = signed_axis(gz).abs() <= r
        radius_mask = x[:, None, None] & y[None, :, None] & z[None, None, :]
        mask = mask & radius_mask

    if threshold_rel > 0.0:
        mag = torch.amax(torch.abs(M_spatial.detach()), dim=(0, 1))
        cutoff = threshold_rel * torch.amax(mag)
        mask = mask & (mag > cutoff)

    M_spatial = M_spatial * mask.to(M_spatial.dtype).unsqueeze(0).unsqueeze(0)
    return torch.fft.fftn(M_spatial, dim=(2, 3, 4))


def _blend_M_hat_identity(M_hat, identity_blend=1.0):
    """Return (1-lambda)I + lambda*M_hat in frequency domain."""
    identity_blend = 1.0 if identity_blend is None else float(identity_blend)
    if identity_blend == 1.0:
        return M_hat
    if identity_blend < 0.0 or identity_blend > 1.0:
        raise ValueError("identity_blend must be in [0, 1]")
    eye = torch.eye(3, dtype=M_hat.dtype, device=M_hat.device)
    return M_hat * identity_blend + (1.0 - identity_blend) * eye[:, :, None, None, None]


# ---------------------------------------------------------------------------
# Probe losses (original)
# ---------------------------------------------------------------------------

def precond_probe_loss(model, cache, fft_matvec, num_probes=10):
    """Compute probe-based preconditioner loss: ||M(A*z) - z||² / ||z||².

    For P random complex probe vectors z:
      1. w = A*z (FFT, no grad, complex128)
      2. M(w) via model.apply_precond (with grad, float32)
      3. loss = ||M(w) - z||² / ||z||²

    Args:
        model: NeuralPrecond instance (must be in train mode for gradients)
        cache: dict from model.encode_geometry() — geometry cache
        fft_matvec: FFTMatVec instance for computing A*z
        num_probes: number of random probe vectors P

    Returns:
        loss: scalar float32 — average normalized probe loss
    """
    n = cache['node_cache'].shape[0]  # number of DOF nodes (= 3*N_dipoles)
    device = cache['node_cache'].device

    # Random complex probe vectors (float32 for efficiency)
    z_re = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z_im = torch.randn(n, num_probes, device=device, dtype=torch.float32)

    # w = A*z via FFT (no gradients, high precision)
    with torch.no_grad():
        z_complex = torch.complex(z_re.double(), z_im.double())
        # FFTMatVec handles (3N, P) directly
        w = fft_matvec(z_complex.to(torch.complex128))  # (3N, P)
        w_re = w.real.float()  # (n, P)
        w_im = w.imag.float()  # (n, P)

    # Apply preconditioner M to each probe: M(w_p) for p=1..P
    total_loss = torch.tensor(0.0, device=device)

    for p in range(num_probes):
        # M(w_p)
        out_re, out_im = model.apply_precond(cache, w_re[:, p], w_im[:, p])

        # Residual: M(A*z) - z
        res_re = out_re - z_re[:, p]
        res_im = out_im - z_im[:, p]

        # ||residual||²
        res_norm_sq = res_re.pow(2).sum() + res_im.pow(2).sum()

        # ||z||²
        z_norm_sq = z_re[:, p].pow(2).sum() + z_im[:, p].pow(2).sum()

        total_loss = total_loss + res_norm_sq / (z_norm_sq + 1e-8)

    return total_loss / num_probes


def poly_precond_probe_loss(model, coefficients, fft_matvec, num_probes=10):
    """Compute probe-based loss for polynomial preconditioner.

    For P random complex probe vectors z:
      1. w = A·z (FFT, no grad, complex128)
      2. h = p(A)·w via Horner (with grad through coefficients)
      3. loss = ||h - z||² / ||z||²

    Gradients flow through the polynomial coefficients c_0...c_K only.
    FFTMatVec calls are exact arithmetic — no approximation error.

    The Horner evaluation is batched: FFTMatVec handles (n, P) directly,
    so all probes are processed in K matvec calls total.

    Args:
        model: PolyPrecond instance (unused in apply, but kept for API consistency)
        coefficients: (K+1,) complex — from model.encode_geometry(), WITH grad
        fft_matvec: FFTMatVec instance for computing A·z
        num_probes: number of random probe vectors P

    Returns:
        loss: scalar — average normalized probe loss
    """
    from neural_precond.model import PolyPrecond

    # Infer n from fft_matvec
    n = fft_matvec.n  # 3 * N_dipoles
    device = coefficients.device

    # Random complex probe vectors z: (n, P)
    z_re = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z_im = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z = torch.complex(z_re.double(), z_im.double()).to(torch.complex128)

    # w = A·z (no grad — A is fixed)
    with torch.no_grad():
        w = fft_matvec(z)  # (n, P) complex128

    # h = p(A)·w via Horner — grad flows through coefficients
    h = PolyPrecond.apply_poly(coefficients, fft_matvec, w)  # (n, P)

    # Target: z (cast to same dtype as h for comparison)
    z_target = z.to(h.dtype)

    # Residual: h - z
    residual = h - z_target  # (n, P)

    # ||residual||² / ||z||² per probe, then average
    res_norm_sq = (residual.real.pow(2) + residual.imag.pow(2)).sum(dim=0)  # (P,)
    z_norm_sq = (z_target.real.pow(2) + z_target.imag.pow(2)).sum(dim=0)    # (P,)

    loss = (res_norm_sq / (z_norm_sq + 1e-8)).mean()

    return loss


def conv_sai_probe_loss(model, kernel, fft_matvec, num_probes=5, chunk_size=2,
                        log_loss=False):
    """Compute probe-based SAI loss for ConvSAI_MLP.

    loss = E_z[ ||M·A·z - z||² / ||z||² ]

    M is applied via FFT convolution with the learned kernel.
    Gradients flow through: kernel → M_hat → result → loss.

    A·z is computed without gradients (A is fixed physics).
    The FFT of the kernel (build_M_hat) is differentiable via out-of-place scatter.

    Probes are processed in chunks to avoid OOM on large grids.

    Args:
        model: ConvSAI_MLP instance (for build_M_hat method)
        kernel: (n_stencil, 3, 3) complex — from model.forward(), WITH grad
        fft_matvec: FFTMatVec instance
        num_probes: number of random probe vectors
        chunk_size: number of probes per chunk (lower = less memory)

    Returns:
        loss: scalar — average normalized probe loss
    """
    N = fft_matvec.N
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    # Build M_hat from kernel (differentiable)
    M_hat = model.build_M_hat(kernel, fft_matvec)  # (3, 3, gx, gy, gz)

    box = fft_matvec.box
    gx = 2 * box[0].item()
    gy = 2 * box[1].item()
    gz = 2 * box[2].item()
    pos = fft_matvec.pos_shifted
    pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]

    M_hat_c128 = M_hat.to(torch.complex128)

    loss = torch.tensor(0.0, device=device)
    total_probes = 0

    for chunk_start in range(0, num_probes, chunk_size):
        P = min(chunk_size, num_probes - chunk_start)

        # Random probes z: (n, P) complex128
        z_re = torch.randn(n, P, device=device, dtype=torch.float32)
        z_im = torch.randn(n, P, device=device, dtype=torch.float32)
        z = torch.complex(z_re.double(), z_im.double()).to(torch.complex128)

        # w = A·z (no grad — A is fixed)
        with torch.no_grad():
            w = fft_matvec(z)  # (n, P) complex128

        # Scatter w to grid (no grad through w)
        w_reshaped = w.reshape(N, 3, P)  # (N, 3, P)
        w_grid = torch.zeros(3, P, gx, gy, gz,
                             dtype=torch.complex128, device=device)
        w_grid[:, :, pi, pj, pk] = w_reshaped.permute(1, 2, 0)  # (3, P, N)

        w_hat = torch.fft.fftn(w_grid, dim=(2, 3, 4))  # (3, P, gx, gy, gz)

        # M·w via FFT convolution (grad flows through M_hat → kernel)
        result_hat = torch.einsum('ijxyz,jpxyz->ipxyz', M_hat_c128, w_hat)

        result_grid = torch.fft.ifftn(result_hat, dim=(2, 3, 4))  # (3, P, gx, gy, gz)

        # Gather from dipole positions
        Mw = result_grid[:, :, pi, pj, pk]  # (3, P, N)
        Mw = Mw.permute(2, 0, 1).reshape(n, P)  # (n, P)

        # Loss: ||M·A·z - z||² / ||z||²
        z_target = z.to(Mw.dtype)
        residual = Mw - z_target
        res_norm_sq = (residual.real.pow(2) + residual.imag.pow(2)).sum(dim=0)  # (P,)
        z_norm_sq = (z_target.real.pow(2) + z_target.imag.pow(2)).sum(dim=0)    # (P,)

        ratio = res_norm_sq / (z_norm_sq + 1e-8)  # (P,)
        if log_loss:
            loss = loss + torch.log(ratio + 1e-8).sum()
        else:
            loss = loss + ratio.sum()
        total_probes += P

    return loss / total_probes


# ---------------------------------------------------------------------------
# Helper: apply M·v via FFT convolution (differentiable through M_hat)
# ---------------------------------------------------------------------------

def _apply_M_conv(M_hat, v, fft_matvec):
    """Apply preconditioner M·v via FFT convolution.

    Differentiable through M_hat (for training the kernel).
    v is treated as input data — no gradient through v needed.

    Args:
        M_hat: (3, 3, gx, gy, gz) complex — FFT of kernel, WITH grad
        v: (n,) complex128 — input vector
        fft_matvec: FFTMatVec instance (for grid geometry)

    Returns:
        (n,) complex128 — M·v
    """
    N = fft_matvec.N
    n = fft_matvec.n
    box = fft_matvec.box
    gx = 2 * box[0].item()
    gy = 2 * box[1].item()
    gz = 2 * box[2].item()
    pos = fft_matvec.pos_shifted
    pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
    device = M_hat.device

    # Scatter v to grid: (n,) → (N, 3) → (3, N) → grid
    v_3 = v.reshape(N, 3).T  # (3, N)
    v_grid = torch.zeros(3, gx, gy, gz, dtype=torch.complex128, device=device)
    v_grid[:, pi, pj, pk] = v_3

    # FFT → multiply by M_hat → IFFT
    v_hat = torch.fft.fftn(v_grid, dim=(1, 2, 3))
    result_hat = torch.einsum('ijxyz,jxyz->ixyz', M_hat.to(torch.complex128), v_hat)
    result_grid = torch.fft.ifftn(result_hat, dim=(1, 2, 3))

    # Gather from grid → flatten
    return result_grid[:, pi, pj, pk].T.reshape(n)


# ---------------------------------------------------------------------------
# Differentiable BiCGStab loss
# ---------------------------------------------------------------------------

def conv_sai_bicgstab_loss(model, kernel, fft_matvec, num_iters=30, num_rhs=2):
    """Differentiable BiCGStab loss for ConvSAI preconditioner.

    Unrolls num_iters of left-preconditioned BiCGStab and minimizes the
    log-ratio of final to initial preconditioned residual norms:

        loss = log(||r_hat_K||² / ||r_hat_0||²)

    Lower loss = faster convergence = better preconditioner.
    At perfect convergence in K steps, loss → -inf.

    Gradients flow through M·v applications at each iteration
    (via M_hat → kernel → MLP). A·v is detached (fixed physics).

    Memory: ~2 FFT convolutions stored per iteration × num_iters.
    For grid=12: ~180 MB for 30 iterations. For grid=24: ~1.2 GB.

    Args:
        model: ConvSAI_MLP instance
        kernel: (n_stencil, 3, 3) complex — from model.forward(), WITH grad
        fft_matvec: FFTMatVec instance
        num_iters: number of BiCGStab iterations to unroll (default 30)
        num_rhs: number of random right-hand sides to average over

    Returns:
        loss: scalar — log residual reduction (lower is better)
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    # Build M_hat once (differentiable through kernel)
    M_hat = model.build_M_hat(kernel, fft_matvec)

    total_loss = torch.tensor(0.0, device=device)
    eps = 1e-30

    for _ in range(num_rhs):
        # Random normalized RHS
        b = torch.randn(n, dtype=torch.complex128, device=device)
        b = b / torch.linalg.vector_norm(b)

        # Initial: x=0, r=b
        r = b.clone()

        # Preconditioned initial residual (grad through M)
        r_hat = _apply_M_conv(M_hat, r, fft_matvec)
        r_hat_0_norm_sq = (r_hat.real.pow(2).sum()
                           + r_hat.imag.pow(2).sum()).detach()

        # Shadow residual — fixed, detached (standard BiCGStab choice)
        r_tilde = r_hat.detach().clone()

        # BiCGStab scalars
        rho = torch.tensor(1.0, dtype=torch.complex128, device=device)
        alpha = torch.tensor(1.0, dtype=torch.complex128, device=device)
        omega = torch.tensor(1.0, dtype=torch.complex128, device=device)

        v_vec = torch.zeros(n, dtype=torch.complex128, device=device)
        p = torch.zeros(n, dtype=torch.complex128, device=device)

        for _ in range(num_iters):
            # rho_new = <r_tilde, r_hat>
            rho_new = torch.dot(r_tilde.conj(), r_hat)
            if rho_new.abs().item() < eps:
                break  # breakdown

            beta = (rho_new / (rho + eps)) * (alpha / (omega + eps))
            p = r_hat + beta * (p - omega * v_vec)

            # v = M · (A · p) — A detached, M with grad
            with torch.no_grad():
                Ap = fft_matvec(p.unsqueeze(1)).squeeze(1)
            v_vec = _apply_M_conv(M_hat, Ap, fft_matvec)

            sigma = torch.dot(r_tilde.conj(), v_vec)
            if sigma.abs().item() < eps:
                break
            alpha = rho_new / sigma

            s = r_hat - alpha * v_vec

            # t = M · (A · s) — A detached, M with grad
            with torch.no_grad():
                As = fft_matvec(s.unsqueeze(1)).squeeze(1)
            t = _apply_M_conv(M_hat, As, fft_matvec)

            tt = torch.dot(t.conj(), t)
            if tt.abs().item() < eps:
                break
            omega = torch.dot(t.conj(), s) / tt

            # Update preconditioned residual
            r_hat = s - omega * t
            rho = rho_new

        # Loss: log ratio of final to initial preconditioned residual
        r_hat_K_norm_sq = r_hat.real.pow(2).sum() + r_hat.imag.pow(2).sum()
        loss = torch.log(r_hat_K_norm_sq / (r_hat_0_norm_sq + eps) + eps)
        total_loss = total_loss + loss

    return total_loss / num_rhs


# ---------------------------------------------------------------------------
# Spectral loss via power iteration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Batched helpers (differentiable)
# ---------------------------------------------------------------------------

def _apply_M_batched(M_hat, v, fft_matvec, probe_chunk=2):
    """Apply preconditioner M to batched vectors v: (n, P) -> (n, P).

    Differentiable through M_hat. v can also carry gradients.
    probe_chunk: process this many probes at a time to limit memory.
    """
    N = fft_matvec.N
    n = fft_matvec.n
    P = v.shape[1] if v.dim() > 1 else 1
    if v.dim() == 1:
        v = v.unsqueeze(1)

    box = fft_matvec.box
    gx = 2 * box[0].item()
    gy = 2 * box[1].item()
    gz = 2 * box[2].item()
    pos = fft_matvec.pos_shifted
    pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
    device = M_hat.device

    M_hat_c128 = M_hat.to(torch.complex128)

    results = []
    for start in range(0, P, probe_chunk):
        end = min(start + probe_chunk, P)
        Pc = end - start

        v_c = v[:, start:end].reshape(N, 3, Pc)
        v_grid = torch.zeros(3, Pc, gx, gy, gz, dtype=torch.complex128, device=device)
        v_grid[:, :, pi, pj, pk] = v_c.permute(1, 2, 0)

        v_hat = torch.fft.fftn(v_grid, dim=(2, 3, 4))
        result_hat = torch.einsum('ijxyz,jpxyz->ipxyz', M_hat_c128, v_hat)
        result_grid = torch.fft.ifftn(result_hat, dim=(2, 3, 4))

        rc = result_grid[:, :, pi, pj, pk].permute(2, 0, 1).reshape(n, Pc)
        results.append(rc)

    return torch.cat(results, dim=1)


def _apply_A_batched(v, fft_matvec, probe_chunk=2):
    """Apply A·v for batched vectors: (n, P) -> (n, P).

    Differentiable through v. Uses out-of-place ops for autograd safety.
    probe_chunk: process this many probes at a time to limit memory.
    """
    N = fft_matvec.N
    n = fft_matvec.n
    P = v.shape[1] if v.dim() > 1 else 1
    if v.dim() == 1:
        v = v.unsqueeze(1)

    box = fft_matvec.box
    gx = 2 * box[0].item()
    gy = 2 * box[1].item()
    gz = 2 * box[2].item()
    pos = fft_matvec.pos_shifted
    pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
    device = v.device

    D = fft_matvec.D_hat  # (6, gx, gy, gz)

    results = []
    for start in range(0, P, probe_chunk):
        end = min(start + probe_chunk, P)
        Pc = end - start

        v_c = v[:, start:end].reshape(N, 3, Pc)
        v_grid = torch.zeros(3, Pc, gx, gy, gz, dtype=torch.complex128, device=device)
        v_grid[:, :, pi, pj, pk] = v_c.permute(1, 2, 0)

        v_hat = torch.fft.fftn(v_grid, dim=(2, 3, 4))

        y0 = D[0].unsqueeze(0)*v_hat[0] + D[1].unsqueeze(0)*v_hat[1] + D[2].unsqueeze(0)*v_hat[2]
        y1 = D[1].unsqueeze(0)*v_hat[0] + D[3].unsqueeze(0)*v_hat[1] + D[4].unsqueeze(0)*v_hat[2]
        y2 = D[2].unsqueeze(0)*v_hat[0] + D[4].unsqueeze(0)*v_hat[1] + D[5].unsqueeze(0)*v_hat[2]
        y_hat = torch.stack([y0, y1, y2], dim=0)

        y_grid = torch.fft.ifftn(y_hat, dim=(2, 3, 4))
        conv = y_grid[:, :, pi, pj, pk].permute(2, 0, 1).reshape(n, Pc)

        results.append(v[:, start:end] - fft_matvec.alpha * conv)

    return torch.cat(results, dim=1)


# ---------------------------------------------------------------------------
# Adversarial probe loss
# ---------------------------------------------------------------------------

def _fft_matvec_chunked(fft_matvec, v, chunk=2):
    """Apply fft_matvec to (n, P) in chunks to limit memory."""
    P = v.shape[1] if v.dim() > 1 else 1
    if P <= chunk:
        return fft_matvec(v)
    results = []
    for s in range(0, P, chunk):
        results.append(fft_matvec(v[:, s:min(s + chunk, P)]))
    return torch.cat(results, dim=1)


def conv_sai_adversarial_probe_loss(model, kernel, fft_matvec,
                                     num_probes=5, adversarial_iters=10,
                                     probe_chunk=2, log_loss=False,
                                     m_hat_max_radius=None,
                                     m_hat_threshold_rel=0.0,
                                     m_hat_identity_blend=1.0):
    """Adversarial probe loss: find worst-case z via power iteration on (I-MA).

    Phase 1 (NO grad): power iteration finds z that maximizes ||MAz - z||/||z||.
    Phase 2 (WITH grad): compute probe loss on these worst-case vectors.

    Focuses training on the worst eigenmodes of (I - MA), unlike random probes
    which optimize the average (Frobenius norm).

    All batched FFT operations are chunked by probe_chunk to avoid OOM on
    large grids (e.g. grid=128 → doubled 256³).

    Args:
        model: ConvSAI_MLP instance
        kernel: (n_stencil, 3, 3) complex — from model.forward(), WITH grad
        fft_matvec: FFTMatVec instance
        num_probes: number of adversarial vectors
        adversarial_iters: power iteration steps to find worst-case z
        probe_chunk: number of probes per FFT chunk (lower = less memory)

    Returns:
        loss: scalar — probe loss on adversarial vectors
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    M_hat = model.build_M_hat(kernel, fft_matvec)
    M_hat = _blend_M_hat_identity(M_hat, m_hat_identity_blend)
    M_hat = _truncate_M_hat_spatial(
        M_hat,
        max_radius=m_hat_max_radius,
        threshold_rel=m_hat_threshold_rel,
    )

    # Phase 1: Find adversarial vectors via power iteration (NO grad)
    with torch.no_grad():
        M_hat_det = M_hat.detach()
        z = torch.randn(n, num_probes, dtype=torch.complex128, device=device)
        z = z / torch.linalg.vector_norm(z, dim=0, keepdim=True)

        for _ in range(adversarial_iters):
            Az = _fft_matvec_chunked(fft_matvec, z, probe_chunk)
            MAz = _apply_M_batched(M_hat_det, Az, fft_matvec, probe_chunk)
            w = z - MAz
            norms = torch.linalg.vector_norm(w, dim=0, keepdim=True)
            z = w / (norms + 1e-30)

    z = z.detach()

    # Phase 2: Compute probe loss on adversarial vectors (WITH grad)
    with torch.no_grad():
        w = _fft_matvec_chunked(fft_matvec, z, probe_chunk)

    # M·(A·z) with grad through M_hat (chunked)
    Mw = _apply_M_batched(M_hat, w, fft_matvec, probe_chunk)

    # Loss: ||MA·z - z||² / ||z||²
    z_target = z.to(Mw.dtype)
    residual = Mw - z_target
    res_norm_sq = (residual.real.pow(2) + residual.imag.pow(2)).sum(dim=0)
    z_norm_sq = (z_target.real.pow(2) + z_target.imag.pow(2)).sum(dim=0)

    ratio = res_norm_sq / (z_norm_sq + 1e-8)
    if log_loss:
        loss = torch.log(ratio + 1e-8).mean()
    else:
        loss = ratio.mean()
    return loss


# ---------------------------------------------------------------------------
# Right preconditioning probe loss
# ---------------------------------------------------------------------------

def conv_sai_right_probe_loss(model, kernel, fft_matvec, num_probes=5):
    """Right preconditioning probe loss: ||A·M·z - z||² / ||z||².

    For right preconditioning, we want A·M ≈ I (vs left: M·A ≈ I).
    Right-preconditioned BiCGStab solves A·M·y = b, then x = M·y.

    May work better than left preconditioning for nonsymmetric systems
    where left and right spectra differ significantly.

    Gradient flow: z → M(z) [through M_hat] → A(Mz) [through Mz → M_hat].
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    M_hat = model.build_M_hat(kernel, fft_matvec)

    # Random probes z: (n, P) complex128 (no grad)
    z_re = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z_im = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z = torch.complex(z_re.double(), z_im.double()).to(torch.complex128)

    # Step 1: M·z (grad through M_hat)
    Mz = _apply_M_batched(M_hat, z, fft_matvec)  # (n, P)

    # Step 2: A·(M·z) (grad flows through Mz → M_hat)
    AMz = _apply_A_batched(Mz, fft_matvec)  # (n, P)

    # Loss: ||A·M·z - z||² / ||z||²
    z_target = z.to(AMz.dtype)
    residual = AMz - z_target
    res_norm_sq = (residual.real.pow(2) + residual.imag.pow(2)).sum(dim=0)
    z_norm_sq = (z_target.real.pow(2) + z_target.imag.pow(2)).sum(dim=0)

    loss = (res_norm_sq / (z_norm_sq + 1e-8)).mean()
    return loss


# ---------------------------------------------------------------------------
# GMRES loss (unrolled Arnoldi process)
# ---------------------------------------------------------------------------

def conv_sai_gmres_loss(model, kernel, fft_matvec,
                         gmres_iters=10, num_rhs=2):
    """Differentiable GMRES loss for ConvSAI preconditioner.

    Unrolls K steps of left-preconditioned GMRES (Arnoldi process) and
    minimizes the final residual norm via least squares on the Hessenberg matrix.

    GMRES is more stable than BiCGStab for backpropagation:
    - Residual decreases monotonically (no erratic oscillations)
    - No divisions by potentially small scalars (rho, omega)
    - Well-conditioned least squares problem

    loss = log(||r_K|| / ||r_0||)

    Gradients flow through M applications at each Arnoldi step.
    A·v is detached (fixed physics). Memory: ~1 M application per Arnoldi step.

    Args:
        model: ConvSAI_MLP instance
        kernel: (n_stencil, 3, 3) complex — from model.forward(), WITH grad
        fft_matvec: FFTMatVec instance
        gmres_iters: number of Arnoldi steps K (default 10)
        num_rhs: number of random RHS to average over

    Returns:
        loss: scalar — log residual reduction (lower is better)
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    M_hat = model.build_M_hat(kernel, fft_matvec)

    total_loss = torch.tensor(0.0, device=device)
    counted = 0
    k = gmres_iters

    for _ in range(num_rhs):
        # Random normalized RHS
        b = torch.randn(n, dtype=torch.complex128, device=device)
        b = b / torch.linalg.vector_norm(b)

        # r0 = M·b (grad through M_hat)
        r0 = _apply_M_conv(M_hat, b.detach(), fft_matvec)
        beta = torch.linalg.vector_norm(r0)

        if beta.real.item() < 1e-30:
            continue

        # Arnoldi process — all H entries stored as Python lists (out-of-place)
        V = [r0 / beta]
        H_columns = []  # list of column tensors, built out-of-place

        k_actual = k
        for j in range(k):
            # w = M · (A · v_j)  — A detached, M with grad
            with torch.no_grad():
                Avj = fft_matvec(V[j].detach().unsqueeze(1)).squeeze(1)
            w = _apply_M_conv(M_hat, Avj, fft_matvec)

            # Modified Gram-Schmidt — collect entries in Python list
            h_col_entries = []
            for i in range(j + 1):
                h_ij = torch.dot(V[i].conj(), w)
                h_col_entries.append(h_ij)
                w = w - h_ij * V[i]

            h_norm = torch.linalg.vector_norm(w)
            h_col_entries.append(h_norm)

            # Pad column to k+1 entries with zeros
            zero = torch.tensor(0.0, dtype=torch.complex128, device=device)
            while len(h_col_entries) < k + 1:
                h_col_entries.append(zero)

            H_columns.append(torch.stack(h_col_entries))  # out-of-place

            if h_norm.real.item() < 1e-14:
                k_actual = j + 1
                break

            V.append(w / h_norm)

        # Build H_k out-of-place: (k_actual+1, k_actual)
        H_k = torch.stack(H_columns[:k_actual], dim=1)[:k_actual + 1, :]

        # Solve least squares: min ||beta * e1 - H_k @ y||
        e1 = torch.cat([beta.unsqueeze(0),
                         torch.zeros(k_actual, dtype=torch.complex128, device=device)])

        y = torch.linalg.lstsq(H_k, e1.unsqueeze(1)).solution.squeeze(1)

        # Residual in Hessenberg space
        residual_vec = e1 - H_k @ y
        res_norm = torch.linalg.vector_norm(residual_vec)

        # Loss: log ratio (detach beta in denominator)
        loss = torch.log(res_norm / (beta.detach() + 1e-30) + 1e-30)
        total_loss = total_loss + loss
        counted += 1

    if counted == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / counted


def conv_sai_spectral_loss(model, kernel, fft_matvec,
                           num_power_iters=20, num_vectors=3):
    """Spectral loss: minimize spectral radius of (I - M·A).

    The convergence rate of any Krylov method is bounded by the spectral
    radius rho(I - M·A). Lower rho → faster convergence.

    Algorithm:
      1. Power iteration (WITHOUT grad) to find the dominant eigenvector
         of (I - M·A) — the worst-case direction for the preconditioner.
      2. One final application of (I - M·A) WITH grad through M.
      3. Loss = ||result||² ≈ |lambda_max|² (spectral radius squared).

    This focuses the gradient on reducing the WORST eigenvalue, unlike
    probe loss which optimizes the average (Frobenius norm).

    Args:
        model: ConvSAI_MLP instance
        kernel: (n_stencil, 3, 3) complex — from model.forward(), WITH grad
        fft_matvec: FFTMatVec instance
        num_power_iters: iterations of power method (more = better eigenvalue estimate)
        num_vectors: number of independent starting vectors (finds multiple eigenvalues)

    Returns:
        loss: scalar — estimated spectral radius squared of (I - M·A)
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    M_hat = model.build_M_hat(kernel, fft_matvec)

    total_loss = torch.tensor(0.0, device=device)

    for _ in range(num_vectors):
        # Random unit starting vector
        v = torch.randn(n, dtype=torch.complex128, device=device)
        v = v / torch.linalg.vector_norm(v)

        # Power iteration: find dominant eigenvector of (I - M·A)
        # All WITHOUT gradients — just finding the worst direction
        with torch.no_grad():
            M_hat_detached = M_hat.detach()
            for _ in range(num_power_iters):
                # w = (I - M·A)·v
                Av = fft_matvec(v.unsqueeze(1)).squeeze(1)
                MAv = _apply_M_conv(M_hat_detached, Av, fft_matvec)
                w = v - MAv

                w_norm = torch.linalg.vector_norm(w)
                if w_norm.item() < 1e-30:
                    break
                v = w / w_norm

        # Final application WITH grad through M_hat
        # v is the approximate dominant eigenvector (detached)
        v = v.detach()
        with torch.no_grad():
            Av = fft_matvec(v.unsqueeze(1)).squeeze(1)
        MAv = _apply_M_conv(M_hat, Av, fft_matvec)  # grad through M_hat
        w = v - MAv  # (I - M·A)·v

        # ||w||² ≈ |lambda_max|² — spectral radius squared
        spectral_sq = w.real.pow(2).sum() + w.imag.pow(2).sum()
        total_loss = total_loss + spectral_sq

    return total_loss / num_vectors

def conv_sai_krylov_loss(model, kernel, fft_matvec, num_iters=10, num_rhs=2,
                         m_hat_max_radius=None, m_hat_threshold_rel=0.0,
                         m_hat_identity_blend=1.0):
    """True-residual Krylov path loss.

    The previous variant propagated the preconditioned residual
    ``r <- (I - M A) r`` starting from ``r0 = M b``. That objective is
    vulnerable to the degenerate solution ``M ~= 0``: the model can make the
    preconditioned residual tiny without improving the original linear system.

    This loss follows the true residual of a preconditioned Richardson/Krylov
    step instead:

        x_{k+1} = x_k + M r_k
        r_{k+1} = r_k - A M r_k

    Now ``M = 0`` leaves ``r`` unchanged and receives no artificial reward.
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device
    M_hat = model.build_M_hat(kernel, fft_matvec)
    M_hat = _blend_M_hat_identity(M_hat, m_hat_identity_blend)
    M_hat = _truncate_M_hat_spatial(
        M_hat,
        max_radius=m_hat_max_radius,
        threshold_rel=m_hat_threshold_rel,
    )

    total_loss = torch.tensor(0.0, device=device)
    for _ in range(num_rhs):
        b = torch.randn(n, dtype=torch.complex128, device=device)
        b = b / torch.linalg.vector_norm(b)

        r = b
        r0_norm = torch.linalg.vector_norm(r).detach()
        path_loss = torch.tensor(0.0, device=device)
        for _ in range(num_iters):
            Mr = _apply_M_conv(M_hat, r, fft_matvec)
            AMr = _apply_A_batched(Mr.unsqueeze(1), fft_matvec).squeeze(1)
            r = r - AMr

            ratio = torch.linalg.vector_norm(r) / (r0_norm + 1e-30)
            path_loss = path_loss + torch.log(ratio + 1e-12)

        total_loss = total_loss + path_loss / num_iters

    return total_loss / num_rhs


def conv_sai_anchored_krylov_loss(model, kernel, fft_matvec,
                                  num_iters=4, num_rhs=1,
                                  num_probes=1, probe_chunk=1,
                                  krylov_weight=1.0,
                                  probe_weight=0.5,
                                  right_probe_weight=0.2,
                                  m_hat_max_radius=None,
                                  m_hat_threshold_rel=0.0,
                                  m_hat_identity_blend=1.0):
    """True-residual Krylov loss with inverse-quality anchors.

    The Krylov term optimizes residual reduction of the original system:
    ``r <- r - A(M r)``. The probe anchors keep the learned operator close to
    an actual inverse from both sides:

      - left:  ``M A z ~= z``
      - right: ``A M z ~= z``

    All terms use log normalized residuals, so ``M ~= 0`` scores around zero
    instead of looking artificially good.
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device
    M_hat = model.build_M_hat(kernel, fft_matvec)
    M_hat = _blend_M_hat_identity(M_hat, m_hat_identity_blend)
    M_hat = _truncate_M_hat_spatial(
        M_hat,
        max_radius=m_hat_max_radius,
        threshold_rel=m_hat_threshold_rel,
    )

    total_krylov = torch.tensor(0.0, device=device)
    for _ in range(num_rhs):
        b = torch.randn(n, dtype=torch.complex128, device=device)
        b = b / torch.linalg.vector_norm(b)

        r = b
        r0_norm = torch.linalg.vector_norm(r).detach()
        path_loss = torch.tensor(0.0, device=device)
        for _ in range(num_iters):
            Mr = _apply_M_conv(M_hat, r, fft_matvec)
            AMr = _apply_A_batched(Mr.unsqueeze(1), fft_matvec,
                                   probe_chunk=probe_chunk).squeeze(1)
            r = r - AMr

            ratio = torch.linalg.vector_norm(r) / (r0_norm + 1e-30)
            path_loss = path_loss + torch.log(ratio + 1e-12)

        total_krylov = total_krylov + path_loss / num_iters
    krylov_loss = total_krylov / num_rhs

    left_loss = torch.tensor(0.0, device=device)
    right_loss = torch.tensor(0.0, device=device)
    if num_probes > 0 and (probe_weight != 0.0 or right_probe_weight != 0.0):
        z = torch.randn(n, num_probes, dtype=torch.complex128, device=device)
        z = z / torch.linalg.vector_norm(z, dim=0, keepdim=True)

        if probe_weight != 0.0:
            with torch.no_grad():
                Az = _fft_matvec_chunked(fft_matvec, z, chunk=probe_chunk)
            MAz = _apply_M_batched(M_hat, Az, fft_matvec, probe_chunk=probe_chunk)
            left_res = MAz - z.to(MAz.dtype)
            left_num = (left_res.real.square() + left_res.imag.square()).sum(dim=0)
            left_den = (z.real.square() + z.imag.square()).sum(dim=0)
            left_loss = torch.log(left_num / (left_den + 1e-30) + 1e-12).mean()

        if right_probe_weight != 0.0:
            Mz = _apply_M_batched(M_hat, z, fft_matvec, probe_chunk=probe_chunk)
            AMz = _apply_A_batched(Mz, fft_matvec, probe_chunk=probe_chunk)
            right_res = AMz - z.to(AMz.dtype)
            right_num = (right_res.real.square() + right_res.imag.square()).sum(dim=0)
            right_den = (z.real.square() + z.imag.square()).sum(dim=0)
            right_loss = torch.log(right_num / (right_den + 1e-30) + 1e-12).mean()

    return (krylov_weight * krylov_loss
            + probe_weight * left_loss
            + right_probe_weight * right_loss)


def _planewave_rhs(fft_matvec, prop_dir, pol_dir, device):
    """Build ADDA default plane-wave RHS for a given polarization."""
    positions = fft_matvec.positions.to(device=device, dtype=torch.float64)
    real_dtype = torch.float64

    prop = torch.tensor(prop_dir, dtype=real_dtype, device=device)
    prop = prop / torch.linalg.vector_norm(prop)

    pol = torch.tensor(pol_dir, dtype=real_dtype, device=device)
    pol = pol - torch.dot(pol, prop) * prop
    pol = pol / torch.linalg.vector_norm(pol)

    phase_arg = fft_matvec.k * fft_matvec.d * (positions @ prop)
    phase = torch.exp(1j * phase_arg).to(torch.complex128)

    b = torch.zeros(positions.shape[0], 3, dtype=torch.complex128, device=device)
    b[:, 0] = pol[0].to(torch.complex128) * phase
    b[:, 1] = pol[1].to(torch.complex128) * phase
    b[:, 2] = pol[2].to(torch.complex128) * phase
    b = b.reshape(-1)
    return b / (torch.linalg.vector_norm(b) + 1e-30)


def conv_sai_planewave_krylov_loss(model, kernel, fft_matvec,
                                   num_iters=8,
                                   num_probes=1, probe_chunk=1,
                                   krylov_weight=1.0,
                                   probe_weight=0.5,
                                   right_probe_weight=0.5,
                                   m_hat_max_radius=None,
                                   m_hat_threshold_rel=0.0,
                                   m_hat_identity_blend=1.0):
    """True-residual Krylov loss on ADDA's default plane-wave RHS.

    ADDA's default fixed-orientation solve uses propagation along z and two
    transverse incident polarizations. Random-RHS losses can improve a generic
    surrogate while still leaving the actual plane-wave solve stuck; this loss
    optimizes the residual path for those real right-hand sides directly.
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device
    M_hat = model.build_M_hat(kernel, fft_matvec)
    M_hat = _blend_M_hat_identity(M_hat, m_hat_identity_blend)
    M_hat = _truncate_M_hat_spatial(
        M_hat,
        max_radius=m_hat_max_radius,
        threshold_rel=m_hat_threshold_rel,
    )

    rhs_list = [
        _planewave_rhs(fft_matvec, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), device),
        _planewave_rhs(fft_matvec, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), device),
    ]

    total_krylov = torch.tensor(0.0, device=device)
    for b in rhs_list:
        r = b
        r0_norm = torch.linalg.vector_norm(r).detach()
        path_loss = torch.tensor(0.0, device=device)
        for _ in range(num_iters):
            Mr = _apply_M_conv(M_hat, r, fft_matvec)
            AMr = _apply_A_batched(Mr.unsqueeze(1), fft_matvec,
                                   probe_chunk=probe_chunk).squeeze(1)
            r = r - AMr
            ratio = torch.linalg.vector_norm(r) / (r0_norm + 1e-30)
            path_loss = path_loss + torch.log(ratio + 1e-12)
        total_krylov = total_krylov + path_loss / num_iters
    krylov_loss = total_krylov / len(rhs_list)

    left_loss = torch.tensor(0.0, device=device)
    right_loss = torch.tensor(0.0, device=device)
    if num_probes > 0 and (probe_weight != 0.0 or right_probe_weight != 0.0):
        z = torch.randn(n, num_probes, dtype=torch.complex128, device=device)
        z = z / torch.linalg.vector_norm(z, dim=0, keepdim=True)

        if probe_weight != 0.0:
            with torch.no_grad():
                Az = _fft_matvec_chunked(fft_matvec, z, chunk=probe_chunk)
            MAz = _apply_M_batched(M_hat, Az, fft_matvec, probe_chunk=probe_chunk)
            left_res = MAz - z.to(MAz.dtype)
            left_num = (left_res.real.square() + left_res.imag.square()).sum(dim=0)
            left_den = (z.real.square() + z.imag.square()).sum(dim=0)
            left_loss = torch.log(left_num / (left_den + 1e-30) + 1e-12).mean()

        if right_probe_weight != 0.0:
            Mz = _apply_M_batched(M_hat, z, fft_matvec, probe_chunk=probe_chunk)
            AMz = _apply_A_batched(Mz, fft_matvec, probe_chunk=probe_chunk)
            right_res = AMz - z.to(AMz.dtype)
            right_num = (right_res.real.square() + right_res.imag.square()).sum(dim=0)
            right_den = (z.real.square() + z.imag.square()).sum(dim=0)
            right_loss = torch.log(right_num / (right_den + 1e-30) + 1e-12).mean()

    return (krylov_weight * krylov_loss
            + probe_weight * left_loss
            + right_probe_weight * right_loss)


def conv_sai_planewave_bicgstab_loss(model, kernel, fft_matvec,
                                     num_iters=12,
                                     num_probes=1, probe_chunk=1,
                                     krylov_weight=1.0,
                                     probe_weight=0.5,
                                     right_probe_weight=0.5,
                                     m_hat_max_radius=None,
                                     m_hat_threshold_rel=0.0,
                                     m_hat_identity_blend=1.0):
    """Unrolled left-preconditioned BiCGStab loss on ADDA plane-wave RHS."""
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device
    M_hat = model.build_M_hat(kernel, fft_matvec)
    M_hat = _blend_M_hat_identity(M_hat, m_hat_identity_blend)
    M_hat = _truncate_M_hat_spatial(
        M_hat,
        max_radius=m_hat_max_radius,
        threshold_rel=m_hat_threshold_rel,
    )
    eps = 1e-30

    rhs_list = [
        _planewave_rhs(fft_matvec, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), device),
        _planewave_rhs(fft_matvec, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), device),
    ]

    total = torch.tensor(0.0, device=device)
    for b in rhs_list:
        r = b.clone()
        r_hat = _apply_M_conv(M_hat, r, fft_matvec)
        r_hat_0 = (r_hat.real.square().sum() + r_hat.imag.square().sum()).detach()
        r_tilde = r_hat.detach().clone()

        rho = torch.tensor(1.0, dtype=torch.complex128, device=device)
        alpha = torch.tensor(1.0, dtype=torch.complex128, device=device)
        omega = torch.tensor(1.0, dtype=torch.complex128, device=device)
        v_vec = torch.zeros(n, dtype=torch.complex128, device=device)
        p = torch.zeros(n, dtype=torch.complex128, device=device)

        for _ in range(num_iters):
            rho_new = torch.dot(r_tilde.conj(), r_hat)
            beta = (rho_new / (rho + eps)) * (alpha / (omega + eps))
            p = r_hat + beta * (p - omega * v_vec)

            Ap = _fft_matvec_chunked(fft_matvec, p.unsqueeze(1),
                                     chunk=probe_chunk).squeeze(1)
            v_vec = _apply_M_conv(M_hat, Ap, fft_matvec)
            alpha = rho_new / (torch.dot(r_tilde.conj(), v_vec) + eps)
            s = r_hat - alpha * v_vec

            As = _fft_matvec_chunked(fft_matvec, s.unsqueeze(1),
                                     chunk=probe_chunk).squeeze(1)
            t = _apply_M_conv(M_hat, As, fft_matvec)
            omega = torch.dot(t.conj(), s) / (torch.dot(t.conj(), t) + eps)
            r_hat = s - omega * t
            rho = rho_new

        r_hat_k = r_hat.real.square().sum() + r_hat.imag.square().sum()
        total = total + torch.log(r_hat_k / (r_hat_0 + eps) + eps)

    bicg_loss = total / len(rhs_list)

    left_loss = torch.tensor(0.0, device=device)
    right_loss = torch.tensor(0.0, device=device)
    if num_probes > 0 and (probe_weight != 0.0 or right_probe_weight != 0.0):
        z = torch.randn(n, num_probes, dtype=torch.complex128, device=device)
        z = z / torch.linalg.vector_norm(z, dim=0, keepdim=True)

        if probe_weight != 0.0:
            with torch.no_grad():
                Az = _fft_matvec_chunked(fft_matvec, z, chunk=probe_chunk)
            MAz = _apply_M_batched(M_hat, Az, fft_matvec, probe_chunk=probe_chunk)
            left_res = MAz - z.to(MAz.dtype)
            left_num = (left_res.real.square() + left_res.imag.square()).sum(dim=0)
            left_den = (z.real.square() + z.imag.square()).sum(dim=0)
            left_loss = torch.log(left_num / (left_den + 1e-30) + 1e-12).mean()

        if right_probe_weight != 0.0:
            Mz = _apply_M_batched(M_hat, z, fft_matvec, probe_chunk=probe_chunk)
            AMz = _apply_A_batched(Mz, fft_matvec, probe_chunk=probe_chunk)
            right_res = AMz - z.to(AMz.dtype)
            right_num = (right_res.real.square() + right_res.imag.square()).sum(dim=0)
            right_den = (z.real.square() + z.imag.square()).sum(dim=0)
            right_loss = torch.log(right_num / (right_den + 1e-30) + 1e-12).mean()

    return (krylov_weight * bicg_loss
            + probe_weight * left_loss
            + right_probe_weight * right_loss)
