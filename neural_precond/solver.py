"""DDA solver with polynomial neural preconditioner.

Provides a single-call interface to solve A·x = b for DDA problems
using BiCGStab with PolyPrecond.

Usage:
    from neural_precond.solver import PolyPrecondSolver

    solver = PolyPrecondSolver("results/poly_k3_finetune/best_model.pt")
    x, info = solver.solve(positions, k, m, b)

    # Or with geometry reuse (multiple RHS for same particle):
    solver.setup_geometry(positions, k, m)
    x1, info1 = solver.solve_rhs(b1)
    x2, info2 = solver.solve_rhs(b2)
"""
import torch
import numpy as np

from core.fft_matvec import FFTMatVec
from krylov.bicgstab import bicgstab
try:
    from apps.generate_sai_dataset import build_sai_graph
except ImportError:
    build_sai_graph = None
from neural_precond.model import PolyPrecond


class PolyPrecondSolver:
    """DDA solver with polynomial neural preconditioner.

    The solver:
      1. Builds FFTMatVec for matrix-free A·v
      2. Builds a graph from geometry and encodes it via GNN
      3. Obtains K+1 polynomial coefficients (once per geometry)
      4. Solves A·x = b via BiCGStab with M(v) = p(A)·v (Horner scheme)

    Args:
        model_path: path to trained PolyPrecond checkpoint
        poly_degree: polynomial degree K (must match checkpoint)
        device: device for GNN inference ('cpu' or 'cuda:0')
        rtol: relative residual tolerance for BiCGStab
        max_iter: maximum BiCGStab iterations
    """

    def __init__(self, model_path, poly_degree=3, device='cpu',
                 rtol=1e-6, max_iter=5000):
        self.device = device
        self.rtol = rtol
        self.max_iter = max_iter
        self.poly_degree = poly_degree

        # Load model
        self.model = PolyPrecond(
            poly_degree=poly_degree,
            latent_size=64,
            message_passing_steps=6,
            activation='relu',
        )
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        # Will be set by setup_geometry
        self._fft_mv = None
        self._precond_fn = None
        self._n = None

    def setup_geometry(self, positions, k, m, d=1.0):
        """Prepare preconditioner for a given particle geometry.

        Call once per geometry. After this, solve_rhs() can be called
        multiple times for different RHS vectors (e.g. different
        incident directions/polarizations).

        Args:
            positions: (N, 3) integer grid positions of dipoles
            k: wavenumber (2*pi/lambda)
            m: complex refractive index
            d: interdipole spacing (default 1.0)
        """
        positions = np.asarray(positions, dtype=float)
        N = len(positions)
        self._n = 3 * N

        # FFTMatVec for matrix-free A·v
        self._fft_mv = FFTMatVec(
            torch.tensor(positions, dtype=torch.long),
            k, m, d, device='cpu'
        )

        # Build graph and encode geometry → polynomial coefficients
        if build_sai_graph is None:
            raise ImportError(
                "apps.generate_sai_dataset is unavailable; PolyPrecondSolver cannot build the graph"
            )
        data = build_sai_graph(positions, k, m, d)
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.global_features = data.global_features.float()

        with torch.no_grad():
            coefficients = self.model.encode_geometry(data)

        # Build preconditioner callable
        self._precond_fn = self.model.make_precond_fn(coefficients, self._fft_mv)

        # Store coefficients for inspection
        self._coefficients = coefficients.detach().cpu().numpy()

    def _matvec(self, v):
        """A·v operator for BiCGStab."""
        v2d = v.unsqueeze(1) if v.dim() == 1 else v
        return self._fft_mv(v2d).squeeze(1)

    def solve_rhs(self, b, x0=None, use_precond=True):
        """Solve A·x = b for a previously set up geometry.

        Args:
            b: (3N,) complex RHS vector (torch tensor or numpy array)
            x0: initial guess (default: zero)
            use_precond: if False, solve without preconditioner

        Returns:
            x: (3N,) complex solution (numpy array)
            info: dict with 'iterations', 'residuals', 'converged'
        """
        if self._fft_mv is None:
            raise RuntimeError("Call setup_geometry() first")

        # Convert to torch
        if isinstance(b, np.ndarray):
            b = torch.tensor(b, dtype=torch.complex128)
        b = b.to(dtype=torch.complex128)

        M = self._precond_fn if use_precond else None
        residuals, x = bicgstab(self._matvec, b, x0=x0, M=M,
                                rtol=self.rtol, max_iter=self.max_iter)

        iters = len(residuals) - 1
        converged = residuals[-1] < self.rtol

        return x.numpy(), {
            'iterations': iters,
            'residuals': residuals,
            'converged': converged,
            'final_residual': residuals[-1],
            'coefficients': self._coefficients,
        }

    def solve(self, positions, k, m, b, d=1.0, x0=None):
        """One-call solve: set up geometry and solve A·x = b.

        Args:
            positions: (N, 3) integer grid positions
            k: wavenumber
            m: complex refractive index
            b: (3N,) complex RHS vector
            d: interdipole spacing (default 1.0)
            x0: initial guess (default: zero)

        Returns:
            x: (3N,) complex solution (numpy array)
            info: dict with 'iterations', 'residuals', 'converged'
        """
        self.setup_geometry(positions, k, m, d)
        return self.solve_rhs(b, x0=x0)

    @property
    def coefficients(self):
        """Polynomial coefficients from last encode_geometry() call."""
        return self._coefficients
