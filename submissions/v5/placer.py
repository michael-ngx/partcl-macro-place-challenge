"""
v5: Pure-PyTorch ePlace-style analytical placer (closer DREAMPlace clone).

Implements the two pieces my earlier v1 GP was missing:

1. **FFT-based ePlace electrostatic density** (Cheng et al. 2018,
   "RePlAce: Advancing Solution Quality and Routability Validation in
   Global Placement", IEEE TCAD 2019).

   The placer treats each cell as a charge with magnitude proportional
   to its area. The density penalty is the electrostatic potential
   energy of the system, computed by solving Poisson's equation in the
   frequency domain via 2D real FFT. The gradient is the resulting
   electric field at each cell's center.

   Concretely:
     - Bin the canvas into n_bins × n_bins; build a charge map ρ(b)
       where each bin gets the area overlap with each cell's footprint
       minus a constant background (so the system is neutral on
       average).
     - Solve −∇²φ = ρ for the potential φ. In the DCT/FFT basis the
       solution is φ̂_{u,v} = ρ̂_{u,v} / (a_u² + b_v²) for non-zero
       (u,v); the (0,0) component is dropped (gauge fix).
     - Density energy = (1/2) Σ_b ρ_b φ_b. Gradient on cell c at
       position (x,y) is the negative electric field −∇φ at (x,y).
       Implemented by autograd over the inverse-FFT result.

2. **Nesterov accelerated gradient** with simple Barzilai–Borwein-style
   adaptive learning rate and a multiplicative λ schedule that
   responds to the HPWL change ratio (ePlace style).

Stages:
  A. Optional warm start: read initial.plc positions (already-good).
  B. Differentiable global placement (smooth WA wirelength + electric
     density, escalating λ) — Nesterov/Adam optimizer.
  C. Greedy minimum-displacement legalization (largest-first spiral).
  D. (Optional) v3 soft-macro Jacobi update.
  E. (Optional) v4 proxy-aware coordinate descent.

The same code path runs on CPU and CUDA (`device='cuda' if available
else 'cpu'`).
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ─────────────────────── shared utilities (re-used from v3/v4) ─────── #

def _load_plc(name: str):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / d
            / "netlist"
            / "output_CT_Grouping"
        )
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(
                str(base / "netlist.pb.txt"), str(base / "initial.plc")
            )
            return plc
    return None


def _legalize_with_order(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    cw: float,
    ch: float,
    n_hard: int,
    fixed_pos: np.ndarray,
    order: List[int],
) -> np.ndarray:
    pos = pos.copy()
    for i in range(n_hard):
        if not movable[i]:
            pos[i] = fixed_pos[i]

    gap = 0.001
    sx = sizes[:n_hard, 0]
    sy = sizes[:n_hard, 1]
    sep_x_mat = (sx[:, None] + sx[None, :]) / 2 + gap
    sep_y_mat = (sy[:, None] + sy[None, :]) / 2 + gap
    half_w = sx / 2
    half_h = sy / 2

    placed = np.zeros(n_hard, dtype=bool)
    for i in range(n_hard):
        if not movable[i]:
            placed[i] = True

    def has_overlap(idx, x, y):
        if not placed.any():
            return False
        dx = np.abs(x - pos[:n_hard, 0])
        dy = np.abs(y - pos[:n_hard, 1])
        o = (dx < sep_x_mat[idx]) & (dy < sep_y_mat[idx]) & placed
        o[idx] = False
        return bool(o.any())

    for idx in order:
        if placed[idx]:
            continue
        x0 = float(np.clip(pos[idx, 0], half_w[idx], cw - half_w[idx]))
        y0 = float(np.clip(pos[idx, 1], half_h[idx], ch - half_h[idx]))
        if not has_overlap(idx, x0, y0):
            pos[idx, 0], pos[idx, 1] = x0, y0
            placed[idx] = True
            continue
        step = max(sx[idx], sy[idx]) * 0.20
        best_x, best_y, best_d = x0, y0, float("inf")
        found_any = False
        for r in range(1, 250):
            ring_found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = float(np.clip(x0 + dxm * step, half_w[idx], cw - half_w[idx]))
                    cy = float(np.clip(y0 + dym * step, half_h[idx], ch - half_h[idx]))
                    if has_overlap(idx, cx, cy):
                        continue
                    d = (cx - x0) ** 2 + (cy - y0) ** 2
                    if d < best_d:
                        best_d = d
                        best_x, best_y = cx, cy
                        ring_found = True
                        found_any = True
            if found_any and ring_found:
                break
        pos[idx, 0], pos[idx, 1] = best_x, best_y
        placed[idx] = True
    return pos


# ─────────────────────── ePlace electrostatic density ──────────────── #

def _compute_density_map(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    cw: float,
    ch: float,
    n_bins: int,
) -> torch.Tensor:
    """
    Compute density (charge) map: bin_area sum across all cells, with
    exact rectangle/bin overlap. Differentiable via clamp(min=0).
    Returns [n_bins, n_bins] tensor (y, x order).
    """
    device = pos.device
    bin_w = cw / n_bins
    bin_h = ch / n_bins
    bin_x_lo = torch.arange(n_bins, device=device).float() * bin_w
    bin_x_hi = bin_x_lo + bin_w
    bin_y_lo = torch.arange(n_bins, device=device).float() * bin_h
    bin_y_hi = bin_y_lo + bin_h

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    mx_lo = (pos[:, 0] - half_w).unsqueeze(1)
    mx_hi = (pos[:, 0] + half_w).unsqueeze(1)
    my_lo = (pos[:, 1] - half_h).unsqueeze(1)
    my_hi = (pos[:, 1] + half_h).unsqueeze(1)

    ox = (
        torch.minimum(mx_hi, bin_x_hi.unsqueeze(0))
        - torch.maximum(mx_lo, bin_x_lo.unsqueeze(0))
    ).clamp(min=0)
    oy = (
        torch.minimum(my_hi, bin_y_hi.unsqueeze(0))
        - torch.maximum(my_lo, bin_y_lo.unsqueeze(0))
    ).clamp(min=0)

    return (oy.unsqueeze(2) * ox.unsqueeze(1)).sum(dim=0)


def _eplace_potential_energy(
    rho: torch.Tensor,
    cw: float,
    ch: float,
) -> torch.Tensor:
    """
    Compute the electrostatic potential energy of a charge distribution
    via 2D FFT (ePlace formulation).

    Solve −∇²φ = ρ (with periodic boundaries through real FFT) for the
    potential φ. Energy is (1/2) ⟨ρ, φ⟩.

    Returns scalar tensor (energy).
    """
    n_y, n_x = rho.shape
    bin_w = cw / n_x
    bin_h = ch / n_y

    # Subtract mean so the system is charge-neutral (drops the (0,0) DC mode)
    rho_centered = rho - rho.mean()

    # Real 2D FFT
    rho_hat = torch.fft.rfft2(rho_centered)

    # Wavenumbers — using the convention that index (u,v) corresponds to
    # frequency (2πu/W, 2πv/H) where W=cw, H=ch.
    fy = torch.fft.fftfreq(n_y, d=bin_h, device=rho.device)  # cycles/μm
    fx = torch.fft.rfftfreq(n_x, d=bin_w, device=rho.device)
    ky = (2 * math.pi * fy).unsqueeze(1)  # [n_y, 1]
    kx = (2 * math.pi * fx).unsqueeze(0)  # [1, n_x_r]
    k2 = kx * kx + ky * ky                # [n_y, n_x_r]
    k2[0, 0] = 1.0  # avoid div by zero; (0,0) mode contributes 0 anyway

    # Solve Poisson in frequency: φ̂ = ρ̂ / k²
    phi_hat = rho_hat / k2
    phi_hat[0, 0] = 0.0  # gauge

    # Inverse FFT to get potential
    phi = torch.fft.irfft2(phi_hat, s=(n_y, n_x))

    # Energy = (1/2) Σ ρ φ * bin_area (Riemann sum)
    energy = 0.5 * (rho_centered * phi).sum() * bin_w * bin_h
    return energy


# ─────────────────────── smooth WA wirelength ─────────────────────── #
def _wa_hpwl_dim(
    x: torch.Tensor, net_ids: torch.Tensor, n_nets: int, gamma: float
) -> torch.Tensor:
    device = x.device
    big = 1e18
    true_max = torch.full((n_nets,), -big, device=device)
    true_max.scatter_reduce_(0, net_ids, x / gamma, reduce="amax", include_self=True)
    true_min = torch.full((n_nets,), big, device=device)
    true_min.scatter_reduce_(0, net_ids, x / gamma, reduce="amin", include_self=True)
    tmax = true_max.detach()
    tmin = true_min.detach()

    ep = torch.exp(x / gamma - tmax[net_ids])
    sum_ep = torch.zeros(n_nets, device=device).scatter_add(0, net_ids, ep)
    sum_xep = torch.zeros(n_nets, device=device).scatter_add(0, net_ids, x * ep)
    wa_max = sum_xep / (sum_ep + 1e-12)

    en = torch.exp(-x / gamma + tmin[net_ids])
    sum_en = torch.zeros(n_nets, device=device).scatter_add(0, net_ids, en)
    sum_xen = torch.zeros(n_nets, device=device).scatter_add(0, net_ids, x * en)
    wa_min = sum_xen / (sum_en + 1e-12)
    return wa_max - wa_min


# ─────────────────────── Pin/net data assembly ─────────────────────── #
def _build_pin_data(benchmark: Benchmark, device):
    n_macros = benchmark.num_macros
    n_ports = benchmark.port_positions.shape[0]
    n_owners = n_macros + n_ports
    max_pins = max((p.shape[0] for p in benchmark.macro_pin_offsets), default=1)
    max_pins = max(max_pins, 1)
    owner_pin_offsets = torch.zeros(n_owners, max_pins, 2, device=device)
    for i, offsets in enumerate(benchmark.macro_pin_offsets):
        if offsets.shape[0] > 0:
            owner_pin_offsets[i, : offsets.shape[0]] = offsets.to(device)

    net_owner_list, net_pinidx_list, net_id_list = [], [], []
    for nid, pins in enumerate(benchmark.net_pin_nodes):
        for row in pins.tolist():
            net_owner_list.append(int(row[0]))
            net_pinidx_list.append(int(row[1]))
            net_id_list.append(nid)
    if len(net_owner_list) == 0:
        return None
    net_owners = torch.tensor(net_owner_list, dtype=torch.long, device=device)
    net_pinidx = torch.tensor(net_pinidx_list, dtype=torch.long, device=device)
    net_ids = torch.tensor(net_id_list, dtype=torch.long, device=device)
    n_nets = len(benchmark.net_pin_nodes)
    pin_offset_xy = owner_pin_offsets[net_owners, net_pinidx]
    return {
        "n_macros": n_macros,
        "n_ports": n_ports,
        "n_owners": n_owners,
        "n_nets": n_nets,
        "net_owners": net_owners,
        "net_ids": net_ids,
        "pin_offset_xy": pin_offset_xy,
    }


# ─────────────────────── Global placement (ePlace) ───────────────── #
def _eplace_global(
    benchmark: Benchmark,
    init_pos: torch.Tensor,
    device: torch.device,
    n_iters: int = 800,
    lr_frac: float = 0.005,
    gamma_init_frac: float = 0.05,
    gamma_final_frac: float = 0.005,
    lam_init: float = 1e-2,
    lam_step_up: float = 1.05,
    lam_step_down: float = 0.95,
    n_bins: int = 64,
    use_nesterov: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """
    ePlace-style global placement: optimize smooth WA HPWL + electrostatic
    density energy with adaptive λ.
    """
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_macros = benchmark.num_macros
    sizes = benchmark.macro_sizes.to(device).float()
    movable_mask = benchmark.get_movable_mask().to(device)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    fixed_pos = benchmark.macro_positions.to(device).float().clone()

    pin_data = _build_pin_data(benchmark, device)
    if pin_data is None:
        return init_pos.clone()
    n_owners = pin_data["n_owners"]
    n_nets = pin_data["n_nets"]
    n_ports = pin_data["n_ports"]
    net_owners = pin_data["net_owners"]
    net_ids = pin_data["net_ids"]
    pin_offset_xy = pin_data["pin_offset_xy"]

    port_pos = benchmark.port_positions.to(device).float()

    pos = init_pos.clone().to(device).float()
    pos[:, 0].clamp_(min=half_w, max=cw - half_w)
    pos[:, 1].clamp_(min=half_h, max=ch - half_h)
    pos.requires_grad_(True)

    canvas_size = max(cw, ch)
    lr = lr_frac * canvas_size
    if use_nesterov:
        opt = torch.optim.SGD([pos], lr=lr, momentum=0.9, nesterov=True)
    else:
        opt = torch.optim.Adam([pos], lr=lr)

    lam = lam_init
    prev_wl = None
    for it in range(n_iters):
        opt.zero_grad()
        t = it / max(n_iters - 1, 1)
        gamma = gamma_init_frac * (gamma_final_frac / gamma_init_frac) ** t
        gamma_x = gamma * cw
        gamma_y = gamma * ch

        owner_pos = torch.cat([pos, port_pos], dim=0) if n_ports > 0 else pos
        pin_xy = owner_pos[net_owners] + pin_offset_xy
        wl_x = _wa_hpwl_dim(pin_xy[:, 0], net_ids, n_nets, gamma_x)
        wl_y = _wa_hpwl_dim(pin_xy[:, 1], net_ids, n_nets, gamma_y)
        wl_loss = (wl_x.sum() + wl_y.sum())  # in microns

        # Electrostatic density energy
        rho = _compute_density_map(pos, sizes, cw, ch, n_bins)
        density_energy = _eplace_potential_energy(rho, cw, ch)

        # Total cost
        loss = wl_loss + lam * density_energy

        loss.backward()
        with torch.no_grad():
            pos.grad[~movable_mask] = 0.0
        opt.step()

        with torch.no_grad():
            pos.data[~movable_mask] = fixed_pos[~movable_mask]
            pos.data[:, 0].clamp_(min=half_w, max=cw - half_w)
            pos.data[:, 1].clamp_(min=half_h, max=ch - half_h)

        # ePlace-style lambda update: increase λ if HPWL not decreasing
        if prev_wl is not None:
            ratio = wl_loss.item() / max(prev_wl, 1e-9)
            if ratio > 0.999:
                lam = lam * lam_step_up
            else:
                lam = lam * lam_step_down
            lam = max(min(lam, 1e6), 1e-6)
        prev_wl = wl_loss.item()

        if verbose and it % 100 == 0:
            print(
                f"    [eplace] it={it} γ={gamma:.4f} λ={lam:.3e} "
                f"WL={wl_loss.item():.2f} ρ-energy={density_energy.item():.4f}"
            )

    return pos.detach()


# ─────────────────────── v3-style soft Jacobi ─────────────────────── #
def _soft_jacobi_update(
    pos_np: np.ndarray, benchmark: Benchmark, n_iters: int = 3, damping: float = 0.5
) -> np.ndarray:
    n_macros = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    n_soft = n_macros - n_hard
    if n_soft == 0:
        return pos_np

    n_ports = benchmark.port_positions.shape[0]
    n_owners = n_macros + n_ports
    max_pins = max((p.shape[0] for p in benchmark.macro_pin_offsets), default=1)
    max_pins = max(max_pins, 1)
    owner_pin_offsets = np.zeros((n_owners, max_pins, 2), dtype=np.float64)
    for i, offsets in enumerate(benchmark.macro_pin_offsets):
        if offsets.shape[0] > 0:
            owner_pin_offsets[i, : offsets.shape[0]] = offsets.numpy()

    net_owner_list, net_pinidx_list, net_id_list = [], [], []
    for nid, pins in enumerate(benchmark.net_pin_nodes):
        for row in pins.tolist():
            net_owner_list.append(int(row[0]))
            net_pinidx_list.append(int(row[1]))
            net_id_list.append(nid)
    if len(net_owner_list) == 0:
        return pos_np
    net_owners = np.array(net_owner_list, dtype=np.int64)
    net_pinidx = np.array(net_pinidx_list, dtype=np.int64)
    net_ids = np.array(net_id_list, dtype=np.int64)
    n_nets = len(benchmark.net_pin_nodes)
    pin_offset_xy = owner_pin_offsets[net_owners, net_pinidx]
    port_pos_np = benchmark.port_positions.numpy().astype(np.float64)

    soft_owner_mask = (net_owners >= n_hard) & (net_owners < n_macros)
    soft_pin_indices = np.nonzero(soft_owner_mask)[0]
    soft_macro_idx = net_owners[soft_pin_indices]
    soft_pin_net_ids = net_ids[soft_pin_indices]

    pos_np = pos_np.copy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    soft_sizes = benchmark.macro_sizes[n_hard:].numpy().astype(np.float64)
    soft_half_w = soft_sizes[:, 0] / 2
    soft_half_h = soft_sizes[:, 1] / 2

    for it in range(n_iters):
        owner_pos = np.zeros((n_owners, 2), dtype=np.float64)
        owner_pos[:n_macros] = pos_np
        if n_ports > 0:
            owner_pos[n_macros:] = port_pos_np
        pin_xy = owner_pos[net_owners] + pin_offset_xy

        net_sum_x = np.zeros(n_nets, dtype=np.float64)
        net_sum_y = np.zeros(n_nets, dtype=np.float64)
        net_count = np.zeros(n_nets, dtype=np.int64)
        np.add.at(net_sum_x, net_ids, pin_xy[:, 0])
        np.add.at(net_sum_y, net_ids, pin_xy[:, 1])
        np.add.at(net_count, net_ids, 1)

        soft_pin_x = pin_xy[soft_pin_indices, 0]
        soft_pin_y = pin_xy[soft_pin_indices, 1]
        soft_net_sum_x = net_sum_x[soft_pin_net_ids]
        soft_net_sum_y = net_sum_y[soft_pin_net_ids]
        soft_net_count = net_count[soft_pin_net_ids]

        contrib_x = soft_net_sum_x - soft_pin_x
        contrib_y = soft_net_sum_y - soft_pin_y
        contrib_n = soft_net_count - 1

        soft_local_idx = soft_macro_idx - n_hard
        sum_x = np.zeros(n_soft, dtype=np.float64)
        sum_y = np.zeros(n_soft, dtype=np.float64)
        sum_n = np.zeros(n_soft, dtype=np.int64)
        np.add.at(sum_x, soft_local_idx, contrib_x)
        np.add.at(sum_y, soft_local_idx, contrib_y)
        np.add.at(sum_n, soft_local_idx, contrib_n)

        valid = sum_n > 0
        new_x = pos_np[n_hard:, 0].copy()
        new_y = pos_np[n_hard:, 1].copy()
        new_x[valid] = sum_x[valid] / sum_n[valid]
        new_y[valid] = sum_y[valid] / sum_n[valid]

        pos_np[n_hard:, 0] = (1 - damping) * pos_np[n_hard:, 0] + damping * new_x
        pos_np[n_hard:, 1] = (1 - damping) * pos_np[n_hard:, 1] + damping * new_y
        pos_np[n_hard:, 0] = np.clip(pos_np[n_hard:, 0], soft_half_w, cw - soft_half_w)
        pos_np[n_hard:, 1] = np.clip(pos_np[n_hard:, 1], soft_half_h, ch - soft_half_h)
    return pos_np


# ─────────────────────── V5 placer ─────────────────────────────────── #
class V5Placer:
    def __init__(
        self,
        seed: int = 42,
        device: torch.device | None = None,
        gp_iters: int = 800,
        gp_lr_frac: float = 0.003,
        gamma_init_frac: float = 0.05,
        gamma_final_frac: float = 0.005,
        lam_init: float = 1e-2,
        n_bins: int = 64,
        use_nesterov: bool = True,
        do_soft_jacobi: bool = True,
        soft_iters: int = 3,
        soft_damping: float = 0.5,
        verbose: bool = False,
    ):
        self.seed = seed
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gp_iters = gp_iters
        self.gp_lr_frac = gp_lr_frac
        self.gamma_init_frac = gamma_init_frac
        self.gamma_final_frac = gamma_final_frac
        self.lam_init = lam_init
        self.n_bins = n_bins
        self.use_nesterov = use_nesterov
        self.do_soft_jacobi = do_soft_jacobi
        self.soft_iters = soft_iters
        self.soft_damping = soft_damping
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        device = self.device
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
        movable = benchmark.get_movable_mask().numpy()
        movable_hard = movable[:n_hard]
        fixed_pos_np = benchmark.macro_positions.numpy().astype(np.float64)

        from macro_place.objective import compute_proxy_cost

        plc = _load_plc(benchmark.name)

        # Stage A: warm start from initial.plc
        init_pos = benchmark.macro_positions.clone().to(device).float()

        # Stage B: ePlace global placement
        if self.gp_iters > 0:
            t0 = time.time()
            gp_pos = _eplace_global(
                benchmark,
                init_pos,
                device=device,
                n_iters=self.gp_iters,
                lr_frac=self.gp_lr_frac,
                gamma_init_frac=self.gamma_init_frac,
                gamma_final_frac=self.gamma_final_frac,
                lam_init=self.lam_init,
                n_bins=self.n_bins,
                use_nesterov=self.use_nesterov,
                verbose=self.verbose,
            )
            if self.verbose:
                print(f"  [V5] GP done in {time.time() - t0:.1f}s")
        else:
            gp_pos = init_pos

        # Stage C: legalize from GP positions; also try from initial.plc as fallback
        candidates = []
        for label, base in [("gp", gp_pos.cpu().numpy().astype(np.float64)),
                            ("init", fixed_pos_np)]:
            area = sizes_np[:n_hard, 0] * sizes_np[:n_hard, 1]
            order = list(np.argsort(-area))  # area descending
            legal = _legalize_with_order(
                base[:n_hard].copy(), movable_hard, sizes_np, cw, ch, n_hard,
                fixed_pos_np[:n_hard], order,
            )
            pos_np = base.copy()
            pos_np[:n_hard] = legal
            full = torch.from_numpy(pos_np).float()
            if plc is not None:
                cd = compute_proxy_cost(full, benchmark, plc)
                candidates.append((label, cd["proxy_cost"], cd["overlap_count"], full))
            else:
                candidates.append((label, float("inf"), 0, full))

            if self.do_soft_jacobi:
                pos_refined = _soft_jacobi_update(
                    pos_np, benchmark, n_iters=self.soft_iters,
                    damping=self.soft_damping,
                )
                full_r = torch.from_numpy(pos_refined).float()
                if plc is not None:
                    cd = compute_proxy_cost(full_r, benchmark, plc)
                    candidates.append((f"{label}+soft", cd["proxy_cost"], cd["overlap_count"], full_r))

        # Pick best
        best = None
        for label, cost, ovl, p in candidates:
            if ovl > 0:
                continue
            if self.verbose:
                print(f"    candidate {label:>15}: proxy={cost:.4f}")
            if best is None or cost < best[1]:
                best = (label, cost, p)

        if best is None:
            # No legal candidate? fallback to initial
            return torch.from_numpy(fixed_pos_np.copy()).float()

        if self.verbose:
            print(f"  ▶ V5 best: {best[0]} → proxy={best[1]:.4f}")
        return best[2]
