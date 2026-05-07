"""
v12: Iterative DREAMPlace → Reweight → QA (legalization) loop.

Builds on v11's ePlace global placement + multi-start legalization
with an outer reweighting loop inspired by Lagrangian / game-theoretic
"raise the weight of the worst component, re-solve":

  GLOBAL ──► REWEIGHT ──► QA
  (DREAMPlace) (per-net)   (legalize+check)
       ↑                    │
       └──────loop──────────┘

Per-iteration reweighting:
  1. After each ePlace + legalize, compute per-net HPWL.
  2. Identify "long" nets — those whose HPWL is in the top K% of
     (HPWL / sqrt(net_pin_count)).
  3. Boost their net weights by a multiplicative factor (e.g. 1.5).
  4. Re-run ePlace global placement starting from current pos with
     new net_weights.
  5. Legalize, compute actual proxy. Keep best.

QA step (legalization) is unchanged from v11 — multi-start spiral
search with 7 orderings + soft-Jacobi update.

The intuition: when we identify nets that are "stretched" (high HPWL),
boosting their weight tells the next ePlace solve to prioritize
shrinking them, which routes around the local optimum we're stuck in.

Otherwise identical to v11 (Nesterov+BB, FFT density, filler nodes,
γ + density-weight schedules).

OLD v11 NOTES below.

v11: DREAMPlace-faithful pure-PyTorch reimplementation.

Closely follows the algorithms in external/DREAMPlace/dreamplace/:
  - PlaceObj.obj_fn:    wirelength + density_weight × electrostatic_density
  - WeightedAverageWirelength: γ-smoothed weighted-average HPWL per net
  - electric_potential.ElectricPotentialFunction: ePlace electrostatic
    density via FFT Poisson solve (here using rfft2; DREAMPlace uses
    DCT2/IDCT2/IDXST_IDCT for Neumann BCs; the energy is still
    monotonic in real density, so the gradient direction matches).
  - NesterovAcceleratedGradientOptimizer: Nesterov + Barzilai-Borwein
    adaptive step size.
  - PlaceObj.update_gamma:  γ = base_γ × 10^((overflow - 0.1)·20/9 - 1)
  - PlaceObj.update_density_weight (HPWL mode):
        if Δ_hpwl < 0: μ = 1.05 × max(0.9999^iter, 0.98)
        else:           μ = 1.05 × clamp(1.05^(-Δhpwl/ref_hpwl), 0.95, 1.05)
        density_weight *= μ
  - Filler nodes: fictitious cells filling empty area so density
    spreads uniformly to target_density. Critical — without filler,
    real cells over-cluster.

Pipeline:
  1. ePlace global placement (Nesterov + electrostatic, with filler).
  2. Greedy minimum-displacement legalization of hard macros.
  3. Soft-macro Jacobi update (HPWL-optimal centroid, from v3).
  4. Compare ePlace-warmed vs initial.plc-warmed pipelines; keep best.
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


# ───────────────────────── PLC loader ───────────────────────── #
def _load_plc(name: str):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile"}
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


# ───────────────────── pin/net data builder ──────────────────── #
def _build_pin_data(benchmark: Benchmark, device: torch.device):
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
    if not net_owner_list:
        return None
    net_owners = torch.tensor(net_owner_list, dtype=torch.long, device=device)
    net_pinidx = torch.tensor(net_pinidx_list, dtype=torch.long, device=device)
    net_ids = torch.tensor(net_id_list, dtype=torch.long, device=device)
    n_nets = len(benchmark.net_pin_nodes)
    pin_offset_xy = owner_pin_offsets[net_owners, net_pinidx]
    return dict(
        n_macros=n_macros, n_ports=n_ports, n_owners=n_owners, n_nets=n_nets,
        net_owners=net_owners, net_ids=net_ids, pin_offset_xy=pin_offset_xy,
    )


# ─────────────────── smooth WA wirelength (same as DP) ─────────── #
def _wa_hpwl_dim(x: torch.Tensor, net_ids: torch.Tensor, n_nets: int, gamma: float) -> torch.Tensor:
    """Per-net weighted-average wirelength along one axis, smoothed by γ.
    Returns [n_nets] tensor of per-net (WA_max - WA_min) values (NOT yet
    multiplied by net weights — caller does that)."""
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


# ─────────────────── density map (rectangle/bin overlap) ────────── #
def _density_map(pos: torch.Tensor, sizes: torch.Tensor, cw: float, ch: float, n_bins: int) -> torch.Tensor:
    """
    Differentiable density map = exact macro/bin overlap area.
    Returns [n_bins, n_bins] tensor (y-row, x-col).
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
    ox = (torch.minimum(mx_hi, bin_x_hi.unsqueeze(0))
          - torch.maximum(mx_lo, bin_x_lo.unsqueeze(0))).clamp(min=0)
    oy = (torch.minimum(my_hi, bin_y_hi.unsqueeze(0))
          - torch.maximum(my_lo, bin_y_lo.unsqueeze(0))).clamp(min=0)
    return (oy.unsqueeze(2) * ox.unsqueeze(1)).sum(dim=0)


# ─────────────── ePlace electrostatic energy via FFT ──────────── #
def _eplace_energy(rho: torch.Tensor, cw: float, ch: float) -> torch.Tensor:
    """
    Solve −∇²φ = ρ on a periodic grid via 2D FFT (Poisson in frequency
    domain), then compute electrostatic energy E = (1/2) ∫ ρ φ dA.
    """
    n_y, n_x = rho.shape
    bin_w = cw / n_x
    bin_h = ch / n_y
    rho_centered = rho - rho.mean()  # charge neutrality
    rho_hat = torch.fft.rfft2(rho_centered)
    fy = torch.fft.fftfreq(n_y, d=bin_h, device=rho.device)
    fx = torch.fft.rfftfreq(n_x, d=bin_w, device=rho.device)
    ky = (2 * math.pi * fy).unsqueeze(1)
    kx = (2 * math.pi * fx).unsqueeze(0)
    k2 = kx * kx + ky * ky
    k2_safe = k2.clone()
    k2_safe[0, 0] = 1.0
    phi_hat = rho_hat / k2_safe
    phi_hat[0, 0] = 0.0
    phi = torch.fft.irfft2(phi_hat, s=(n_y, n_x))
    energy = 0.5 * (rho_centered * phi).sum() * bin_w * bin_h
    return energy


# ───────────── Nesterov accelerated gradient + BB step ────────── #
class NesterovBB:
    """
    Nesterov + Barzilai-Borwein step size, ported from
    external/DREAMPlace/dreamplace/NesterovAcceleratedGradientOptimizer.py
    """
    def __init__(self, init_pos: torch.Tensor, obj_grad_fn, constraint_fn, lr_init: float):
        self.obj_grad_fn = obj_grad_fn
        self.constraint_fn = constraint_fn
        # v_k carries gradient
        self.v_k = init_pos.detach().clone().requires_grad_(True)
        self.u_k = self.v_k.detach().clone()
        self.a_k = torch.tensor(1.0, device=init_pos.device)

        obj_k, g_k = obj_grad_fn(self.v_k)
        self.g_k = g_k.detach().clone()
        self.obj_k = obj_k.detach().clone()

        v_k_1 = (self.v_k.detach() - lr_init * self.g_k).requires_grad_(True)
        obj_k_1, g_k_1 = obj_grad_fn(v_k_1)
        self.v_k_1 = v_k_1.detach()
        self.g_k_1 = g_k_1.detach().clone()
        denom = (self.g_k - self.g_k_1).norm(p=2).clamp(min=1e-12)
        self.alpha_k = ((self.v_k.detach() - self.v_k_1).norm(p=2) / denom).abs()

    def step(self):
        s_k = self.v_k.detach() - self.v_k_1
        y_k = self.g_k - self.g_k_1
        sk_dot_yk = (s_k * y_k).sum().clamp(min=1e-20)
        bb_short = (sk_dot_yk / (y_k * y_k).sum().clamp(min=1e-20)).abs()
        lip_step = (s_k.norm(p=2) / y_k.norm(p=2).clamp(min=1e-12)).abs()
        if bb_short.item() > 0:
            step_size = bb_short
        else:
            step_size = torch.minimum(lip_step, self.alpha_k)

        a_kp1 = (1 + (4 * self.a_k.pow(2) + 1).sqrt()) / 2
        coef = (self.a_k - 1) / a_kp1

        u_kp1 = self.v_k.detach() - step_size * self.g_k
        v_kp1 = u_kp1 + coef * (u_kp1 - self.u_k)
        v_kp1 = self.constraint_fn(v_kp1)
        v_kp1 = v_kp1.detach().clone().requires_grad_(True)
        obj_kp1, g_kp1 = self.obj_grad_fn(v_kp1)

        self.v_k_1 = self.v_k.detach().clone()
        self.g_k_1 = self.g_k.clone()
        self.alpha_k = step_size
        self.u_k = u_kp1.detach().clone()
        self.v_k = v_kp1
        self.g_k = g_kp1.detach().clone()
        self.obj_k = obj_kp1.detach().clone()
        self.a_k = a_kp1
        return obj_kp1.item()


# ───────────────────── ePlace global placement ──────────────── #
def _eplace_global(
    benchmark: Benchmark,
    device: torch.device,
    n_iters: int = 600,
    n_bins: int = 64,
    target_density: float = 0.85,
    base_gamma_factor: float = 4.0,
    init_lr_frac: float = 1e-3,
    add_filler: bool = True,
    verbose: bool = False,
    net_weights: "torch.Tensor | None" = None,
    init_pos_real: "torch.Tensor | None" = None,
    macro_halo_frac: float = 0.0,  # NEW: halo as fraction of canvas (each side)
):
    """DREAMPlace-style global placement. Returns (final_real_pos, hist_dict).

    With macro_halo_frac > 0, hard macros are inflated by 2 * halo on each
    dimension during density computation only (AutoDMP trick). Pin offsets
    and final macro positions are unchanged. The halo reserves space around
    macros so legalization barely needs to move them.
    """
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    n_real = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    sizes_real = benchmark.macro_sizes.to(device).float()
    movable_real = benchmark.get_movable_mask().to(device)
    fixed_pos = benchmark.macro_positions.to(device).float().clone()

    # Macro halos: inflate hard macros only (soft macros are already small)
    halo_size = macro_halo_frac * max(cw, ch)
    if halo_size > 0:
        sizes_inflated = sizes_real.clone()
        sizes_inflated[:n_hard, 0] += 2 * halo_size
        sizes_inflated[:n_hard, 1] += 2 * halo_size
    else:
        sizes_inflated = sizes_real

    # Filler cells (use REAL areas to compute filler need; halo doesn't add real area)
    canvas_area = cw * ch
    real_area = float((sizes_real[:, 0] * sizes_real[:, 1]).sum().item())
    target_total_area = canvas_area * target_density
    filler_area = max(target_total_area - real_area, 0.0)
    filler_size = math.sqrt(filler_area / max(n_real, 1)) if filler_area > 0 else 0.0
    n_filler = int(filler_area / max(filler_size * filler_size, 1e-12)) if add_filler and filler_size > 0 else 0
    if verbose:
        print(f"  [eplace] cw×ch={cw:.1f}×{ch:.1f}  real_area={real_area:.1f} "
              f"target={target_total_area:.1f}  fillers={n_filler}")

    if n_filler > 0:
        filler_sizes = torch.full((n_filler, 2), filler_size, device=device)
        all_sizes = torch.cat([sizes_inflated, filler_sizes], dim=0)
    else:
        all_sizes = sizes_inflated
    n_total = n_real + n_filler

    # Pin data
    pin_data = _build_pin_data(benchmark, device)
    if pin_data is None:
        return fixed_pos[:n_real].cpu(), {}
    n_owners = pin_data["n_owners"]; n_nets = pin_data["n_nets"]; n_ports = pin_data["n_ports"]
    net_owners = pin_data["net_owners"]; net_ids = pin_data["net_ids"]
    pin_offset_xy = pin_data["pin_offset_xy"]
    port_pos = benchmark.port_positions.to(device).float()

    # Initial positions: warm-start if provided
    pos = torch.zeros(n_total, 2, device=device)
    if init_pos_real is not None:
        pos[:n_real] = init_pos_real.to(device).float()
    else:
        pos[:n_real] = fixed_pos
    if n_filler > 0:
        ng = max(1, int(math.ceil(math.sqrt(n_filler))))
        for k in range(n_filler):
            ix = k % ng; iy = k // ng
            pos[n_real + k, 0] = (ix + 0.5) * cw / ng
            pos[n_real + k, 1] = (iy + 0.5) * ch / ng

    half_w_all = all_sizes[:, 0] / 2
    half_h_all = all_sizes[:, 1] / 2

    movable_all = torch.zeros(n_total, dtype=torch.bool, device=device)
    movable_all[:n_real] = movable_real
    if n_filler > 0:
        movable_all[n_real:] = True
    fixed_pos_full = pos.clone()  # for restoring fixed entries

    def constraint_fn(p: torch.Tensor) -> torch.Tensor:
        p_clamped = p.clone()
        p_clamped[:, 0] = p_clamped[:, 0].clamp(min=half_w_all, max=cw - half_w_all)
        p_clamped[:, 1] = p_clamped[:, 1].clamp(min=half_h_all, max=ch - half_h_all)
        if (~movable_all).any():
            p_clamped[~movable_all] = fixed_pos_full[~movable_all]
        return p_clamped

    bin_size = (cw / n_bins) + (ch / n_bins)
    base_gamma = base_gamma_factor * bin_size
    bin_w = cw / n_bins
    bin_h = ch / n_bins
    target_cap = target_density * bin_w * bin_h

    # Per-net weights tensor (default 1.0 each)
    if net_weights is None:
        nw_tensor = torch.ones(n_nets, device=device)
    else:
        nw_tensor = net_weights.to(device).float()
        assert nw_tensor.shape[0] == n_nets, f"net_weights len {nw_tensor.shape[0]} != n_nets {n_nets}"

    state = dict(density_weight=1.0e-5, gamma=10 * base_gamma, prev_hpwl=None,
                 hpwl_history=[], overflow_history=[], dw_history=[])

    def compute_overflow_val(rho: torch.Tensor) -> float:
        excess = (rho - target_cap).clamp(min=0).sum().item()
        denom = real_area + (n_filler * filler_size * filler_size if n_filler > 0 else 0.0)
        return excess / max(denom, 1e-12)

    def update_gamma(overflow: float):
        coef = 10.0 ** ((overflow - 0.1) * 20.0 / 9.0 - 1.0)
        return base_gamma * coef

    def update_density_weight(cur_hpwl: float, prev_hpwl: float, iteration: int):
        if prev_hpwl is None:
            return state["density_weight"]
        ref_hpwl = max(abs(prev_hpwl) * 0.1, 1e-3)
        delta = cur_hpwl - prev_hpwl
        if delta < 0:
            mu = 1.05 * max(0.9999 ** float(iteration), 0.98)
        else:
            inner = 1.05 ** (-delta / ref_hpwl)
            mu = 1.05 * float(np.clip(inner, 0.95, 1.05))
        return state["density_weight"] * mu

    def obj_grad_fn(p: torch.Tensor):
        if p.grad is not None:
            p.grad.zero_()
        owner_pos = torch.cat([p[:n_real], port_pos], dim=0) if n_ports > 0 else p[:n_real]
        pin_xy = owner_pos[net_owners] + pin_offset_xy
        wl_x = _wa_hpwl_dim(pin_xy[:, 0], net_ids, n_nets, state["gamma"])
        wl_y = _wa_hpwl_dim(pin_xy[:, 1], net_ids, n_nets, state["gamma"])
        # Per-net WL multiplied by net weight
        wl_total = (nw_tensor * wl_x).sum() + (nw_tensor * wl_y).sum()
        rho = _density_map(p, all_sizes, cw, ch, n_bins)
        density_energy = _eplace_energy(rho, cw, ch)
        loss = wl_total + state["density_weight"] * density_energy
        if loss.requires_grad:
            loss.backward()
        with torch.no_grad():
            if (~movable_all).any() and p.grad is not None:
                p.grad[~movable_all] = 0.0
        return loss, p.grad if p.grad is not None else torch.zeros_like(p)

    pos.requires_grad_(True)
    canvas_size = max(cw, ch)
    init_lr = init_lr_frac * canvas_size
    optimizer = NesterovBB(pos, obj_grad_fn, constraint_fn, init_lr)

    # Estimate initial density_weight: balance gradient magnitudes
    with torch.no_grad():
        owner_pos0 = torch.cat([optimizer.v_k.detach()[:n_real], port_pos], dim=0) if n_ports > 0 else optimizer.v_k.detach()[:n_real]
        pin_xy0 = owner_pos0[net_owners] + pin_offset_xy
        wl0 = ((nw_tensor * _wa_hpwl_dim(pin_xy0[:, 0], net_ids, n_nets, state["gamma"])).sum().item()
               + (nw_tensor * _wa_hpwl_dim(pin_xy0[:, 1], net_ids, n_nets, state["gamma"])).sum().item())
        rho0 = _density_map(optimizer.v_k.detach(), all_sizes, cw, ch, n_bins)
        de0 = abs(_eplace_energy(rho0, cw, ch).item())
        if de0 > 1e-12:
            state["density_weight"] = 8e-5 * abs(wl0) / de0
        if verbose:
            print(f"  [eplace] init wl={wl0:.2f}  density_energy={de0:.2f}  λ0={state['density_weight']:.3e}")

    best_overflow = float("inf")
    best_pos = optimizer.v_k.detach().clone()
    diverged_count = 0

    for it in range(n_iters):
        with torch.no_grad():
            rho_cur = _density_map(optimizer.v_k.detach(), all_sizes, cw, ch, n_bins)
            overflow = compute_overflow_val(rho_cur)
            state["gamma"] = update_gamma(overflow)
            owner_pos_cur = torch.cat([optimizer.v_k.detach()[:n_real], port_pos], dim=0) if n_ports > 0 else optimizer.v_k.detach()[:n_real]
            pin_xy_cur = owner_pos_cur[net_owners] + pin_offset_xy
            cur_hpwl = ((nw_tensor * _wa_hpwl_dim(pin_xy_cur[:, 0], net_ids, n_nets, state["gamma"])).sum().item()
                        + (nw_tensor * _wa_hpwl_dim(pin_xy_cur[:, 1], net_ids, n_nets, state["gamma"])).sum().item())

        new_dw = update_density_weight(cur_hpwl, state["prev_hpwl"], it)
        state["density_weight"] = float(np.clip(new_dw, 1e-10, 1e3))

        try:
            optimizer.step()
        except Exception as e:
            if verbose:
                print(f"  [eplace] iter {it}: step failed: {e}")
            break

        state["prev_hpwl"] = cur_hpwl
        state["hpwl_history"].append(cur_hpwl)
        state["overflow_history"].append(overflow)
        state["dw_history"].append(state["density_weight"])

        if overflow < best_overflow:
            best_overflow = overflow
            best_pos = optimizer.v_k.detach().clone()

        if verbose and it % 50 == 0:
            print(f"  [eplace] it={it} γ={state['gamma']:.3f} λ={state['density_weight']:.3e} "
                  f"hpwl={cur_hpwl:.2f} ovfl={overflow:.3f}")

        # Convergence
        if overflow < 0.10 and it > 100 and cur_hpwl > state["hpwl_history"][-10]:
            if verbose:
                print(f"  [eplace] converged at iter {it}: overflow={overflow:.3f}")
            break

        # Divergence
        if it > 50 and len(state["hpwl_history"]) > 20:
            if cur_hpwl > 5 * state["hpwl_history"][20]:
                diverged_count += 1
                if diverged_count > 5:
                    if verbose:
                        print(f"  [eplace] iter {it}: HPWL diverged, stopping")
                    break

    final_pos = optimizer.v_k.detach()[:n_real].cpu().clone()
    return final_pos, state


# ─────────────────────── Legalization (v3) ───────────────────── #
def _legalize_with_order(
    pos: np.ndarray, movable: np.ndarray, sizes: np.ndarray,
    cw: float, ch: float, n_hard: int, fixed_pos: np.ndarray, order: List[int],
) -> np.ndarray:
    pos = pos.copy()
    for i in range(n_hard):
        if not movable[i]:
            pos[i] = fixed_pos[i]
    gap = 0.001
    sx = sizes[:n_hard, 0]; sy = sizes[:n_hard, 1]
    sep_x_mat = (sx[:, None] + sx[None, :]) / 2 + gap
    sep_y_mat = (sy[:, None] + sy[None, :]) / 2 + gap
    half_w = sx / 2; half_h = sy / 2
    placed = np.zeros(n_hard, dtype=bool)
    for i in range(n_hard):
        if not movable[i]:
            placed[i] = True

    def has_overlap(idx, x, y):
        if not placed.any():
            return False
        dx = np.abs(x - pos[:n_hard, 0]); dy = np.abs(y - pos[:n_hard, 1])
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
                        best_d = d; best_x, best_y = cx, cy
                        ring_found = True; found_any = True
            if found_any and ring_found:
                break
        pos[idx, 0], pos[idx, 1] = best_x, best_y
        placed[idx] = True
    return pos


def _build_orderings(sizes_np, fixed_pos, n_hard, cw, ch):
    orderings = []
    area = sizes_np[:n_hard, 0] * sizes_np[:n_hard, 1]
    orderings.append(("area_desc", list(np.argsort(-area))))
    orderings.append(("area_asc", list(np.argsort(area))))
    cx, cy = cw / 2, ch / 2
    dc = (fixed_pos[:n_hard, 0] - cx) ** 2 + (fixed_pos[:n_hard, 1] - cy) ** 2
    orderings.append(("center_first", list(np.argsort(dc))))
    orderings.append(("edges_first", list(np.argsort(-dc))))
    orderings.append(("width_desc", list(np.argsort(-sizes_np[:n_hard, 0]))))
    for s in (1, 7):
        rng = np.random.RandomState(s)
        order = list(range(n_hard))
        rng.shuffle(order)
        orderings.append((f"random_{s}", order))
    return orderings


# ─────────────────────── Soft Jacobi (v3) ────────────────────── #
def _soft_jacobi_update(pos_np, benchmark, n_iters=3, damping=0.5):
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
    if not net_owner_list:
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
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    soft_sizes = benchmark.macro_sizes[n_hard:].numpy().astype(np.float64)
    soft_half_w = soft_sizes[:, 0] / 2; soft_half_h = soft_sizes[:, 1] / 2

    for _ in range(n_iters):
        owner_pos = np.zeros((n_owners, 2), dtype=np.float64)
        owner_pos[:n_macros] = pos_np
        if n_ports > 0:
            owner_pos[n_macros:] = port_pos_np
        pin_xy = owner_pos[net_owners] + pin_offset_xy
        net_sum_x = np.zeros(n_nets); net_sum_y = np.zeros(n_nets); net_count = np.zeros(n_nets, dtype=np.int64)
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
        sum_x = np.zeros(n_soft); sum_y = np.zeros(n_soft); sum_n = np.zeros(n_soft, dtype=np.int64)
        np.add.at(sum_x, soft_local_idx, contrib_x)
        np.add.at(sum_y, soft_local_idx, contrib_y)
        np.add.at(sum_n, soft_local_idx, contrib_n)

        valid = sum_n > 0
        new_x = pos_np[n_hard:, 0].copy(); new_y = pos_np[n_hard:, 1].copy()
        new_x[valid] = sum_x[valid] / sum_n[valid]
        new_y[valid] = sum_y[valid] / sum_n[valid]
        pos_np[n_hard:, 0] = (1 - damping) * pos_np[n_hard:, 0] + damping * new_x
        pos_np[n_hard:, 1] = (1 - damping) * pos_np[n_hard:, 1] + damping * new_y
        pos_np[n_hard:, 0] = np.clip(pos_np[n_hard:, 0], soft_half_w, cw - soft_half_w)
        pos_np[n_hard:, 1] = np.clip(pos_np[n_hard:, 1], soft_half_h, ch - soft_half_h)
    return pos_np


# ─────────────────────── Fast incremental CD (from v7) ────── #
class IncrementalProxy:
    """Incremental WL+density surrogate (HPWL per net + density grid)."""

    def __init__(self, benchmark: Benchmark, full_pos: np.ndarray,
                 wl_weight: float = 1.0, den_weight: float = 0.5):
        self.benchmark = benchmark
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.n_macros = benchmark.num_macros
        self.n_hard = benchmark.num_hard_macros
        self.sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        self.wl_weight = wl_weight
        self.den_weight = den_weight

        n_ports = benchmark.port_positions.shape[0]
        self.n_ports = n_ports
        self.n_owners = self.n_macros + n_ports
        self.port_pos = (benchmark.port_positions.numpy().astype(np.float64)
                         if n_ports > 0 else np.zeros((0, 2)))

        max_pins = max((p.shape[0] for p in benchmark.macro_pin_offsets), default=1)
        max_pins = max(max_pins, 1)
        self.owner_pin_offsets = np.zeros((self.n_owners, max_pins, 2), dtype=np.float64)
        for i, offsets in enumerate(benchmark.macro_pin_offsets):
            if offsets.shape[0] > 0:
                self.owner_pin_offsets[i, : offsets.shape[0]] = offsets.numpy()

        n_nets = len(benchmark.net_pin_nodes)
        self.n_nets = n_nets

        net_owner_list, net_pinidx_list, net_id_list = [], [], []
        for nid, pins in enumerate(benchmark.net_pin_nodes):
            for row in pins.tolist():
                net_owner_list.append(int(row[0]))
                net_pinidx_list.append(int(row[1]))
                net_id_list.append(nid)
        self.net_owners = np.array(net_owner_list, dtype=np.int64)
        self.net_pinidx = np.array(net_pinidx_list, dtype=np.int64)
        self.net_ids = np.array(net_id_list, dtype=np.int64)

        self.owner_to_pin_entries = [None] * self.n_owners
        for owner in range(self.n_owners):
            self.owner_to_pin_entries[owner] = np.nonzero(self.net_owners == owner)[0]
        self.pin_offset_xy = self.owner_pin_offsets[self.net_owners, self.net_pinidx]
        self.net_to_pin_entries = [None] * n_nets
        for nid in range(n_nets):
            self.net_to_pin_entries[nid] = np.nonzero(self.net_ids == nid)[0]

        self.pos = full_pos.copy()
        self.pin_xy = self._pin_xy_full()
        self.net_hpwl = np.zeros(n_nets, dtype=np.float64)
        self._recompute_all_hpwl()

        self.gr = benchmark.grid_rows
        self.gc = benchmark.grid_cols
        self.bin_w = self.cw / self.gc
        self.bin_h = self.ch / self.gr
        self.density = np.zeros((self.gr, self.gc), dtype=np.float64)
        self._recompute_density_full()

    def _pin_xy_full(self) -> np.ndarray:
        owner_pos = np.zeros((self.n_owners, 2), dtype=np.float64)
        owner_pos[:self.n_macros] = self.pos
        if self.n_ports > 0:
            owner_pos[self.n_macros:] = self.port_pos
        return owner_pos[self.net_owners] + self.pin_offset_xy

    def _recompute_all_hpwl(self):
        for nid in range(self.n_nets):
            entries = self.net_to_pin_entries[nid]
            if len(entries) <= 1:
                self.net_hpwl[nid] = 0.0
                continue
            xs = self.pin_xy[entries, 0]; ys = self.pin_xy[entries, 1]
            self.net_hpwl[nid] = (xs.max() - xs.min()) + (ys.max() - ys.min())

    def _recompute_density_full(self):
        self.density.fill(0.0)
        for i in range(self.n_macros):
            self._add_density_contribution(i, +1.0)

    def _add_density_contribution(self, macro_idx: int, sign: float):
        sx = self.sizes[macro_idx, 0]; sy = self.sizes[macro_idx, 1]
        x = self.pos[macro_idx, 0]; y = self.pos[macro_idx, 1]
        x_lo = max(x - sx / 2, 0.0); x_hi = min(x + sx / 2, self.cw)
        y_lo = max(y - sy / 2, 0.0); y_hi = min(y + sy / 2, self.ch)
        if x_hi <= x_lo or y_hi <= y_lo:
            return
        col_lo = int(x_lo / self.bin_w); col_hi = int(min((x_hi - 1e-12) / self.bin_w, self.gc - 1))
        row_lo = int(y_lo / self.bin_h); row_hi = int(min((y_hi - 1e-12) / self.bin_h, self.gr - 1))
        col_lo = max(0, min(col_lo, self.gc - 1)); col_hi = max(0, min(col_hi, self.gc - 1))
        row_lo = max(0, min(row_lo, self.gr - 1)); row_hi = max(0, min(row_hi, self.gr - 1))

        for r in range(row_lo, row_hi + 1):
            ry_lo = r * self.bin_h; ry_hi = ry_lo + self.bin_h
            oy = max(0.0, min(y_hi, ry_hi) - max(y_lo, ry_lo))
            if oy <= 0: continue
            for c in range(col_lo, col_hi + 1):
                rx_lo = c * self.bin_w; rx_hi = rx_lo + self.bin_w
                ox = max(0.0, min(x_hi, rx_hi) - max(x_lo, rx_lo))
                if ox <= 0: continue
                self.density[r, c] += sign * ox * oy

    def total_hpwl(self) -> float:
        return float(self.net_hpwl.sum())

    def density_cost(self) -> float:
        flat = self.density.flatten()
        n_top = max(1, int(np.ceil(len(flat) * 0.1)))
        top_vals = np.partition(flat, -n_top)[-n_top:]
        bin_area = self.bin_w * self.bin_h
        return float(top_vals.mean() / bin_area)

    def surrogate_cost(self) -> float:
        wl = self.total_hpwl() / max(self.n_nets * (self.cw + self.ch), 1e-12)
        return self.wl_weight * wl + self.den_weight * self.density_cost()

    def proposed_move_cost(self, macro_idx: int, new_x: float, new_y: float) -> float:
        old_x = self.pos[macro_idx, 0]; old_y = self.pos[macro_idx, 1]
        affected_nets = np.unique(self.net_ids[self.owner_to_pin_entries[macro_idx]])
        old_hpwl_for_nets = {nid: self.net_hpwl[nid] for nid in affected_nets}
        new_hpwl_for_nets = {}
        for entry in self.owner_to_pin_entries[macro_idx]:
            self.pin_xy[entry, 0] = new_x + self.pin_offset_xy[entry, 0]
            self.pin_xy[entry, 1] = new_y + self.pin_offset_xy[entry, 1]
        for nid in affected_nets:
            entries = self.net_to_pin_entries[nid]
            if len(entries) <= 1:
                new_hpwl_for_nets[nid] = 0.0
                continue
            xs = self.pin_xy[entries, 0]; ys = self.pin_xy[entries, 1]
            new_hpwl_for_nets[nid] = (xs.max() - xs.min()) + (ys.max() - ys.min())
        for entry in self.owner_to_pin_entries[macro_idx]:
            self.pin_xy[entry, 0] = old_x + self.pin_offset_xy[entry, 0]
            self.pin_xy[entry, 1] = old_y + self.pin_offset_xy[entry, 1]
        delta_hpwl = sum(new_hpwl_for_nets[nid] - old_hpwl_for_nets[nid]
                         for nid in affected_nets)
        new_total_hpwl = self.total_hpwl() + delta_hpwl

        self._add_density_contribution(macro_idx, -1.0)
        self.pos[macro_idx, 0] = new_x; self.pos[macro_idx, 1] = new_y
        self._add_density_contribution(macro_idx, +1.0)
        new_density_cost = self.density_cost()
        self._add_density_contribution(macro_idx, -1.0)
        self.pos[macro_idx, 0] = old_x; self.pos[macro_idx, 1] = old_y
        self._add_density_contribution(macro_idx, +1.0)

        wl_norm = new_total_hpwl / max(self.n_nets * (self.cw + self.ch), 1e-12)
        return self.wl_weight * wl_norm + self.den_weight * new_density_cost

    def commit_move(self, macro_idx: int, new_x: float, new_y: float):
        old_x = self.pos[macro_idx, 0]; old_y = self.pos[macro_idx, 1]
        self._add_density_contribution(macro_idx, -1.0)
        self.pos[macro_idx, 0] = new_x; self.pos[macro_idx, 1] = new_y
        self._add_density_contribution(macro_idx, +1.0)
        for entry in self.owner_to_pin_entries[macro_idx]:
            self.pin_xy[entry, 0] = new_x + self.pin_offset_xy[entry, 0]
            self.pin_xy[entry, 1] = new_y + self.pin_offset_xy[entry, 1]
        affected_nets = np.unique(self.net_ids[self.owner_to_pin_entries[macro_idx]])
        for nid in affected_nets:
            entries = self.net_to_pin_entries[nid]
            if len(entries) <= 1:
                self.net_hpwl[nid] = 0.0
                continue
            xs = self.pin_xy[entries, 0]; ys = self.pin_xy[entries, 1]
            self.net_hpwl[nid] = (xs.max() - xs.min()) + (ys.max() - ys.min())


def _try_swap(
    inc: "IncrementalProxy", i: int, j: int,
    sep_x: np.ndarray, sep_y: np.ndarray,
    half_w: np.ndarray, half_h: np.ndarray, cw: float, ch: float,
) -> "float | None":
    pos_i_old = (inc.pos[i, 0], inc.pos[i, 1])
    pos_j_old = (inc.pos[j, 0], inc.pos[j, 1])
    n_hard = len(half_w)

    nx_i = float(np.clip(pos_j_old[0], half_w[i], cw - half_w[i]))
    ny_i = float(np.clip(pos_j_old[1], half_h[i], ch - half_h[i]))
    nx_j = float(np.clip(pos_i_old[0], half_w[j], cw - half_w[j]))
    ny_j = float(np.clip(pos_i_old[1], half_h[j], ch - half_h[j]))

    ddx = np.abs(nx_i - inc.pos[:n_hard, 0]); ddy = np.abs(ny_i - inc.pos[:n_hard, 1])
    mask_i = (ddx < sep_x[i]) & (ddy < sep_y[i]); mask_i[i] = False; mask_i[j] = False
    if mask_i.any():
        return None
    ddx = np.abs(nx_j - inc.pos[:n_hard, 0]); ddy = np.abs(ny_j - inc.pos[:n_hard, 1])
    mask_j = (ddx < sep_x[j]) & (ddy < sep_y[j]); mask_j[i] = False; mask_j[j] = False
    if mask_j.any():
        return None
    if abs(nx_i - nx_j) < sep_x[i, j] and abs(ny_i - ny_j) < sep_y[i, j]:
        return None

    inc.commit_move(i, nx_i, ny_i)
    inc.commit_move(j, nx_j, ny_j)
    cost = inc.surrogate_cost()
    inc.commit_move(i, pos_i_old[0], pos_i_old[1])
    inc.commit_move(j, pos_j_old[0], pos_j_old[1])
    return cost


def _fast_cd(
    full_pos: torch.Tensor, benchmark: Benchmark, plc,
    n_passes: int, step_fracs: tuple, rng_seed: int,
    do_swaps: bool = True, n_swap_neighbors: int = 5,
    verbose: bool = False,
) -> Tuple[torch.Tensor, float]:
    """Fast surrogate-cost CD with shifts + swaps + real-proxy verification."""
    from macro_place.objective import compute_proxy_cost
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
    movable = benchmark.get_movable_mask().numpy()
    movable_hard = movable[:n_hard]
    movable_idx = np.where(movable_hard)[0]
    sx = sizes_np[:n_hard, 0]; sy = sizes_np[:n_hard, 1]
    sep_x = (sx[:, None] + sx[None, :]) / 2 + 0.001
    sep_y = (sy[:, None] + sy[None, :]) / 2 + 0.001
    half_w = sx / 2; half_h = sy / 2

    pos_np = full_pos.numpy().astype(np.float64).copy()
    inc = IncrementalProxy(benchmark, pos_np)

    rng = random.Random(rng_seed)
    cur_real = compute_proxy_cost(torch.from_numpy(pos_np).float(), benchmark, plc)["proxy_cost"]
    best_real = cur_real
    best_pos = full_pos.clone()
    if verbose:
        print(f"    [fast-cd] start real proxy={cur_real:.4f}")

    for p_idx, step_frac in enumerate(step_fracs[:n_passes]):
        step = max(cw, ch) * step_frac
        offsets = [(step, 0), (-step, 0), (0, step), (0, -step),
                   (step, step), (-step, step), (step, -step), (-step, -step)]
        order = list(movable_idx); rng.shuffle(order)
        improved = 0
        baseline_surr = inc.surrogate_cost()

        for i in order:
            ox = inc.pos[i, 0]; oy = inc.pos[i, 1]
            best_local = (ox, oy, baseline_surr)
            for dx, dy in offsets:
                nx = float(np.clip(ox + dx, half_w[i], cw - half_w[i]))
                ny = float(np.clip(oy + dy, half_h[i], ch - half_h[i]))
                ddx = np.abs(nx - inc.pos[:n_hard, 0]); ddy = np.abs(ny - inc.pos[:n_hard, 1])
                mask = (ddx < sep_x[i]) & (ddy < sep_y[i]); mask[i] = False
                if mask.any():
                    continue
                c = inc.proposed_move_cost(i, nx, ny)
                if c < best_local[2]:
                    best_local = (nx, ny, c)
            if best_local[2] < baseline_surr:
                inc.commit_move(i, best_local[0], best_local[1])
                baseline_surr = best_local[2]
                improved += 1

        if do_swaps and p_idx == n_passes - 1:
            for i in order:
                ix, iy = inc.pos[i, 0], inc.pos[i, 1]
                d = np.hypot(inc.pos[movable_idx, 0] - ix, inc.pos[movable_idx, 1] - iy)
                neighbor_local = np.argsort(d)[1 : n_swap_neighbors + 1]
                neighbors = movable_idx[neighbor_local]
                for j in neighbors:
                    if j == i: continue
                    new_cost = _try_swap(inc, int(i), int(j), sep_x, sep_y, half_w, half_h, cw, ch)
                    if new_cost is not None and new_cost < baseline_surr:
                        pi = (inc.pos[i, 0], inc.pos[i, 1])
                        pj = (inc.pos[j, 0], inc.pos[j, 1])
                        nx_i = float(np.clip(pj[0], half_w[i], cw - half_w[i]))
                        ny_i = float(np.clip(pj[1], half_h[i], ch - half_h[i]))
                        nx_j = float(np.clip(pi[0], half_w[j], cw - half_w[j]))
                        ny_j = float(np.clip(pi[1], half_h[j], ch - half_h[j]))
                        inc.commit_move(int(i), nx_i, ny_i)
                        inc.commit_move(int(j), nx_j, ny_j)
                        baseline_surr = new_cost

        new_pos_t = torch.from_numpy(inc.pos).float()
        new_real = compute_proxy_cost(new_pos_t, benchmark, plc)["proxy_cost"]
        if verbose:
            print(f"    [fast-cd] pass {p_idx+1} step={step_frac:.2%}: "
                  f"shifts={improved}/{len(order)} surrogate={baseline_surr:.4f} real={new_real:.4f}")
        if new_real < best_real:
            best_real = new_real
            best_pos = new_pos_t.clone()
        else:
            inc.pos = best_pos.numpy().astype(np.float64).copy()
            inc.pin_xy = inc._pin_xy_full()
            inc._recompute_all_hpwl()
            inc._recompute_density_full()

    return best_pos, best_real


# ─────────────────────── Helpers for v12 ───────────────────── #
def _compute_per_net_hpwl(
    pos_t: torch.Tensor,
    benchmark: Benchmark,
    device: torch.device,
) -> np.ndarray:
    """Compute true per-net HPWL (max−min of pin x + max−min of pin y) at given pos."""
    pin_data = _build_pin_data(benchmark, device)
    if pin_data is None:
        return np.zeros(0)
    n_nets = pin_data["n_nets"]
    n_real = pin_data["n_macros"]
    n_ports = pin_data["n_ports"]
    net_owners = pin_data["net_owners"]
    net_ids = pin_data["net_ids"]
    pin_offset_xy = pin_data["pin_offset_xy"]
    port_pos = benchmark.port_positions.to(device).float()

    p = pos_t.to(device).float()
    owner_pos = torch.cat([p[:n_real], port_pos], dim=0) if n_ports > 0 else p[:n_real]
    pin_xy = owner_pos[net_owners] + pin_offset_xy

    big = 1e18
    max_x = torch.full((n_nets,), -big, device=device).scatter_reduce_(
        0, net_ids, pin_xy[:, 0], reduce="amax", include_self=True)
    min_x = torch.full((n_nets,), big, device=device).scatter_reduce_(
        0, net_ids, pin_xy[:, 0], reduce="amin", include_self=True)
    max_y = torch.full((n_nets,), -big, device=device).scatter_reduce_(
        0, net_ids, pin_xy[:, 1], reduce="amax", include_self=True)
    min_y = torch.full((n_nets,), big, device=device).scatter_reduce_(
        0, net_ids, pin_xy[:, 1], reduce="amin", include_self=True)
    hpwl = (max_x - min_x) + (max_y - min_y)
    # Replace -inf for empty nets
    hpwl = torch.where(hpwl < 0, torch.zeros_like(hpwl), hpwl)
    return hpwl.detach().cpu().numpy()


def _multi_start_legalize_and_score(
    spos: np.ndarray,
    fixed_pos: np.ndarray,
    sizes_np: np.ndarray,
    movable_hard: np.ndarray,
    n_hard: int,
    cw: float,
    ch: float,
    benchmark: Benchmark,
    plc,
    soft_iters: int,
    soft_damping: float,
    orderings: List[Tuple[str, List[int]]],
) -> Tuple[float, torch.Tensor, str]:
    """Run multi-start legalize + soft jacobi from spos. Returns (best_cost, best_pos_t, best_label)."""
    from macro_place.objective import compute_proxy_cost
    best_cost = float("inf")
    best_pos = None
    best_label = "none"
    for oname, order in orderings:
        pos_np = spos.copy()
        legal = _legalize_with_order(
            pos_np[:n_hard].copy(), movable_hard, sizes_np, cw, ch, n_hard,
            fixed_pos[:n_hard], order,
        )
        pos_np[:n_hard] = legal

        full_legal = torch.from_numpy(pos_np).float()
        if plc is not None:
            cd = compute_proxy_cost(full_legal, benchmark, plc)
            if cd["overlap_count"] == 0 and cd["proxy_cost"] < best_cost:
                best_cost = cd["proxy_cost"]
                best_pos = full_legal.clone()
                best_label = f"{oname}/legal"

        pos_refined = _soft_jacobi_update(
            pos_np, benchmark, n_iters=soft_iters, damping=soft_damping,
        )
        full_ref = torch.from_numpy(pos_refined).float()
        if plc is not None:
            cd = compute_proxy_cost(full_ref, benchmark, plc)
            if cd["overlap_count"] == 0 and cd["proxy_cost"] < best_cost:
                best_cost = cd["proxy_cost"]
                best_pos = full_ref.clone()
                best_label = f"{oname}/refined"
    return best_cost, best_pos, best_label


# ─────────────────────── V12 Placer class ───────────────────── #
class V13Placer:
    """
    v12: Iterative DREAMPlace → Reweight → QA loop.

    Outer iterations (default 3):
      1. ePlace global placement (with current net_weights, warm-start
         from previous result).
      2. QA: multi-start legalization + soft Jacobi → proxy.
      3. REWEIGHT: bump net weights for nets with high HPWL/√net_size,
         decay weights toward 1 for low-HPWL nets.
      4. If proxy improved, keep; else reduce step size on reweighting.

    Compares against initial.plc-warmed baseline (v3-style) and picks
    the lowest proxy across all candidates.
    """

    def __init__(
        self, seed: int = 42, device: "torch.device | None" = None,
        gp_iters: int = 500, gp_iters_warm: int = 250,
        n_bins: int = 64,
        target_densities: tuple = (0.70, 0.85, 0.95),  # NEW: try multiple
        verbose: bool = False, soft_iters: int = 3, soft_damping: float = 0.5,
        try_init_plc: bool = True,
        n_outer_iters: int = 1,  # reweighting was net-negative; keep 1
        reweight_top_frac: float = 0.10,
        reweight_boost: float = 1.5, reweight_decay: float = 0.95,
        do_cd_refine: bool = True,
        cd_passes: int = 5,
        cd_step_fracs: tuple = (0.06, 0.04, 0.02, 0.01, 0.005),
        macro_halo_frac: float = 0.0,  # halos hurt empirically; off by default
    ):
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gp_iters = gp_iters
        self.gp_iters_warm = gp_iters_warm
        self.n_bins = n_bins
        self.target_densities = target_densities
        self.verbose = verbose
        self.soft_iters = soft_iters
        self.soft_damping = soft_damping
        self.try_init_plc = try_init_plc
        self.n_outer_iters = n_outer_iters
        self.reweight_top_frac = reweight_top_frac
        self.reweight_boost = reweight_boost
        self.reweight_decay = reweight_decay
        self.do_cd_refine = do_cd_refine
        self.cd_passes = cd_passes
        self.cd_step_fracs = cd_step_fracs
        self.macro_halo_frac = macro_halo_frac

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
        movable = benchmark.get_movable_mask().numpy()
        movable_hard = movable[:n_hard]
        fixed_pos = benchmark.macro_positions.numpy().astype(np.float64)

        plc = _load_plc(benchmark.name)
        n_nets = len(benchmark.net_pin_nodes)
        if n_nets == 0:
            return torch.from_numpy(fixed_pos).float()

        # Net pin counts (for HPWL normalization)
        net_pin_counts = np.array(
            [pins.shape[0] for pins in benchmark.net_pin_nodes], dtype=np.float64
        )

        orderings = _build_orderings(sizes_np, fixed_pos, n_hard, cw, ch)

        best_cost = float("inf"); best_pos = None; best_label = None

        # ─── Baseline path: init.plc warm + multi-start legalize + CD ───
        if self.try_init_plc:
            cost_init, pos_init, label_init = _multi_start_legalize_and_score(
                fixed_pos, fixed_pos, sizes_np, movable_hard, n_hard, cw, ch,
                benchmark, plc, self.soft_iters, self.soft_damping, orderings,
            )
            if pos_init is not None and cost_init < best_cost:
                best_cost = cost_init
                best_pos = pos_init
                best_label = f"init/{label_init}"
            if self.verbose:
                print(f"  [init.plc legal] proxy={cost_init:.4f}")
            # CD refinement on init.plc legalized
            if pos_init is not None and self.do_cd_refine and plc is not None:
                refined, refined_cost = _fast_cd(
                    pos_init, benchmark, plc,
                    n_passes=self.cd_passes, step_fracs=self.cd_step_fracs,
                    rng_seed=self.seed, do_swaps=True, n_swap_neighbors=5,
                    verbose=False,
                )
                if refined_cost < best_cost:
                    best_cost = refined_cost
                    best_pos = refined
                    best_label = f"init/CD"
                if self.verbose:
                    print(f"  [init.plc + CD] proxy={refined_cost:.4f}")

        # ─── Multi-target-density DREAMPlace runs ────────────────
        for td in self.target_densities:
            net_weights = torch.ones(n_nets, device=self.device)
            cur_pos_real = None

            for it in range(self.n_outer_iters):
                iters_this = self.gp_iters if it == 0 else self.gp_iters_warm
                gp_pos_t, _ = _eplace_global(
                    benchmark, self.device,
                    n_iters=iters_this, n_bins=self.n_bins,
                    target_density=td, verbose=False,
                    net_weights=net_weights,
                    init_pos_real=cur_pos_real,
                    macro_halo_frac=self.macro_halo_frac,
                )
                cur_pos_real = gp_pos_t.clone()
                gp_full = np.zeros_like(fixed_pos)
                gp_full[:benchmark.num_macros] = gp_pos_t.numpy().astype(np.float64)

                cost_iter, pos_iter, label_iter = _multi_start_legalize_and_score(
                    gp_full, fixed_pos, sizes_np, movable_hard, n_hard, cw, ch,
                    benchmark, plc, self.soft_iters, self.soft_damping, orderings,
                )

                if pos_iter is not None and cost_iter < best_cost:
                    best_cost = cost_iter
                    best_pos = pos_iter
                    best_label = f"td={td}/iter{it}/{label_iter}"
                if self.verbose:
                    print(f"  [td={td:.2f} iter {it}] gp_iters={iters_this} legal → proxy={cost_iter:.4f}  best={best_cost:.4f}")

                # CD refinement on this iteration's legalized result
                if pos_iter is not None and self.do_cd_refine and plc is not None:
                    refined, refined_cost = _fast_cd(
                        pos_iter, benchmark, plc,
                        n_passes=self.cd_passes, step_fracs=self.cd_step_fracs,
                        rng_seed=self.seed + it + 1, do_swaps=True, n_swap_neighbors=5,
                        verbose=False,
                    )
                    if refined_cost < best_cost:
                        best_cost = refined_cost
                        best_pos = refined
                        best_label = f"td={td}/iter{it}/CD"
                    if self.verbose:
                        print(f"  [td={td:.2f} iter {it}] + CD → proxy={refined_cost:.4f}  best={best_cost:.4f}")

                # REWEIGHT (only if multiple outer iters)
                if it < self.n_outer_iters - 1 and pos_iter is not None:
                    hpwl_per_net = _compute_per_net_hpwl(pos_iter, benchmark, self.device)
                    normalized = hpwl_per_net / np.sqrt(np.maximum(net_pin_counts, 1.0))
                    k = max(1, int(self.reweight_top_frac * n_nets))
                    top_idx = np.argsort(-normalized)[:k]
                    nw_np = net_weights.cpu().numpy()
                    nw_np = 1.0 + (nw_np - 1.0) * self.reweight_decay
                    nw_np[top_idx] *= self.reweight_boost
                    nw_np = np.clip(nw_np, 0.5, 10.0)
                    net_weights = torch.from_numpy(nw_np).float().to(self.device)

        if best_pos is None:
            best_pos = torch.from_numpy(fixed_pos.copy()).float()

        if self.verbose:
            print(f"  ▶ V13 best: {best_label} → proxy={best_cost:.4f}")

        return best_pos
