"""
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
    """Per-net weighted-average wirelength along one axis, smoothed by γ."""
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
):
    """DREAMPlace-style global placement. Returns (final_real_pos, hist_dict)."""
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    n_real = benchmark.num_macros
    sizes_real = benchmark.macro_sizes.to(device).float()
    movable_real = benchmark.get_movable_mask().to(device)
    fixed_pos = benchmark.macro_positions.to(device).float().clone()

    # Filler cells
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
        all_sizes = torch.cat([sizes_real, filler_sizes], dim=0)
    else:
        all_sizes = sizes_real
    n_total = n_real + n_filler

    # Pin data
    pin_data = _build_pin_data(benchmark, device)
    if pin_data is None:
        return fixed_pos[:n_real].cpu(), {}
    n_owners = pin_data["n_owners"]; n_nets = pin_data["n_nets"]; n_ports = pin_data["n_ports"]
    net_owners = pin_data["net_owners"]; net_ids = pin_data["net_ids"]
    pin_offset_xy = pin_data["pin_offset_xy"]
    port_pos = benchmark.port_positions.to(device).float()

    # Initial positions
    pos = torch.zeros(n_total, 2, device=device)
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
        wl_total = wl_x.sum() + wl_y.sum()
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
        wl0 = (_wa_hpwl_dim(pin_xy0[:, 0], net_ids, n_nets, state["gamma"]).sum().item()
               + _wa_hpwl_dim(pin_xy0[:, 1], net_ids, n_nets, state["gamma"]).sum().item())
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
            cur_hpwl = (_wa_hpwl_dim(pin_xy_cur[:, 0], net_ids, n_nets, state["gamma"]).sum().item()
                        + _wa_hpwl_dim(pin_xy_cur[:, 1], net_ids, n_nets, state["gamma"]).sum().item())

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


# ─────────────────────── V11 Placer class ───────────────────── #
class V11Placer:
    """v11: ePlace global + multi-start legalize + soft Jacobi (with init.plc fallback)."""

    def __init__(
        self, seed: int = 42, device: torch.device | None = None,
        gp_iters: int = 600, n_bins: int = 64, target_density: float = 0.85,
        verbose: bool = False, soft_iters: int = 3, soft_damping: float = 0.5,
        try_init_plc: bool = True,
    ):
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gp_iters = gp_iters
        self.n_bins = n_bins
        self.target_density = target_density
        self.verbose = verbose
        self.soft_iters = soft_iters
        self.soft_damping = soft_damping
        self.try_init_plc = try_init_plc

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
        movable = benchmark.get_movable_mask().numpy()
        movable_hard = movable[:n_hard]
        fixed_pos = benchmark.macro_positions.numpy().astype(np.float64)

        from macro_place.objective import compute_proxy_cost
        plc = _load_plc(benchmark.name)

        # Stage 0: ePlace global placement
        gp_pos_t, _ = _eplace_global(
            benchmark, self.device,
            n_iters=self.gp_iters, n_bins=self.n_bins,
            target_density=self.target_density, verbose=self.verbose,
        )
        gp_pos_np = np.zeros_like(fixed_pos)
        gp_pos_np[:benchmark.num_macros] = gp_pos_t.numpy().astype(np.float64)

        starts = []
        if self.try_init_plc:
            starts.append(("init", fixed_pos.copy()))
        starts.append(("eplace", gp_pos_np))

        orderings = _build_orderings(sizes_np, fixed_pos, n_hard, cw, ch)
        best_cost = float("inf"); best_pos = None; best_label = None

        for sname, spos in starts:
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
                        best_label = f"{sname}/{oname}/legal"

                pos_refined = _soft_jacobi_update(
                    pos_np, benchmark, n_iters=self.soft_iters, damping=self.soft_damping,
                )
                full_ref = torch.from_numpy(pos_refined).float()
                if plc is not None:
                    cd = compute_proxy_cost(full_ref, benchmark, plc)
                    if cd["overlap_count"] == 0 and cd["proxy_cost"] < best_cost:
                        best_cost = cd["proxy_cost"]
                        best_pos = full_ref.clone()
                        best_label = f"{sname}/{oname}/refined"

        if best_pos is None:
            best_pos = torch.from_numpy(fixed_pos.copy()).float()

        if self.verbose:
            print(f"  ▶ V11 best: {best_label} → proxy={best_cost:.4f}")

        return best_pos
