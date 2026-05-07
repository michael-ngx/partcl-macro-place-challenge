"""
v4: v3 baseline + proxy-aware coordinate descent on hard macros.

After v3 produces a near-optimal multi-start legalized + soft-refined
placement, v4 runs a second-stage coordinate descent that directly
minimizes the actual proxy cost (compute_proxy_cost): for each movable
hard macro, try a handful of small shifts in cardinal directions, check
for overlaps, and apply the move with the lowest proxy cost (greedy).

Budget: each plc.compute_proxy_cost call is ~50 ms. Per-pass cost is
roughly num_movable_hard_macros × num_candidates × 50 ms. With 4
candidates per macro and 2 passes, the largest benchmark (ibm18, ~537
movable macros) takes ~ 4*537*0.05*2 ≈ 215 s extra. All 17 benchmarks
fit comfortably within the 1-hour-per-benchmark cap.
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


# ─────────────────────── v3 helpers (re-used) ─────────────────────── #
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


def _build_orderings(
    sizes_np: np.ndarray, fixed_pos: np.ndarray, n_hard: int, cw: float, ch: float
) -> List[Tuple[str, List[int]]]:
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


def _soft_jacobi_update(
    pos_np: np.ndarray,
    benchmark: Benchmark,
    n_iters: int = 3,
    damping: float = 0.5,
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
    n_pins = len(net_owners)
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


# ─────────────────────── v4: proxy-aware coordinate descent ───────── #
def _proxy_aware_cd(
    full_pos: torch.Tensor,
    benchmark: Benchmark,
    plc,
    movable_hard_idx: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    n_hard: int,
    n_passes: int = 2,
    step_fracs: tuple = (0.04, 0.02),
    rng_seed: int = 42,
    verbose: bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    Greedy coordinate descent with actual proxy cost as objective.
    For each movable hard macro, try shifts in 4 cardinal + 4 diagonal
    directions at the given step fraction of canvas. Keep best.
    """
    from macro_place.objective import compute_proxy_cost

    rng = random.Random(rng_seed)
    pos_np = full_pos.numpy().astype(np.float64).copy()

    cur_cost_dict = compute_proxy_cost(full_pos, benchmark, plc)
    cur_cost = cur_cost_dict["proxy_cost"]
    best_cost = cur_cost
    best_pos = full_pos.clone()
    if verbose:
        print(f"    [cd] start proxy={cur_cost:.4f}")

    for p_idx, step_frac in enumerate(step_fracs[:n_passes]):
        step = max(cw, ch) * step_frac
        # Direction set: 8 neighbors + zero (no-op)
        offsets = [
            (step, 0), (-step, 0), (0, step), (0, -step),
            (step, step), (-step, step), (step, -step), (-step, -step),
        ]
        idx_order = list(movable_hard_idx)
        rng.shuffle(idx_order)
        improved_count = 0

        for i in idx_order:
            ox = pos_np[i, 0]
            oy = pos_np[i, 1]
            best_local = (ox, oy, cur_cost)

            for dx, dy in offsets:
                nx = float(np.clip(ox + dx, half_w[i], cw - half_w[i]))
                ny = float(np.clip(oy + dy, half_h[i], ch - half_h[i]))
                # Quick overlap check
                ddx = np.abs(nx - pos_np[:n_hard, 0])
                ddy = np.abs(ny - pos_np[:n_hard, 1])
                mask = (ddx < sep_x[i]) & (ddy < sep_y[i])
                mask[i] = False
                if mask.any():
                    continue
                # Evaluate
                pos_np[i, 0] = nx
                pos_np[i, 1] = ny
                t = torch.from_numpy(pos_np).float()
                c = compute_proxy_cost(t, benchmark, plc)["proxy_cost"]
                if c < best_local[2]:
                    best_local = (nx, ny, c)
                # Restore for next candidate
                pos_np[i, 0] = ox
                pos_np[i, 1] = oy

            if best_local[2] < cur_cost:
                pos_np[i, 0] = best_local[0]
                pos_np[i, 1] = best_local[1]
                cur_cost = best_local[2]
                improved_count += 1
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_pos = torch.from_numpy(pos_np.copy()).float()

        if verbose:
            print(
                f"    [cd] pass {p_idx+1} step={step_frac:.2%}: "
                f"improved {improved_count}/{len(idx_order)} → proxy={cur_cost:.4f}"
            )

    return best_pos, best_cost


class V4Placer:
    def __init__(
        self,
        seed: int = 42,
        soft_iters: int = 3,
        soft_damping: float = 0.5,
        cd_passes: int = 2,
        cd_step_fracs: tuple = (0.04, 0.02),
        verbose: bool = False,
    ):
        self.seed = seed
        self.soft_iters = soft_iters
        self.soft_damping = soft_damping
        self.cd_passes = cd_passes
        self.cd_step_fracs = cd_step_fracs
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
        movable = benchmark.get_movable_mask().numpy()
        movable_hard = movable[:n_hard]
        fixed_pos = benchmark.macro_positions.numpy().astype(np.float64)

        from macro_place.objective import compute_proxy_cost

        plc = _load_plc(benchmark.name)
        orderings = _build_orderings(sizes_np, fixed_pos, n_hard, cw, ch)

        # Stage 1+2: multi-start legalize + optional soft refine; pick best
        best_cost = float("inf")
        best_pos = None
        best_name = None

        for ord_name, order in orderings:
            pos_np = fixed_pos.copy()
            legal = _legalize_with_order(
                pos_np[:n_hard].copy(), movable_hard, sizes_np, cw, ch,
                n_hard, fixed_pos[:n_hard], order,
            )
            pos_np[:n_hard] = legal

            full_pos_legal = torch.from_numpy(pos_np).float()
            if plc is not None:
                cd = compute_proxy_cost(full_pos_legal, benchmark, plc)
                if cd["overlap_count"] == 0 and cd["proxy_cost"] < best_cost:
                    best_cost = cd["proxy_cost"]
                    best_pos = full_pos_legal.clone()
                    best_name = f"{ord_name}/legal"

            pos_refined = _soft_jacobi_update(
                pos_np, benchmark, n_iters=self.soft_iters, damping=self.soft_damping
            )
            full_pos_refined = torch.from_numpy(pos_refined).float()
            if plc is not None:
                cd = compute_proxy_cost(full_pos_refined, benchmark, plc)
                if cd["overlap_count"] == 0 and cd["proxy_cost"] < best_cost:
                    best_cost = cd["proxy_cost"]
                    best_pos = full_pos_refined.clone()
                    best_name = f"{ord_name}/refined"

        if best_pos is None:
            best_pos = torch.from_numpy(fixed_pos.copy()).float()

        if self.verbose:
            print(f"  ▶ Multi-start best: {best_name} → proxy={best_cost:.4f}")

        # Stage 3: proxy-aware CD on top of best
        if plc is not None and self.cd_passes > 0:
            sx = sizes_np[:n_hard, 0]
            sy = sizes_np[:n_hard, 1]
            sep_x = (sx[:, None] + sx[None, :]) / 2 + 0.001
            sep_y = (sy[:, None] + sy[None, :]) / 2 + 0.001
            half_w = sx / 2
            half_h = sy / 2
            movable_hard_idx = np.where(movable_hard)[0]
            best_pos, best_cost = _proxy_aware_cd(
                best_pos, benchmark, plc, movable_hard_idx,
                sep_x, sep_y, half_w, half_h, cw, ch, n_hard,
                n_passes=self.cd_passes, step_fracs=self.cd_step_fracs,
                rng_seed=self.seed, verbose=self.verbose,
            )

        return best_pos
