"""
v3: Multi-start legalization + fast custom soft-macro repositioning.

Insight: soft macros total 20–37% of canvas area in IBM benchmarks, so
they meaningfully drive density and congestion. plc.optimize_stdcells
(force-directed) takes 6+ minutes per call AND is destructive with default
settings. Instead, v3 implements a fast Python/NumPy Jacobi-style centroid
update that nudges soft macros toward HPWL-optimal positions in a few
seconds.

Pipeline:
  1. Multi-start hard-macro legalization (7 orderings; v2's logic).
  2. Custom soft-macro Jacobi update (start from current positions, 3-5
     rounds of pin-centroid updates). Each round: for each soft macro,
     set its position to the mean of all pin positions in nets it
     belongs to (excluding its own pin), with damping.
  3. Verify: compute actual proxy cost, keep the better of (legalize-only,
     legalize + soft refine).
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
    """
    Fast soft-macro repositioning via Jacobi centroid updates.

    For each soft macro i, compute the centroid (mean) of all pin positions
    in nets it belongs to (excluding its own pin). Move soft macro toward
    that centroid, damped to avoid oscillation.

    Hard macros, ports, and other soft macros are treated as fixed for each
    update step (Jacobi: all updates use last iteration's positions).

    Args:
        pos_np: [num_macros, 2] current positions (hard already legalized)
        benchmark: the Benchmark with net_pin_nodes
        n_iters: number of Jacobi iterations
        damping: damping factor (0=no move, 1=full move to centroid)

    Returns:
        Updated pos_np with soft macro positions refined.
    """
    n_macros = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    n_soft = n_macros - n_hard
    if n_soft == 0:
        return pos_np

    # Build pin offset table for hard macros (soft macros and ports have no offset)
    n_ports = benchmark.port_positions.shape[0]
    n_owners = n_macros + n_ports
    max_pins = max((p.shape[0] for p in benchmark.macro_pin_offsets), default=1)
    max_pins = max(max_pins, 1)
    owner_pin_offsets = np.zeros((n_owners, max_pins, 2), dtype=np.float64)
    for i, offsets in enumerate(benchmark.macro_pin_offsets):
        if offsets.shape[0] > 0:
            owner_pin_offsets[i, : offsets.shape[0]] = offsets.numpy()

    # Flatten pins
    net_owner_list = []
    net_pinidx_list = []
    net_id_list = []
    for nid, pins in enumerate(benchmark.net_pin_nodes):
        arr = pins.numpy()
        for row in arr:
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

    pin_offset_xy = owner_pin_offsets[net_owners, net_pinidx]  # [n_pins, 2]
    port_pos_np = benchmark.port_positions.numpy().astype(np.float64)

    # For each soft macro (indices [n_hard, n_macros)), gather pin entries it owns
    soft_owner_mask = (net_owners >= n_hard) & (net_owners < n_macros)
    soft_pin_indices = np.nonzero(soft_owner_mask)[0]  # indices into the flat pin arrays
    # Soft macro index per soft pin entry
    soft_macro_idx = net_owners[soft_pin_indices]  # values in [n_hard, n_macros)
    soft_pin_net_ids = net_ids[soft_pin_indices]   # net id for each soft pin entry

    pos_np = pos_np.copy()

    for it in range(n_iters):
        # Build owner positions
        owner_pos = np.zeros((n_owners, 2), dtype=np.float64)
        owner_pos[:n_macros] = pos_np
        if n_ports > 0:
            owner_pos[n_macros:] = port_pos_np

        # Pin positions
        pin_xy = owner_pos[net_owners] + pin_offset_xy  # [n_pins, 2]

        # Per-net sum and count
        net_sum_x = np.zeros(n_nets, dtype=np.float64)
        net_sum_y = np.zeros(n_nets, dtype=np.float64)
        net_count = np.zeros(n_nets, dtype=np.int64)
        np.add.at(net_sum_x, net_ids, pin_xy[:, 0])
        np.add.at(net_sum_y, net_ids, pin_xy[:, 1])
        np.add.at(net_count, net_ids, 1)

        # For each soft pin entry, contribution is (net_sum - pin_xy_self) / (net_count - 1)
        # But we need to aggregate per soft macro across all its nets
        # Sum_neighbor_x for soft macro i = Σ_{net ∋ i} (net_sum_x[net] - pin_x_of_i_in_net)
        # Count_neighbor for soft macro i = Σ_{net ∋ i} (net_count[net] - 1)
        soft_pin_x = pin_xy[soft_pin_indices, 0]
        soft_pin_y = pin_xy[soft_pin_indices, 1]
        soft_net_sum_x = net_sum_x[soft_pin_net_ids]
        soft_net_sum_y = net_sum_y[soft_pin_net_ids]
        soft_net_count = net_count[soft_pin_net_ids]

        # Per-pin neighbor contributions (sum minus self)
        contrib_x = soft_net_sum_x - soft_pin_x
        contrib_y = soft_net_sum_y - soft_pin_y
        contrib_n = soft_net_count - 1

        # Aggregate per soft macro (using soft_macro_idx)
        # soft_macro_idx values are in [n_hard, n_macros); we want zero-based [0, n_soft)
        soft_local_idx = soft_macro_idx - n_hard
        sum_x = np.zeros(n_soft, dtype=np.float64)
        sum_y = np.zeros(n_soft, dtype=np.float64)
        sum_n = np.zeros(n_soft, dtype=np.int64)
        np.add.at(sum_x, soft_local_idx, contrib_x)
        np.add.at(sum_y, soft_local_idx, contrib_y)
        np.add.at(sum_n, soft_local_idx, contrib_n)

        # Compute centroid; skip macros with no neighbors
        valid = sum_n > 0
        new_x = pos_np[n_hard:, 0].copy()
        new_y = pos_np[n_hard:, 1].copy()
        new_x[valid] = sum_x[valid] / sum_n[valid]
        new_y[valid] = sum_y[valid] / sum_n[valid]

        # Apply damped update
        pos_np[n_hard:, 0] = (1 - damping) * pos_np[n_hard:, 0] + damping * new_x
        pos_np[n_hard:, 1] = (1 - damping) * pos_np[n_hard:, 1] + damping * new_y

        # Clamp to canvas (centers, account for soft macro extent)
        soft_sizes = benchmark.macro_sizes[n_hard:].numpy().astype(np.float64)
        soft_half_w = soft_sizes[:, 0] / 2
        soft_half_h = soft_sizes[:, 1] / 2
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        pos_np[n_hard:, 0] = np.clip(pos_np[n_hard:, 0], soft_half_w, cw - soft_half_w)
        pos_np[n_hard:, 1] = np.clip(pos_np[n_hard:, 1], soft_half_h, ch - soft_half_h)

    return pos_np


class V3Placer:
    def __init__(
        self,
        seed: int = 42,
        soft_iters: int = 3,
        soft_damping: float = 0.5,
        verbose: bool = False,
    ):
        self.seed = seed
        self.soft_iters = soft_iters
        self.soft_damping = soft_damping
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

        best_cost = float("inf")
        best_pos = None
        best_name = None

        for ord_name, order in orderings:
            t0 = time.time()
            pos_np = fixed_pos.copy()
            legal = _legalize_with_order(
                pos_np[:n_hard].copy(),
                movable_hard,
                sizes_np,
                cw,
                ch,
                n_hard,
                fixed_pos[:n_hard],
                order,
            )
            pos_np[:n_hard] = legal
            t_legal = time.time() - t0

            # Evaluate legalize-only
            full_pos_legal = torch.from_numpy(pos_np).float()
            cost_legal = float("inf")
            if plc is not None:
                cost_dict = compute_proxy_cost(full_pos_legal, benchmark, plc)
                cost_legal = cost_dict["proxy_cost"]
                ovl_legal = cost_dict["overlap_count"]
            else:
                ovl_legal = 0

            # Soft refinement
            t1 = time.time()
            pos_refined = _soft_jacobi_update(
                pos_np, benchmark, n_iters=self.soft_iters, damping=self.soft_damping
            )
            full_pos_refined = torch.from_numpy(pos_refined).float()
            t_refine = time.time() - t1

            cost_refined = float("inf")
            ovl_refined = 0
            if plc is not None:
                cost_dict = compute_proxy_cost(full_pos_refined, benchmark, plc)
                cost_refined = cost_dict["proxy_cost"]
                ovl_refined = cost_dict["overlap_count"]

            # Pick the better between legalize-only and refined
            if ovl_legal == 0 and cost_legal < best_cost:
                best_cost = cost_legal
                best_pos = full_pos_legal.clone()
                best_name = f"{ord_name}/legal"
            if ovl_refined == 0 and cost_refined < best_cost:
                best_cost = cost_refined
                best_pos = full_pos_refined.clone()
                best_name = f"{ord_name}/refined"

            if self.verbose:
                print(
                    f"  [{ord_name:>14}] legal={cost_legal:.4f} refined={cost_refined:.4f} "
                    f"(t_legal={t_legal:.1f}s t_refine={t_refine:.1f}s)"
                )

        if self.verbose:
            print(f"  ▶ Best: {best_name} → proxy={best_cost:.4f}")

        if best_pos is None:
            best_pos = torch.from_numpy(fixed_pos.copy()).float()
        return best_pos
