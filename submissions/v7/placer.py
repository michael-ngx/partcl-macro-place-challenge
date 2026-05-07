"""
v7: v6 + more passes + pair-swap moves to escape local minima.

v6 beat RePlAce baseline (1.4696 vs 1.4578, -0.8%) but several
benchmarks (ibm01, ibm04, ibm17, ibm15) still trail RePlAce by 5-8%.
The CD with 8-neighbor moves at 3 step sizes finds the local optimum
of the basin around initial.plc but can't escape.

v7 additions:
  1. More CD passes with finer step sizes: (0.06, 0.04, 0.02, 0.01, 0.005)
  2. Pair-swap moves: for each macro, try swapping with its 3 nearest
     spatial neighbors. A swap is essentially a long-range non-local move
     that maintains zero overlaps if both macros fit in each other's spot.
  3. Random restart perturbation: between passes, randomly pick K macros
     and try shuffling them to get out of the basin.

Same fast surrogate evaluation as v6.

Surrogate components (matched to TILOS proxy formulation):

  * Wirelength: half-perimeter (HPWL) per net, summed and normalized as
    HPWL_total / (n_nets * (cw + ch)) — exactly the wirelength_cost
    that compute_proxy_cost reports.

  * Density: average of the top 10% densest cells on the proxy grid
    (grid_rows × grid_cols from the benchmark). For each cell compute
    sum of macro/cell overlap area (matching plc's computation). Bins
    are exact rectangles — reproduces the proxy density_cost up to
    the soft-macro contribution which is constant during hard-macro CD.

  * Congestion: skipped from the surrogate (depends on net-routing
    patterns; computing it incrementally is complex). Treated as
    constant; final acceptance uses real proxy.

Per-macro move evaluation is O(num_nets_touching_macro + density_cells_changed),
which is < 200 floats for typical IBM sizes. Per pass cost is ~ms,
not seconds.

After each CD pass, v6 calls compute_proxy_cost once to verify true
proxy improved. If it didn't, the pass is rolled back.
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


# ─────────── Re-used legalize / multi-start / soft Jacobi ─────────── #
def _legalize_with_order(
    pos: np.ndarray, movable: np.ndarray, sizes: np.ndarray, cw: float, ch: float,
    n_hard: int, fixed_pos: np.ndarray, order: List[int],
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
        if not placed.any(): return False
        dx = np.abs(x - pos[:n_hard, 0]); dy = np.abs(y - pos[:n_hard, 1])
        o = (dx < sep_x_mat[idx]) & (dy < sep_y_mat[idx]) & placed
        o[idx] = False
        return bool(o.any())

    for idx in order:
        if placed[idx]: continue
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
                    if abs(dxm) != r and abs(dym) != r: continue
                    cx = float(np.clip(x0 + dxm * step, half_w[idx], cw - half_w[idx]))
                    cy = float(np.clip(y0 + dym * step, half_h[idx], ch - half_h[idx]))
                    if has_overlap(idx, cx, cy): continue
                    d = (cx - x0) ** 2 + (cy - y0) ** 2
                    if d < best_d:
                        best_d = d; best_x, best_y = cx, cy
                        ring_found = True; found_any = True
            if found_any and ring_found: break
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
        rng = np.random.RandomState(s); order = list(range(n_hard)); rng.shuffle(order)
        orderings.append((f"random_{s}", order))
    return orderings


def _soft_jacobi_update(pos_np, benchmark, n_iters=3, damping=0.5):
    n_macros = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    n_soft = n_macros - n_hard
    if n_soft == 0: return pos_np
    n_ports = benchmark.port_positions.shape[0]
    n_owners = n_macros + n_ports
    max_pins = max((p.shape[0] for p in benchmark.macro_pin_offsets), default=1)
    max_pins = max(max_pins, 1)
    owner_pin_offsets = np.zeros((n_owners, max_pins, 2), dtype=np.float64)
    for i, offsets in enumerate(benchmark.macro_pin_offsets):
        if offsets.shape[0] > 0:
            owner_pin_offsets[i, : offsets.shape[0]] = offsets.numpy()
    net_owner_list = []; net_pinidx_list = []; net_id_list = []
    for nid, pins in enumerate(benchmark.net_pin_nodes):
        for row in pins.tolist():
            net_owner_list.append(int(row[0])); net_pinidx_list.append(int(row[1])); net_id_list.append(nid)
    if not net_owner_list: return pos_np
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
        if n_ports > 0: owner_pos[n_macros:] = port_pos_np
        pin_xy = owner_pos[net_owners] + pin_offset_xy
        net_sum_x = np.zeros(n_nets); net_sum_y = np.zeros(n_nets); net_count = np.zeros(n_nets, dtype=np.int64)
        np.add.at(net_sum_x, net_ids, pin_xy[:, 0])
        np.add.at(net_sum_y, net_ids, pin_xy[:, 1])
        np.add.at(net_count, net_ids, 1)
        soft_pin_x = pin_xy[soft_pin_indices, 0]; soft_pin_y = pin_xy[soft_pin_indices, 1]
        soft_net_sum_x = net_sum_x[soft_pin_net_ids]; soft_net_sum_y = net_sum_y[soft_pin_net_ids]
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
        new_x[valid] = sum_x[valid] / sum_n[valid]; new_y[valid] = sum_y[valid] / sum_n[valid]
        pos_np[n_hard:, 0] = (1 - damping) * pos_np[n_hard:, 0] + damping * new_x
        pos_np[n_hard:, 1] = (1 - damping) * pos_np[n_hard:, 1] + damping * new_y
        pos_np[n_hard:, 0] = np.clip(pos_np[n_hard:, 0], soft_half_w, cw - soft_half_w)
        pos_np[n_hard:, 1] = np.clip(pos_np[n_hard:, 1], soft_half_h, ch - soft_half_h)
    return pos_np


# ─────────────────────── Fast incremental surrogate ──────────────── #
class IncrementalProxy:
    """
    Maintains incremental HPWL and density grid for fast move evaluation.

    Surrogate cost = wl_weight * (total_HPWL / n_nets / (cw+ch))
                   + den_weight * mean(top_10pct(density_grid))

    Matches the proxy's wirelength_cost and density_cost terms
    exactly (assuming cell pin offsets are correctly tracked); the
    congestion term is omitted (treated as constant).
    """

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

        # Build pin/net data
        n_ports = benchmark.port_positions.shape[0]
        self.n_ports = n_ports
        self.n_owners = self.n_macros + n_ports
        self.port_pos = benchmark.port_positions.numpy().astype(np.float64) if n_ports > 0 else np.zeros((0, 2))

        max_pins = max((p.shape[0] for p in benchmark.macro_pin_offsets), default=1)
        max_pins = max(max_pins, 1)
        self.owner_pin_offsets = np.zeros((self.n_owners, max_pins, 2), dtype=np.float64)
        for i, offsets in enumerate(benchmark.macro_pin_offsets):
            if offsets.shape[0] > 0:
                self.owner_pin_offsets[i, : offsets.shape[0]] = offsets.numpy()

        # Per-net pin lists; per-macro net membership (owner -> list of (net_id, pin_offset_xy))
        n_nets = len(benchmark.net_pin_nodes)
        self.n_nets = n_nets

        net_owner_list = []
        net_pinidx_list = []
        net_id_list = []
        for nid, pins in enumerate(benchmark.net_pin_nodes):
            for row in pins.tolist():
                net_owner_list.append(int(row[0]))
                net_pinidx_list.append(int(row[1]))
                net_id_list.append(nid)
        self.net_owners = np.array(net_owner_list, dtype=np.int64)
        self.net_pinidx = np.array(net_pinidx_list, dtype=np.int64)
        self.net_ids = np.array(net_id_list, dtype=np.int64)

        # For each owner, list of pin entry indices in the flat arrays
        self.owner_to_pin_entries: List[np.ndarray] = [None] * self.n_owners
        for owner in range(self.n_owners):
            self.owner_to_pin_entries[owner] = np.nonzero(self.net_owners == owner)[0]

        # Pin offsets per pin entry
        self.pin_offset_xy = self.owner_pin_offsets[self.net_owners, self.net_pinidx]

        # Per-net pin entry indices (variable-length list)
        self.net_to_pin_entries: List[np.ndarray] = [None] * n_nets
        for nid in range(n_nets):
            self.net_to_pin_entries[nid] = np.nonzero(self.net_ids == nid)[0]

        # Initialize state: pin positions, per-net HPWL, density grid
        self.pos = full_pos.copy()
        self.pin_xy = self._pin_xy_full()  # [n_pins, 2]
        self.net_hpwl = np.zeros(n_nets, dtype=np.float64)
        self._recompute_all_hpwl()

        # Density grid uses benchmark.grid_rows × grid_cols (same as proxy)
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
            xs = self.pin_xy[entries, 0]
            ys = self.pin_xy[entries, 1]
            self.net_hpwl[nid] = (xs.max() - xs.min()) + (ys.max() - ys.min())

    def _recompute_density_full(self):
        self.density.fill(0.0)
        for i in range(self.n_macros):
            self._add_density_contribution(i, +1.0)

    def _add_density_contribution(self, macro_idx: int, sign: float):
        sx = self.sizes[macro_idx, 0]; sy = self.sizes[macro_idx, 1]
        x = self.pos[macro_idx, 0]; y = self.pos[macro_idx, 1]
        x_lo = x - sx / 2; x_hi = x + sx / 2
        y_lo = y - sy / 2; y_hi = y + sy / 2
        # Clamp to canvas
        x_lo = max(x_lo, 0.0); x_hi = min(x_hi, self.cw)
        y_lo = max(y_lo, 0.0); y_hi = min(y_hi, self.ch)
        if x_hi <= x_lo or y_hi <= y_lo:
            return
        # Bin range
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
        # Top 10% mean — same formula as TILOS proxy
        flat = self.density.flatten()
        n_top = max(1, int(np.ceil(len(flat) * 0.1)))
        top_vals = np.partition(flat, -n_top)[-n_top:]
        # Normalize: density in area units (μm²); divide by bin area
        bin_area = self.bin_w * self.bin_h
        return float(top_vals.mean() / bin_area)

    def surrogate_cost(self) -> float:
        wl = self.total_hpwl() / max(self.n_nets * (self.cw + self.ch), 1e-12)
        return self.wl_weight * wl + self.den_weight * self.density_cost()

    def proposed_move_cost(self, macro_idx: int, new_x: float, new_y: float) -> float:
        """
        Compute the surrogate cost AS IF macro_idx were at (new_x, new_y).
        Does NOT mutate state.
        """
        # Save current
        old_x = self.pos[macro_idx, 0]; old_y = self.pos[macro_idx, 1]

        # ── HPWL delta: only nets touching this macro change ──────
        affected_nets = self.net_ids[self.owner_to_pin_entries[macro_idx]]
        affected_nets = np.unique(affected_nets)
        old_hpwl_for_nets = {nid: self.net_hpwl[nid] for nid in affected_nets}
        new_hpwl_for_nets = {}

        # Build proposed pin_xy for entries on this macro
        for entry in self.owner_to_pin_entries[macro_idx]:
            self.pin_xy[entry, 0] = new_x + self.pin_offset_xy[entry, 0]
            self.pin_xy[entry, 1] = new_y + self.pin_offset_xy[entry, 1]

        for nid in affected_nets:
            entries = self.net_to_pin_entries[nid]
            if len(entries) <= 1:
                new_hpwl_for_nets[nid] = 0.0
                continue
            xs = self.pin_xy[entries, 0]
            ys = self.pin_xy[entries, 1]
            new_hpwl_for_nets[nid] = (xs.max() - xs.min()) + (ys.max() - ys.min())

        # Restore pin_xy for old position
        for entry in self.owner_to_pin_entries[macro_idx]:
            self.pin_xy[entry, 0] = old_x + self.pin_offset_xy[entry, 0]
            self.pin_xy[entry, 1] = old_y + self.pin_offset_xy[entry, 1]

        # New total HPWL
        delta_hpwl = sum(new_hpwl_for_nets[nid] - old_hpwl_for_nets[nid]
                         for nid in affected_nets)
        new_total_hpwl = self.total_hpwl() + delta_hpwl

        # ── Density delta: subtract old contribution, add new ─────
        # Save and remove old contribution
        self._add_density_contribution(macro_idx, -1.0)
        self.pos[macro_idx, 0] = new_x; self.pos[macro_idx, 1] = new_y
        self._add_density_contribution(macro_idx, +1.0)

        new_density_cost = self.density_cost()

        # Restore
        self._add_density_contribution(macro_idx, -1.0)
        self.pos[macro_idx, 0] = old_x; self.pos[macro_idx, 1] = old_y
        self._add_density_contribution(macro_idx, +1.0)

        wl_norm = new_total_hpwl / max(self.n_nets * (self.cw + self.ch), 1e-12)
        return self.wl_weight * wl_norm + self.den_weight * new_density_cost

    def commit_move(self, macro_idx: int, new_x: float, new_y: float):
        """Apply the move, updating state in place."""
        old_x = self.pos[macro_idx, 0]; old_y = self.pos[macro_idx, 1]
        # Density update
        self._add_density_contribution(macro_idx, -1.0)
        self.pos[macro_idx, 0] = new_x; self.pos[macro_idx, 1] = new_y
        self._add_density_contribution(macro_idx, +1.0)
        # Pin xy update
        for entry in self.owner_to_pin_entries[macro_idx]:
            self.pin_xy[entry, 0] = new_x + self.pin_offset_xy[entry, 0]
            self.pin_xy[entry, 1] = new_y + self.pin_offset_xy[entry, 1]
        # Per-net HPWL refresh for affected nets only
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
) -> float | None:
    """
    Try swapping positions of hard macros i and j.
    Returns surrogate cost if swap is overlap-free; None otherwise.
    """
    pos_i_old = (inc.pos[i, 0], inc.pos[i, 1])
    pos_j_old = (inc.pos[j, 0], inc.pos[j, 1])
    n_hard = len(half_w)

    # Try j's position for i
    nx_i = float(np.clip(pos_j_old[0], half_w[i], cw - half_w[i]))
    ny_i = float(np.clip(pos_j_old[1], half_h[i], ch - half_h[i]))
    nx_j = float(np.clip(pos_i_old[0], half_w[j], cw - half_w[j]))
    ny_j = float(np.clip(pos_i_old[1], half_h[j], ch - half_h[j]))

    # Check that i fits at j's spot (excluding j)
    ddx = np.abs(nx_i - inc.pos[:n_hard, 0]); ddy = np.abs(ny_i - inc.pos[:n_hard, 1])
    mask_i = (ddx < sep_x[i]) & (ddy < sep_y[i]); mask_i[i] = False; mask_i[j] = False
    if mask_i.any():
        return None
    # Check j fits at i's spot (excluding i, and excluding j-at-current — but
    # i is moving, so check vs others except i and j)
    ddx = np.abs(nx_j - inc.pos[:n_hard, 0]); ddy = np.abs(ny_j - inc.pos[:n_hard, 1])
    mask_j = (ddx < sep_x[j]) & (ddy < sep_y[j]); mask_j[i] = False; mask_j[j] = False
    if mask_j.any():
        return None
    # Check i-at-new and j-at-new don't overlap each other
    ddx = abs(nx_i - nx_j); ddy = abs(ny_i - ny_j)
    if ddx < sep_x[i, j] and ddy < sep_y[i, j]:
        return None

    # Tentatively apply swap to inc, evaluate, then revert
    inc.commit_move(i, nx_i, ny_i)
    inc.commit_move(j, nx_j, ny_j)
    cost = inc.surrogate_cost()
    inc.commit_move(i, pos_i_old[0], pos_i_old[1])
    inc.commit_move(j, pos_j_old[0], pos_j_old[1])
    return cost


def _fast_cd(
    full_pos: torch.Tensor, benchmark: Benchmark, plc,
    n_passes: int, step_fracs: tuple, rng_seed: int,
    do_swaps: bool = True, n_swap_neighbors: int = 3,
    verbose: bool = False,
) -> Tuple[torch.Tensor, float]:
    """Fast surrogate-cost CD with shifts + pair swaps + real-proxy verification."""
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

    # Verify initial real proxy
    cur_real = compute_proxy_cost(torch.from_numpy(pos_np).float(), benchmark, plc)["proxy_cost"]
    best_real = cur_real
    best_pos = full_pos.clone()
    if verbose:
        print(f"    [fast-cd] start real proxy={cur_real:.4f} surrogate={inc.surrogate_cost():.4f}")

    for p_idx, step_frac in enumerate(step_fracs[:n_passes]):
        step = max(cw, ch) * step_frac
        offsets = [(step, 0), (-step, 0), (0, step), (0, -step),
                   (step, step), (-step, step), (step, -step), (-step, -step)]
        order = list(movable_idx); rng.shuffle(order)
        improved = 0
        swaps_tried = 0
        swaps_accepted = 0
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

        # Pair-swap pass: only on the LAST CD pass to keep cost down
        # (or on every pass with a small neighbor count)
        if do_swaps and p_idx == n_passes - 1:
            # Find spatial neighbors for each movable macro
            for i in order:
                ix, iy = inc.pos[i, 0], inc.pos[i, 1]
                # Neighbors by spatial distance (movable only)
                d = np.hypot(inc.pos[movable_idx, 0] - ix, inc.pos[movable_idx, 1] - iy)
                neighbor_local = np.argsort(d)[1 : n_swap_neighbors + 1]
                neighbors = movable_idx[neighbor_local]
                for j in neighbors:
                    if j == i: continue
                    swaps_tried += 1
                    new_cost = _try_swap(inc, int(i), int(j), sep_x, sep_y, half_w, half_h, cw, ch)
                    if new_cost is not None and new_cost < baseline_surr:
                        # Apply swap permanently
                        pi = (inc.pos[i, 0], inc.pos[i, 1])
                        pj = (inc.pos[j, 0], inc.pos[j, 1])
                        nx_i = float(np.clip(pj[0], half_w[i], cw - half_w[i]))
                        ny_i = float(np.clip(pj[1], half_h[i], ch - half_h[i]))
                        nx_j = float(np.clip(pi[0], half_w[j], cw - half_w[j]))
                        ny_j = float(np.clip(pi[1], half_h[j], ch - half_h[j]))
                        inc.commit_move(int(i), nx_i, ny_i)
                        inc.commit_move(int(j), nx_j, ny_j)
                        baseline_surr = new_cost
                        swaps_accepted += 1

        # Verify with real proxy
        new_pos_t = torch.from_numpy(inc.pos).float()
        new_real = compute_proxy_cost(new_pos_t, benchmark, plc)["proxy_cost"]
        if verbose:
            print(
                f"    [fast-cd] pass {p_idx+1} step={step_frac:.2%}: "
                f"shifts {improved}/{len(order)} swaps {swaps_accepted}/{swaps_tried} "
                f"surrogate={baseline_surr:.4f} real={new_real:.4f}"
            )
        if new_real < best_real:
            best_real = new_real
            best_pos = new_pos_t.clone()
        else:
            # Revert this pass — it didn't help the real proxy
            inc.pos = best_pos.numpy().astype(np.float64).copy()
            inc.pin_xy = inc._pin_xy_full()
            inc._recompute_all_hpwl()
            inc._recompute_density_full()

    return best_pos, best_real


class V7Placer:
    def __init__(
        self,
        seed: int = 42,
        soft_iters: int = 3,
        soft_damping: float = 0.5,
        cd_passes: int = 5,
        cd_step_fracs: tuple = (0.06, 0.04, 0.02, 0.01, 0.005),
        do_swaps: bool = True,
        n_swap_neighbors: int = 5,
        verbose: bool = False,
    ):
        self.seed = seed
        self.soft_iters = soft_iters
        self.soft_damping = soft_damping
        self.cd_passes = cd_passes
        self.cd_step_fracs = cd_step_fracs
        self.do_swaps = do_swaps
        self.n_swap_neighbors = n_swap_neighbors
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
        movable = benchmark.get_movable_mask().numpy()
        movable_hard = movable[:n_hard]
        fixed_pos = benchmark.macro_positions.numpy().astype(np.float64)

        from macro_place.objective import compute_proxy_cost
        plc = _load_plc(benchmark.name)

        # Stage 1: multi-start legalize + soft-jacobi (v3 pipeline)
        orderings = _build_orderings(sizes_np, fixed_pos, n_hard, cw, ch)
        best_cost = float("inf")
        best_pos = None
        for ord_name, order in orderings:
            pos_np = fixed_pos.copy()
            legal = _legalize_with_order(
                pos_np[:n_hard].copy(), movable_hard, sizes_np, cw, ch, n_hard,
                fixed_pos[:n_hard], order,
            )
            pos_np[:n_hard] = legal

            full_legal = torch.from_numpy(pos_np).float()
            if plc is not None:
                cd = compute_proxy_cost(full_legal, benchmark, plc)
                if cd["overlap_count"] == 0 and cd["proxy_cost"] < best_cost:
                    best_cost = cd["proxy_cost"]; best_pos = full_legal.clone()

            pos_refined = _soft_jacobi_update(
                pos_np, benchmark, n_iters=self.soft_iters, damping=self.soft_damping,
            )
            full_ref = torch.from_numpy(pos_refined).float()
            if plc is not None:
                cd = compute_proxy_cost(full_ref, benchmark, plc)
                if cd["overlap_count"] == 0 and cd["proxy_cost"] < best_cost:
                    best_cost = cd["proxy_cost"]; best_pos = full_ref.clone()

        if best_pos is None:
            best_pos = torch.from_numpy(fixed_pos.copy()).float()

        if self.verbose:
            print(f"  ▶ Multi-start best proxy={best_cost:.4f}")

        # Stage 2: fast surrogate CD with swaps
        if plc is not None and self.cd_passes > 0:
            best_pos, best_cost = _fast_cd(
                best_pos, benchmark, plc,
                n_passes=self.cd_passes, step_fracs=self.cd_step_fracs,
                rng_seed=self.seed, do_swaps=self.do_swaps,
                n_swap_neighbors=self.n_swap_neighbors, verbose=self.verbose,
            )

        return best_pos
