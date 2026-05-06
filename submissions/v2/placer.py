"""
v2: Multi-start legalization (different macro orderings) + proxy-aware selection.

v1 (single-pass legalization with largest-first ordering) achieves 1.49 avg.
The legalization output is sensitive to which macros get placed first — early
macros claim space close to their original positions, late macros get pushed
into whatever's left. v2 runs the legalization several times with different
orderings and picks the placement with the lowest actual proxy cost.

Orderings tried:
  - largest-first (default v1)
  - smallest-first
  - x-coordinate sort (left to right)
  - y-coordinate sort (bottom to top)
  - distance-from-center descending (center-out)
  - distance-from-center ascending (edges-in)
  - 3 random shuffles

Budget per benchmark: ~9 legalize passes (~1 s each) + 9 proxy evals (~0.05 s
each) ≈ 10 s. Total ≈ 170 s for all 17 IBM benchmarks.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Callable, List, Tuple

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
    """
    Greedy minimum-displacement legalization with caller-supplied macro order.
    Order: list of macro indices in [0, n_hard); macros placed in this order.
    """
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
    """Build a list of (name, order) tuples for the multi-start sweep."""
    orderings = []

    # Largest-first by area (proven good in v1)
    area = sizes_np[:n_hard, 0] * sizes_np[:n_hard, 1]
    orderings.append(("area_desc", list(np.argsort(-area))))
    # Smallest-first by area
    orderings.append(("area_asc", list(np.argsort(area))))
    # Distance from center descending (place center-most first)
    cx, cy = cw / 2, ch / 2
    dc = (fixed_pos[:n_hard, 0] - cx) ** 2 + (fixed_pos[:n_hard, 1] - cy) ** 2
    orderings.append(("center_first", list(np.argsort(dc))))
    # Distance from center ascending (place edges first)
    orderings.append(("edges_first", list(np.argsort(-dc))))
    # Width descending
    orderings.append(("width_desc", list(np.argsort(-sizes_np[:n_hard, 0]))))
    # 2 random shuffles
    for s in (1, 7):
        rng = np.random.RandomState(s)
        order = list(range(n_hard))
        rng.shuffle(order)
        orderings.append((f"random_{s}", order))
    return orderings


class V2Placer:
    def __init__(
        self,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.seed = seed
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

        # Build orderings to try
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
            full_pos = torch.from_numpy(pos_np).float()
            t_legal = time.time() - t0

            # Quick proxy cost (only if plc is available)
            if plc is not None:
                t1 = time.time()
                cost_dict = compute_proxy_cost(full_pos, benchmark, plc)
                cost = cost_dict["proxy_cost"]
                overlap = cost_dict["overlap_count"]
                t_eval = time.time() - t1
                if self.verbose:
                    print(
                        f"  [{ord_name:>14}] proxy={cost:.4f} ovl={overlap} "
                        f"legal={t_legal:.2f}s eval={t_eval:.2f}s"
                    )
                if overlap > 0:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_pos = full_pos.clone()
                    best_name = ord_name
            else:
                # No plc available — return first legalized result
                if best_pos is None:
                    best_pos = full_pos.clone()
                    best_name = ord_name

        if self.verbose:
            print(f"  ▶ Best: {best_name} → proxy={best_cost:.4f}")

        if best_pos is None:
            best_pos = torch.from_numpy(fixed_pos.copy()).float()
        return best_pos
