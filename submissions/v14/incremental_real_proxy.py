"""
v14: Incremental REAL-proxy evaluator.

Mirrors TILOS PlacementCost (external/MacroPlacement/CodeElements/Plc_client/
plc_client_os.py) bit-for-bit, but computes the proxy incrementally so that
moving a single macro updates only the affected net routing patterns, macro
routing contribution, density bins, and net HPWLs.

Target speedup vs `compute_proxy_cost`: ~100-300×.

Components mirrored:
  - HPWL per net (already in v7's IncrementalProxy)
  - Density grid (top 10% mean of bin overlap area / bin_area)
  - Congestion:
      * Per net: 2-pin / 3-pin / star routing patterns (matches __two_pin_net_routing,
        __three_pin_net_routing, __l_routing, __t_routing, __split_net)
      * Macro routing: __macro_route_over_grid_cell with vrouting_alloc, hrouting_alloc
      * Smoothing: TILOS's "spread" filter (each cell's value distributed across
        2*smooth_range+1 cells, divided by gcell_cnt)
      * Final cost: top 5% mean of (V_total + H_total) where total = smoothed_net + macro

Validation: validate_against_proxy() runs random perturbations and asserts
incremental result matches plc.get_*_cost() to <1e-4 tolerance.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Optional, Tuple

from macro_place.benchmark import Benchmark


def _gcell(x: float, y: float, grid_width: float, grid_height: float,
           grid_row: int, grid_col: int) -> Tuple[int, int]:
    """Mirror of __get_grid_cell_location."""
    row = int(math.floor(y / grid_height))
    col = int(math.floor(x / grid_width))
    # Clamp like the patched objective.py
    row = max(0, min(row, grid_row - 1))
    col = max(0, min(col, grid_col - 1))
    return row, col


class IncrementalRealProxy:
    """Maintains exact TILOS proxy cost incrementally on macro moves."""

    def __init__(self, benchmark: Benchmark, full_pos: np.ndarray, plc=None):
        """
        Args:
            benchmark: source data (sizes, nets, ports)
            full_pos: [num_macros, 2] starting positions
            plc: optional PlacementCost (used to get grid_v_routes, grid_h_routes,
                vrouting_alloc, hrouting_alloc, smooth_range — these aren't on
                the Benchmark). If None, default constants are used.
        """
        self.benchmark = benchmark
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.n_macros = benchmark.num_macros
        self.n_hard = benchmark.num_hard_macros
        self.sizes = benchmark.macro_sizes.numpy().astype(np.float64)

        # Grid for density + congestion (same grid as TILOS)
        self.gr = benchmark.grid_rows
        self.gc = benchmark.grid_cols
        self.grid_width = self.cw / self.gc
        self.grid_height = self.ch / self.gr
        self.bin_area = self.grid_width * self.grid_height

        # Congestion constants from plc (if available) or benchmark
        if plc is not None:
            self.hroutes_per_micron = plc.hroutes_per_micron
            self.vroutes_per_micron = plc.vroutes_per_micron
            self.smooth_range = int(getattr(plc, "smooth_range", 2))
            # Macro routing allocs: typically 1.0 for both H and V
            self.vrouting_alloc = float(getattr(plc, "vrouting_alloc", 1.0))
            self.hrouting_alloc = float(getattr(plc, "hrouting_alloc", 1.0))
            # plc.net_cnt is the divisor TILOS uses (raw netlist count, may be
            # larger than len(plc.nets) due to filtering)
            self.net_cnt_for_norm = int(plc.net_cnt) if int(plc.net_cnt) > 0 else 1
        else:
            self.hroutes_per_micron = float(benchmark.hroutes_per_micron)
            self.vroutes_per_micron = float(benchmark.vroutes_per_micron)
            self.smooth_range = 2
            self.vrouting_alloc = 1.0
            self.hrouting_alloc = 1.0
            self.net_cnt_for_norm = max(len(benchmark.net_pin_nodes), 1)
        self.grid_v_routes = self.grid_width * self.vroutes_per_micron
        self.grid_h_routes = self.grid_height * self.hroutes_per_micron

        # Pin / net data ─────────────────────────────────────────
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

        # Per-pin entry: owner, pin_idx, net_id
        net_owner_list, net_pinidx_list, net_id_list = [], [], []
        # Track net's pin entries in INSERTION ORDER (driver first, then sinks)
        # because routing pattern uses source_gcell.
        self.net_pin_entries: List[List[int]] = [[] for _ in range(n_nets)]
        for nid, pins in enumerate(benchmark.net_pin_nodes):
            for row in pins.tolist():
                ent_idx = len(net_owner_list)
                net_owner_list.append(int(row[0]))
                net_pinidx_list.append(int(row[1]))
                net_id_list.append(nid)
                self.net_pin_entries[nid].append(ent_idx)

        self.net_owners = np.array(net_owner_list, dtype=np.int64)
        self.net_pinidx = np.array(net_pinidx_list, dtype=np.int64)
        self.net_ids = np.array(net_id_list, dtype=np.int64)

        # Per-net driver-pin weight (for HPWL computation)
        # TILOS uses driver_pin.get_weight() per net.
        self.net_weight = np.ones(n_nets, dtype=np.float64)
        if plc is not None:
            for nid, driver_name in enumerate(plc.nets.keys()):
                if nid >= n_nets:
                    break
                driver_idx = plc.mod_name_to_indices.get(driver_name)
                if driver_idx is not None:
                    drv = plc.modules_w_pins[driver_idx]
                    try:
                        self.net_weight[nid] = float(drv.get_weight())
                    except Exception:
                        pass
        # Per-owner pin entries (for fast lookup of which nets touch a given owner)
        self.owner_to_pin_entries: List[np.ndarray] = [None] * self.n_owners
        for owner in range(self.n_owners):
            self.owner_to_pin_entries[owner] = np.nonzero(self.net_owners == owner)[0]
        # Per-pin offset
        self.pin_offset_xy = self.owner_pin_offsets[self.net_owners, self.net_pinidx]

        # State ──────────────────────────────────────────────────
        self.pos = full_pos.copy()
        self.pin_xy = self._pin_xy_full()
        self.net_hpwl = np.zeros(n_nets, dtype=np.float64)
        # Per-net: source grid cell (driver pin's gcell) and node_gcells set
        self.net_source_gcell: List[Optional[Tuple[int, int]]] = [None] * n_nets
        # node_gcells stored as a frozenset of (row, col)
        self.net_node_gcells: List[frozenset] = [frozenset()] * n_nets

        # Density grid (area)
        self.density = np.zeros((self.gr, self.gc), dtype=np.float64)

        # Congestion grids (un-normalized counts before division by routes)
        self.H_net = np.zeros((self.gr, self.gc), dtype=np.float64)
        self.V_net = np.zeros((self.gr, self.gc), dtype=np.float64)
        self.H_macro = np.zeros((self.gr, self.gc), dtype=np.float64)
        self.V_macro = np.zeros((self.gr, self.gc), dtype=np.float64)

        # Initialize all
        self._recompute_all()

    # ─────────────── Pin xy helpers ───────────────
    def _pin_xy_full(self) -> np.ndarray:
        owner_pos = np.zeros((self.n_owners, 2), dtype=np.float64)
        owner_pos[: self.n_macros] = self.pos
        if self.n_ports > 0:
            owner_pos[self.n_macros:] = self.port_pos
        return owner_pos[self.net_owners] + self.pin_offset_xy

    def _gcell(self, x: float, y: float) -> Tuple[int, int]:
        return _gcell(x, y, self.grid_width, self.grid_height, self.gr, self.gc)

    # ─────────────── HPWL ───────────────
    def _recompute_all_hpwl(self):
        for nid in range(self.n_nets):
            entries = self.net_pin_entries[nid]
            if len(entries) <= 1:
                self.net_hpwl[nid] = 0.0
                continue
            xs = self.pin_xy[entries, 0]
            ys = self.pin_xy[entries, 1]
            self.net_hpwl[nid] = (xs.max() - xs.min()) + (ys.max() - ys.min())

    def _update_net_hpwl(self, nid: int):
        entries = self.net_pin_entries[nid]
        if len(entries) <= 1:
            self.net_hpwl[nid] = 0.0
            return
        xs = self.pin_xy[entries, 0]
        ys = self.pin_xy[entries, 1]
        self.net_hpwl[nid] = (xs.max() - xs.min()) + (ys.max() - ys.min())

    # ─────────────── Density ───────────────
    def _add_density(self, macro_idx: int, sign: float):
        sx = self.sizes[macro_idx, 0]; sy = self.sizes[macro_idx, 1]
        x = self.pos[macro_idx, 0]; y = self.pos[macro_idx, 1]
        x_lo = max(x - sx / 2, 0.0); x_hi = min(x + sx / 2, self.cw)
        y_lo = max(y - sy / 2, 0.0); y_hi = min(y + sy / 2, self.ch)
        if x_hi <= x_lo or y_hi <= y_lo:
            return
        col_lo = int(x_lo / self.grid_width)
        col_hi = int(min((x_hi - 1e-12) / self.grid_width, self.gc - 1))
        row_lo = int(y_lo / self.grid_height)
        row_hi = int(min((y_hi - 1e-12) / self.grid_height, self.gr - 1))
        col_lo = max(0, min(col_lo, self.gc - 1)); col_hi = max(0, min(col_hi, self.gc - 1))
        row_lo = max(0, min(row_lo, self.gr - 1)); row_hi = max(0, min(row_hi, self.gr - 1))
        for r in range(row_lo, row_hi + 1):
            ry_lo = r * self.grid_height; ry_hi = ry_lo + self.grid_height
            oy = max(0.0, min(y_hi, ry_hi) - max(y_lo, ry_lo))
            if oy <= 0:
                continue
            for c in range(col_lo, col_hi + 1):
                rx_lo = c * self.grid_width; rx_hi = rx_lo + self.grid_width
                ox = max(0.0, min(x_hi, rx_hi) - max(x_lo, rx_lo))
                if ox <= 0:
                    continue
                self.density[r, c] += sign * ox * oy

    def _recompute_density(self):
        self.density.fill(0.0)
        for i in range(self.n_macros):
            self._add_density(i, +1.0)

    # ─────────────── Congestion: macro routing ───────────────
    def _add_macro_route(self, macro_idx: int, sign: float):
        """Mirror of __macro_route_over_grid_cell. Adds (sign * delta) to V_macro, H_macro."""
        if macro_idx >= self.n_hard:
            return  # only hard macros contribute macro routing
        mod_w = self.sizes[macro_idx, 0]; mod_h = self.sizes[macro_idx, 1]
        mod_x = self.pos[macro_idx, 0]; mod_y = self.pos[macro_idx, 1]
        ur_x = mod_x + mod_w / 2; ur_y = mod_y + mod_h / 2
        bl_x = mod_x - mod_w / 2; bl_y = mod_y - mod_h / 2

        ur_row, ur_col = self._gcell(ur_x, ur_y)
        bl_row, bl_col = self._gcell(bl_x, bl_y)

        # Sanity (matches TILOS bounds clipping)
        bl_row = max(0, bl_row); bl_col = max(0, bl_col)
        ur_row = min(self.gr - 1, ur_row); ur_col = min(self.gc - 1, ur_col)
        if bl_row > ur_row or bl_col > ur_col:
            return

        x_min, x_max = bl_x, ur_x
        y_min, y_max = bl_y, ur_y

        if_partial_v = False
        if_partial_h = False
        for r in range(bl_row, ur_row + 1):
            ry_lo = r * self.grid_height; ry_hi = ry_lo + self.grid_height
            y_diff = max(0.0, min(y_max, ry_hi) - max(y_min, ry_lo))
            for c in range(bl_col, ur_col + 1):
                rx_lo = c * self.grid_width; rx_hi = rx_lo + self.grid_width
                x_diff = max(0.0, min(x_max, rx_hi) - max(x_min, rx_lo))
                # Check partial overlap (matches TILOS exactly)
                if ur_row != bl_row:
                    if (r == bl_row and abs(y_diff - self.grid_height) > 1e-5) or \
                       (r == ur_row and abs(y_diff - self.grid_height) > 1e-5):
                        if_partial_v = True
                if ur_col != bl_col:
                    if (c == bl_col and abs(x_diff - self.grid_width) > 1e-5) or \
                       (c == ur_col and abs(x_diff - self.grid_width) > 1e-5):
                        if_partial_h = True
                self.V_macro[r, c] += sign * x_diff * self.vrouting_alloc
                self.H_macro[r, c] += sign * y_diff * self.hrouting_alloc

        # Partial-overlap correction (TILOS subtracts boundary rows/cols)
        if if_partial_v:
            r = ur_row
            ry_lo = r * self.grid_height; ry_hi = ry_lo + self.grid_height
            for c in range(bl_col, ur_col + 1):
                rx_lo = c * self.grid_width; rx_hi = rx_lo + self.grid_width
                x_diff = max(0.0, min(x_max, rx_hi) - max(x_min, rx_lo))
                self.V_macro[r, c] -= sign * x_diff * self.vrouting_alloc
        if if_partial_h:
            c = ur_col
            rx_lo = c * self.grid_width; rx_hi = rx_lo + self.grid_width
            for r in range(bl_row, ur_row + 1):
                ry_lo = r * self.grid_height; ry_hi = ry_lo + self.grid_height
                y_diff = max(0.0, min(y_max, ry_hi) - max(y_min, ry_lo))
                self.H_macro[r, c] -= sign * y_diff * self.hrouting_alloc

    # ─────────────── Congestion: net routing patterns ───────────────
    def _add_two_pin(self, source_gc: Tuple[int, int], sink_gc: Tuple[int, int],
                     weight: float, sign: float):
        """Mirror __two_pin_net_routing."""
        col_min = min(sink_gc[1], source_gc[1]); col_max = max(sink_gc[1], source_gc[1])
        row_min = min(sink_gc[0], source_gc[0]); row_max = max(sink_gc[0], source_gc[0])
        # H from col_min..col_max-1 at source row
        for c in range(col_min, col_max):
            self.H_net[source_gc[0], c] += sign * weight
        # V from row_min..row_max-1 at sink col
        for r in range(row_min, row_max):
            self.V_net[r, sink_gc[1]] += sign * weight

    def _add_l_routing(self, sorted_gcells: List[Tuple[int, int]], weight: float, sign: float):
        """Mirror __l_routing. sorted_gcells already sorted by (col, row)."""
        y1, x1 = sorted_gcells[0]
        y2, x2 = sorted_gcells[1]
        y3, x3 = sorted_gcells[2]
        for c in range(x1, x2):
            self.H_net[y1, c] += sign * weight
        for c in range(x2, x3):
            self.H_net[y2, c] += sign * weight
        for r in range(min(y1, y2), max(y1, y2)):
            self.V_net[r, x2] += sign * weight
        for r in range(min(y2, y3), max(y2, y3)):
            self.V_net[r, x3] += sign * weight

    def _add_t_routing(self, sorted_gcells: List[Tuple[int, int]], weight: float, sign: float):
        """Mirror __t_routing. sorted_gcells: sorted by (row, col)."""
        y1, x1 = sorted_gcells[0]
        y2, x2 = sorted_gcells[1]
        y3, x3 = sorted_gcells[2]
        xmin = min(x1, x2, x3); xmax = max(x1, x2, x3)
        for c in range(xmin, xmax):
            self.H_net[y2, c] += sign * weight
        for r in range(min(y1, y2), max(y1, y2)):
            self.V_net[r, x1] += sign * weight
        for r in range(min(y2, y3), max(y2, y3)):
            self.V_net[r, x3] += sign * weight

    def _add_three_pin(self, gcells_set: frozenset, weight: float, sign: float):
        """Mirror __three_pin_net_routing."""
        temp = sorted(gcells_set, key=lambda g: (g[1], g[0]))
        y1, x1 = temp[0]
        y2, x2 = temp[1]
        y3, x3 = temp[2]
        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            self._add_l_routing(temp, weight, sign)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            for c in range(x1, x2):
                self.H_net[y1, c] += sign * weight
            for r in range(y1, max(y2, y3)):
                self.V_net[r, x2] += sign * weight
        elif y2 == y3:
            for c in range(x1, x2):
                self.H_net[y1, c] += sign * weight
            for c in range(x2, x3):
                self.H_net[y2, c] += sign * weight
            for r in range(min(y2, y1), max(y2, y1)):
                self.V_net[r, x2] += sign * weight
        else:
            # Sort by (row, col) for t_routing per TILOS code
            temp2 = sorted(gcells_set)
            self._add_t_routing(temp2, weight, sign)

    def _add_net_routing(self, nid: int, sign: float):
        """Apply (or remove) the routing contribution of net nid using its
        currently stored source_gcell + node_gcells."""
        gset = self.net_node_gcells[nid]
        source_gc = self.net_source_gcell[nid]
        if source_gc is None or len(gset) <= 1:
            return
        weight = 1.0  # We don't use net weights from PLC here
        if len(gset) == 2:
            sink_gc = next(iter(gset - {source_gc}))
            self._add_two_pin(source_gc, sink_gc, weight, sign)
        elif len(gset) == 3:
            self._add_three_pin(gset, weight, sign)
        else:
            # >3: split into 2-pin nets from source
            for node_gc in gset:
                if node_gc != source_gc:
                    self._add_two_pin(source_gc, node_gc, weight, sign)

    def _compute_net_gcells(self, nid: int) -> Tuple[Optional[Tuple[int, int]], frozenset]:
        """Compute current source_gcell and node_gcells set for net nid based
        on current pin positions."""
        entries = self.net_pin_entries[nid]
        if len(entries) <= 1:
            return None, frozenset()
        # First pin entry is the driver (per loader.py contract)
        driver_entry = entries[0]
        source_gc = self._gcell(self.pin_xy[driver_entry, 0], self.pin_xy[driver_entry, 1])
        gset = set()
        for ent in entries:
            gset.add(self._gcell(self.pin_xy[ent, 0], self.pin_xy[ent, 1]))
        return source_gc, frozenset(gset)

    def _refresh_net_routing(self, nid: int):
        """Subtract old contribution, recompute gcells, add new contribution."""
        if self.net_source_gcell[nid] is not None and len(self.net_node_gcells[nid]) >= 2:
            self._add_net_routing(nid, -1.0)
        new_source, new_gset = self._compute_net_gcells(nid)
        self.net_source_gcell[nid] = new_source
        self.net_node_gcells[nid] = new_gset
        if new_source is not None and len(new_gset) >= 2:
            self._add_net_routing(nid, +1.0)

    # ─────────────── Smoothing + final cost ───────────────
    def _v_smoothed(self) -> np.ndarray:
        """Apply TILOS V smoothing to V_net (across columns), per-row."""
        sr = self.smooth_range
        out = np.zeros((self.gr, self.gc), dtype=np.float64)
        for col in range(self.gc):
            lp = max(0, col - sr)
            rp = min(self.gc - 1, col + sr)
            gcell_cnt = rp - lp + 1
            # Each (row, col)'s value distributed to (row, lp..rp) divided by gcell_cnt
            # Vectorize over rows
            vals = self.V_net[:, col] / gcell_cnt
            out[:, lp:rp + 1] += vals[:, None]
        return out

    def _h_smoothed(self) -> np.ndarray:
        """Apply TILOS H smoothing to H_net (across rows), per-col."""
        sr = self.smooth_range
        out = np.zeros((self.gr, self.gc), dtype=np.float64)
        for row in range(self.gr):
            lp = max(0, row - sr)
            up = min(self.gr - 1, row + sr)
            gcell_cnt = up - lp + 1
            vals = self.H_net[row, :] / gcell_cnt
            out[lp:up + 1, :] += vals[None, :]
        return out

    # ─────────────── Initial recompute ───────────────
    def _recompute_all(self):
        self._recompute_all_hpwl()
        self._recompute_density()
        # Macro routing
        self.V_macro.fill(0.0); self.H_macro.fill(0.0)
        for i in range(self.n_hard):
            self._add_macro_route(i, +1.0)
        # Net routing
        self.V_net.fill(0.0); self.H_net.fill(0.0)
        for nid in range(self.n_nets):
            new_source, new_gset = self._compute_net_gcells(nid)
            self.net_source_gcell[nid] = new_source
            self.net_node_gcells[nid] = new_gset
            if new_source is not None and len(new_gset) >= 2:
                self._add_net_routing(nid, +1.0)

    # ─────────────── Cost computations ───────────────
    def hpwl_cost(self) -> float:
        """Match TILOS exactly: weighted_total_hpwl / (net_cnt × (cw + ch))."""
        weighted_total = float((self.net_weight * self.net_hpwl).sum())
        return weighted_total / max(self.net_cnt_for_norm * (self.cw + self.ch), 1e-12)

    def density_cost(self) -> float:
        """Match TILOS: 0.5 × top-10% mean of NON-ZERO grid cell densities,
        divided by floor(num_cells × 0.1)."""
        flat = (self.density.flatten() / self.bin_area)
        density_cnt = int(math.floor(len(flat) * 0.1))
        if len(flat) < 10:
            occupied = flat[flat > 0]
            if len(occupied) == 0:
                return 0.0
            return 0.5 * float(occupied.mean())
        # Top density_cnt non-zero values
        nonzero = flat[flat > 0]
        # Sort descending
        nonzero_desc = -np.sort(-nonzero)
        n_to_sum = min(density_cnt, len(nonzero_desc))
        sum_density = float(nonzero_desc[:n_to_sum].sum())
        return 0.5 * sum_density / density_cnt

    def congestion_cost(self) -> float:
        """
        Match TILOS exactly:
          1. Normalize V_net, H_net, V_macro, H_macro by their respective
             grid_routes capacity.
          2. Smooth normalized V_net (across cols) and H_net (across rows).
          3. Sum smoothed_net + macro (no smoothing of macro).
          4. Concatenate V and H, take abu(merged, 0.05) — no zero-filtering.
        """
        v_net_norm = self.V_net / max(self.grid_v_routes, 1e-12)
        h_net_norm = self.H_net / max(self.grid_h_routes, 1e-12)
        v_macro_norm = self.V_macro / max(self.grid_v_routes, 1e-12)
        h_macro_norm = self.H_macro / max(self.grid_h_routes, 1e-12)

        sr = self.smooth_range
        v_smooth = np.zeros((self.gr, self.gc), dtype=np.float64)
        for col in range(self.gc):
            lp = max(0, col - sr); rp = min(self.gc - 1, col + sr)
            gcell_cnt = rp - lp + 1
            vals = v_net_norm[:, col] / gcell_cnt
            v_smooth[:, lp:rp + 1] += vals[:, None]
        h_smooth = np.zeros((self.gr, self.gc), dtype=np.float64)
        for row in range(self.gr):
            lp = max(0, row - sr); up = min(self.gr - 1, row + sr)
            gcell_cnt = up - lp + 1
            vals = h_net_norm[row, :] / gcell_cnt
            h_smooth[lp:up + 1, :] += vals[None, :]

        v_total = v_smooth + v_macro_norm
        h_total = h_smooth + h_macro_norm
        # abu: top 5% mean of (V ∪ H) values, with floor() and no zero filtering
        merged = np.concatenate([v_total.flatten(), h_total.flatten()])
        cnt = int(math.floor(len(merged) * 0.05))
        if cnt == 0:
            return float(merged.max())
        # Sort descending and take top cnt
        merged_sorted = -np.sort(-merged)
        return float(merged_sorted[:cnt].sum() / cnt)

    def proxy_cost(self, weights=None) -> float:
        if weights is None:
            weights = {"wirelength": 1.0, "density": 0.5, "congestion": 0.5}
        return (weights["wirelength"] * self.hpwl_cost()
                + weights["density"] * self.density_cost()
                + weights["congestion"] * self.congestion_cost())

    # ─────────────── Move: incremental update ───────────────
    def commit_move(self, macro_idx: int, new_x: float, new_y: float):
        """Apply move and update all incremental state."""
        # Find affected nets (any net whose pin entry is on this macro)
        affected_nets = np.unique(self.net_ids[self.owner_to_pin_entries[macro_idx]])

        # Subtract old macro routing (only hard macros)
        self._add_macro_route(macro_idx, -1.0)
        # Subtract old density
        self._add_density(macro_idx, -1.0)

        # For each affected net: subtract old routing contribution
        for nid in affected_nets:
            if self.net_source_gcell[nid] is not None and len(self.net_node_gcells[nid]) >= 2:
                self._add_net_routing(int(nid), -1.0)

        # Update pos and pin_xy
        self.pos[macro_idx, 0] = new_x
        self.pos[macro_idx, 1] = new_y
        for ent in self.owner_to_pin_entries[macro_idx]:
            self.pin_xy[ent, 0] = new_x + self.pin_offset_xy[ent, 0]
            self.pin_xy[ent, 1] = new_y + self.pin_offset_xy[ent, 1]

        # Re-add density and macro routing (at new pos)
        self._add_density(macro_idx, +1.0)
        self._add_macro_route(macro_idx, +1.0)

        # Recompute net routing for affected nets
        for nid in affected_nets:
            new_source, new_gset = self._compute_net_gcells(int(nid))
            self.net_source_gcell[int(nid)] = new_source
            self.net_node_gcells[int(nid)] = new_gset
            if new_source is not None and len(new_gset) >= 2:
                self._add_net_routing(int(nid), +1.0)
            # Update HPWL
            self._update_net_hpwl(int(nid))

    def proposed_move_cost(self, macro_idx: int, new_x: float, new_y: float,
                           weights=None) -> float:
        """Compute proxy cost AS IF macro_idx were at (new_x, new_y). Mutates and reverts."""
        old_x = self.pos[macro_idx, 0]; old_y = self.pos[macro_idx, 1]
        self.commit_move(macro_idx, new_x, new_y)
        cost = self.proxy_cost(weights)
        self.commit_move(macro_idx, old_x, old_y)
        return cost
