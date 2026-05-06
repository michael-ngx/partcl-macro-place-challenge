"""
v1: Smart minimum-displacement legalization placer.

Empirical finding: the initial.plc placements (from RePlAce/NTUplace3 in the
TILOS academic flow) are very high quality — for ibm01, the unlegalized initial
placement scores proxy 1.04 (better than RePlAce's 0.998 baseline). Naive
legalization preserves most of that quality, while running gradient-descent
global placement on top of it actively destroys the layout.

Pipeline:
  1. Greedy minimum-displacement legalization of hard macros (spiral search,
     largest-first).
  2. Soft macros stay at their initial positions (touching them with FD made
     things much worse in our experiments — the initial soft-macro positions
     already match the initial hard-macro positions).
  3. (Optional, off by default) Differentiable global placement and/or soft-FD
     for experimentation.

Target: beat RePlAce baseline (1.4578 avg).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _load_plc(name: str):
    """Load a PlacementCost for soft-macro FD optimization (slow but accurate)."""
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


class V1Placer:
    """DREAMPlace-style analytical placer (PyTorch, CPU/GPU)."""

    def __init__(
        self,
        seed: int = 42,
        device: torch.device | None = None,
        gp_iters: int = 0,                # OFF by default — initial.plc is already good
        gp_lr_frac: float = 0.01,
        gamma_init_frac: float = 0.05,
        gamma_final_frac: float = 0.005,
        lambda_init: float = 1e-4,
        lambda_step: float = 1.04,
        target_density_factor: float = 1.05,
        n_bins: int = 32,
        do_soft_fd: bool = False,         # OFF by default — destroys quality with few steps
        fd_steps: tuple = (50, 50, 50),
    ):
        self.seed = seed
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gp_iters = gp_iters
        self.gp_lr_frac = gp_lr_frac
        self.gamma_init_frac = gamma_init_frac
        self.gamma_final_frac = gamma_final_frac
        self.lambda_init = lambda_init
        self.lambda_step = lambda_step
        self.target_density_factor = target_density_factor
        self.n_bins = n_bins
        self.do_soft_fd = do_soft_fd
        self.fd_steps = fd_steps

    # ───────────────────────── Main entry ───────────────────────── #
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        device = self.device
        n_macros = benchmark.num_macros
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes.to(device).float()
        movable_mask = benchmark.get_movable_mask().to(device)

        # ─── Build per-pin net data ─────────────────────────────── #
        port_pos = benchmark.port_positions.to(device).float()
        n_ports = port_pos.shape[0]
        n_owners = n_macros + n_ports

        max_pins = max(
            (p.shape[0] for p in benchmark.macro_pin_offsets), default=1
        )
        max_pins = max(max_pins, 1)
        owner_pin_offsets = torch.zeros(n_owners, max_pins, 2, device=device)
        for i, offsets in enumerate(benchmark.macro_pin_offsets):
            if offsets.shape[0] > 0:
                owner_pin_offsets[i, : offsets.shape[0]] = offsets.to(device)

        net_owner_list, net_pinidx_list, net_id_list = [], [], []
        for nid, pins in enumerate(benchmark.net_pin_nodes):
            arr = pins.numpy()
            for row in arr:
                net_owner_list.append(int(row[0]))
                net_pinidx_list.append(int(row[1]))
                net_id_list.append(nid)
        if len(net_owner_list) == 0:
            # Pathological case: no nets; nothing to optimize. Return initial.
            return benchmark.macro_positions.clone()

        net_owners = torch.tensor(net_owner_list, dtype=torch.long, device=device)
        net_pinidx = torch.tensor(net_pinidx_list, dtype=torch.long, device=device)
        net_ids = torch.tensor(net_id_list, dtype=torch.long, device=device)
        n_nets = len(benchmark.net_pin_nodes)

        pin_offset_xy = owner_pin_offsets[net_owners, net_pinidx]  # [P, 2]

        # ─── Compute target density from utilization ───────────── #
        total_macro_area = float((sizes[:, 0] * sizes[:, 1]).sum().item())
        canvas_area = cw * ch
        utilization = total_macro_area / canvas_area
        target_density = min(0.95, utilization * self.target_density_factor)
        # We compute density as area, so target_cap is per bin in area units
        bin_w = cw / self.n_bins
        bin_h = ch / self.n_bins
        target_cap = bin_w * bin_h * target_density

        # ─── Initial positions ──────────────────────────────────── #
        pos = benchmark.macro_positions.clone().to(device).float()

        if self.gp_iters > 0:
            # Small noise only when running GP (helps escape symmetry)
            noise = 0.005 * torch.randn_like(pos)
            noise[~movable_mask] = 0
            pos = pos + noise
        pos.requires_grad_(self.gp_iters > 0)

        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2

        fixed_pos = benchmark.macro_positions.to(device).float().clone()

        # Pre-compute bin edges for density
        bin_x_lo = torch.arange(self.n_bins, device=device).float() * bin_w
        bin_x_hi = bin_x_lo + bin_w
        bin_y_lo = torch.arange(self.n_bins, device=device).float() * bin_h
        bin_y_hi = bin_y_lo + bin_h

        # ─── Stage 1: Differentiable global placement (optional) ─ #
        canvas_size = max(cw, ch)
        if self.gp_iters > 0:
            opt = torch.optim.Adam([pos], lr=self.gp_lr_frac * canvas_size)

        for it in range(self.gp_iters):
            opt.zero_grad()

            # Smooth γ schedule
            t = it / max(self.gp_iters - 1, 1)
            gamma = self.gamma_init_frac * (
                self.gamma_final_frac / self.gamma_init_frac
            ) ** t
            gamma_x = gamma * cw
            gamma_y = gamma * ch

            # Owner positions: stack movable+fixed macros + ports
            owner_pos = torch.cat([pos, port_pos], dim=0) if n_ports > 0 else pos
            pin_xy = owner_pos[net_owners] + pin_offset_xy

            wl_x = self._wa_hpwl_dim(pin_xy[:, 0], net_ids, n_nets, gamma_x)
            wl_y = self._wa_hpwl_dim(pin_xy[:, 1], net_ids, n_nets, gamma_y)
            # Normalize to keep wirelength loss O(1)
            wirelength = (wl_x.sum() + wl_y.sum()) / (n_nets * (cw + ch))

            density = self._density_grid(
                pos, sizes, bin_x_lo, bin_x_hi, bin_y_lo, bin_y_hi
            )
            overflow = (density - target_cap).clamp(min=0)
            density_loss = overflow.pow(2).sum() / (
                self.n_bins * self.n_bins * target_cap * target_cap + 1e-12
            )

            lam = self.lambda_init * (self.lambda_step ** it)
            loss = wirelength + lam * density_loss
            loss.backward()

            with torch.no_grad():
                pos.grad[~movable_mask] = 0.0
            opt.step()

            with torch.no_grad():
                # Restore fixed macros
                pos.data[~movable_mask] = fixed_pos[~movable_mask]
                # Project to canvas (centers)
                pos.data[:, 0].clamp_(min=half_w, max=cw - half_w)
                pos.data[:, 1].clamp_(min=half_h, max=ch - half_h)

        # ─── Stage 2: Hard-macro legalization ──────────────────── #
        with torch.no_grad():
            legal = self._legalize(
                pos.detach().cpu().numpy().astype(np.float64),
                movable_mask.cpu().numpy(),
                sizes.cpu().numpy().astype(np.float64),
                cw,
                ch,
                n_hard,
                fixed_pos.cpu().numpy().astype(np.float64),
            )

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.from_numpy(legal[:n_hard]).float()
        # Soft macros take the optimizer output for now (soft FD will refine)
        if benchmark.num_soft_macros > 0:
            soft_legal = pos.detach().cpu()[n_hard:].float()
            # Clamp to canvas
            soft_legal[:, 0].clamp_(
                min=sizes.cpu()[n_hard:, 0] / 2,
                max=cw - sizes.cpu()[n_hard:, 0] / 2,
            )
            soft_legal[:, 1].clamp_(
                min=sizes.cpu()[n_hard:, 1] / 2,
                max=ch - sizes.cpu()[n_hard:, 1] / 2,
            )
            full_pos[n_hard:] = soft_legal

        # ─── Stage 3: Soft-macro FD optimization ──────────────── #
        if self.do_soft_fd and benchmark.num_soft_macros > 0:
            plc = _load_plc(benchmark.name)
            if plc is not None:
                try:
                    for i, idx in enumerate(benchmark.hard_macro_indices):
                        x = float(full_pos[i, 0].item())
                        y = float(full_pos[i, 1].item())
                        plc.modules_w_pins[idx].set_pos(x, y)
                    plc.optimize_stdcells(
                        use_current_loc=False,
                        move_stdcells=True,
                        move_macros=False,
                        log_scale_conns=False,
                        use_sizes=False,
                        io_factor=1.0,
                        num_steps=list(self.fd_steps),
                        max_move_distance=[canvas_size / 100] * 3,
                        attract_factor=[100, 1.0e-3, 1.0e-5],
                        repel_factor=[0, 1.0e6, 1.0e7],
                    )
                    for i, idx in enumerate(benchmark.soft_macro_indices):
                        x, y = plc.modules_w_pins[idx].get_pos()
                        full_pos[n_hard + i, 0] = x
                        full_pos[n_hard + i, 1] = y
                except Exception as e:
                    print(f"  [V1] soft FD failed: {e}")

        return full_pos

    # ───────────────────────── Helpers ───────────────────────── #
    @staticmethod
    def _wa_hpwl_dim(
        x: torch.Tensor, net_ids: torch.Tensor, n_nets: int, gamma: float
    ) -> torch.Tensor:
        """
        Smooth HPWL using weighted-average wirelength model (1-D).

        WA_max = Σ x_i exp(x_i/γ) / Σ exp(x_i/γ)        ↑ as γ→0 → max
        WA_min = Σ x_i exp(-x_i/γ) / Σ exp(-x_i/γ)      ↑ as γ→0 → min

        Returns [n_nets] tensor of (WA_max - WA_min).
        """
        device = x.device
        big = 1e18

        # Per-net max for numerical stability (in units of x/γ)
        true_max = torch.full((n_nets,), -big, device=device)
        true_max.scatter_reduce_(0, net_ids, x / gamma, reduce="amax", include_self=True)
        true_min = torch.full((n_nets,), big, device=device)
        true_min.scatter_reduce_(0, net_ids, x / gamma, reduce="amin", include_self=True)
        # Use as constants for the soft-max/-min trick
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

    @staticmethod
    def _density_grid(
        pos: torch.Tensor,
        sizes: torch.Tensor,
        bin_x_lo: torch.Tensor,
        bin_x_hi: torch.Tensor,
        bin_y_lo: torch.Tensor,
        bin_y_hi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable density = exact macro/bin overlap area.
        Returns [n_bins_y, n_bins_x] density (area) tensor.
        """
        half_w = sizes[:, 0] / 2  # [N]
        half_h = sizes[:, 1] / 2  # [N]
        mx_lo = (pos[:, 0] - half_w).unsqueeze(1)  # [N, 1]
        mx_hi = (pos[:, 0] + half_w).unsqueeze(1)
        my_lo = (pos[:, 1] - half_h).unsqueeze(1)
        my_hi = (pos[:, 1] + half_h).unsqueeze(1)

        ox = (
            torch.minimum(mx_hi, bin_x_hi.unsqueeze(0))
            - torch.maximum(mx_lo, bin_x_lo.unsqueeze(0))
        ).clamp(min=0)  # [N, n_bins_x]
        oy = (
            torch.minimum(my_hi, bin_y_hi.unsqueeze(0))
            - torch.maximum(my_lo, bin_y_lo.unsqueeze(0))
        ).clamp(min=0)  # [N, n_bins_y]

        # Outer product [N, n_bins_y, n_bins_x] then sum over N
        density = (oy.unsqueeze(2) * ox.unsqueeze(1)).sum(dim=0)  # [n_y, n_x]
        return density

    @staticmethod
    def _legalize(
        pos: np.ndarray,
        movable: np.ndarray,
        sizes: np.ndarray,
        cw: float,
        ch: float,
        n_hard: int,
        fixed_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Greedy minimum-displacement legalization of hard macros.
        Order: largest macros first. Spiral search around target position.
        """
        pos = pos.copy()
        # Restore fixed macros
        for i in range(n_hard):
            if not movable[i]:
                pos[i] = fixed_pos[i]

        gap = 0.001
        # Pairwise separation thresholds (for the n_hard hard macros)
        sx = sizes[:n_hard, 0]
        sy = sizes[:n_hard, 1]
        sep_x_mat = (sx[:, None] + sx[None, :]) / 2 + gap
        sep_y_mat = (sy[:, None] + sy[None, :]) / 2 + gap

        half_w = sx / 2
        half_h = sy / 2

        order = sorted(range(n_hard), key=lambda i: -sx[i] * sy[i])
        placed = np.zeros(n_hard, dtype=bool)
        # Mark fixed macros placed
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
            # Clamp to canvas
            x0 = float(np.clip(pos[idx, 0], half_w[idx], cw - half_w[idx]))
            y0 = float(np.clip(pos[idx, 1], half_h[idx], ch - half_h[idx]))
            if not has_overlap(idx, x0, y0):
                pos[idx, 0], pos[idx, 1] = x0, y0
                placed[idx] = True
                continue
            # Spiral search
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
                if found_any and ring_found and r > 0:
                    # Already found something at this ring; outer rings strictly farther
                    break
            pos[idx, 0], pos[idx, 1] = best_x, best_y
            placed[idx] = True

        return pos
