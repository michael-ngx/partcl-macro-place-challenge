"""
DREAMPlace wrapper — converts a `Benchmark` to Bookshelf format, runs
real DREAMPlace, and returns macro positions in benchmark index order.

Drop-in replacement for v14's `_eplace_global` in v16/placer.py:

    from .dreamplace_wrapper import DreamPlaceWrapper, make_eplace_global_fn
    dp = DreamPlaceWrapper(dreamplace_dir="/opt/DREAMPlace", use_gpu=True)
    _eplace_global = make_eplace_global_fn(dp)

Then the rest of v14's pipeline (legalize + soft-Jacobi + real-proxy CD)
runs unchanged on top of DREAMPlace's output.

Bookshelf format reference:
    https://vlsicad.eecs.umich.edu/BK/PDtools/UCLAdescription.html
    External: external/MacroPlacement/CodeElements/FormatTranslators/

Tested against:
    - DREAMPlace v3+ (limbo018/DREAMPlace, github head as of 2025)
    - CUDA build on RTX 6000 Ada or CPU build on x86_64 Linux

Local validation:
    `test_bookshelf_roundtrip.py` validates that
    Benchmark → Bookshelf → re-parse matches every macro position
    and net topology to <1e-6 tolerance.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ───────────────────────── Bookshelf format writer ───────────────────────── #
class BookshelfWriter:
    """
    Convert a `Benchmark` to Bookshelf format (UCLA placement format v1.0).

    Files written under <out_dir>/<design_name>.{aux,nodes,nets,pl,scl,wts}.

    Node naming:
      hard macro i (0..n_hard-1)              -> "hard_<i>"
      soft macro i (n_hard..n_macros-1)       -> "soft_<i>"
      port i      (n_macros..n_macros+n_ports-1) -> "port_<i>"

    Hard macros are nodes (movable) or terminals (fixed). Soft macros are
    nodes (movable, with their full size). Ports are terminals (fixed,
    zero size).

    Pin offsets in .nets are stored relative to the node's CENTER as
    Bookshelf expects. Hard macro pins use their `macro_pin_offsets`;
    soft macro and port pins are at offset (0, 0) (single pin per net per
    soft/port owner — same convention as our Benchmark).
    """

    def __init__(self, benchmark: Benchmark, design_name: str = "design"):
        self.b = benchmark
        self.design_name = design_name

        self.n_hard = benchmark.num_hard_macros
        self.n_macros = benchmark.num_macros
        self.n_ports = benchmark.port_positions.shape[0]
        self.n_nodes_total = self.n_macros + self.n_ports

        # Per-owner indexing. Owner 0..n_hard-1 = hard, n_hard..n_macros-1 =
        # soft, n_macros..n_macros+n_ports-1 = ports.
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)

    # ─── Node naming ─── #
    def _node_name(self, owner_idx: int) -> str:
        if owner_idx < self.n_hard:
            return f"hard_{owner_idx}"
        if owner_idx < self.n_macros:
            return f"soft_{owner_idx - self.n_hard}"
        return f"port_{owner_idx - self.n_macros}"

    def _is_terminal(self, owner_idx: int) -> bool:
        """Bookshelf 'terminal' = fixed (not movable). Ports always fixed.
        Hard macros are terminal iff their fixed flag is set."""
        if owner_idx >= self.n_macros:
            return True  # port
        if owner_idx < self.n_hard:
            return bool(self.b.macro_fixed[owner_idx].item())
        # Soft macros: movable in our pipeline (we let DP move them)
        return False

    def _node_size(self, owner_idx: int) -> Tuple[float, float]:
        if owner_idx >= self.n_macros:
            # Port: zero-area terminal. Bookshelf accepts width=height=0.
            return 0.0, 0.0
        w, h = self.b.macro_sizes[owner_idx].tolist()
        return float(w), float(h)

    def _node_position(self, owner_idx: int) -> Tuple[float, float]:
        """Returns (x_lower_left, y_lower_left) per Bookshelf convention.
        Our Benchmark stores CENTER positions, so convert."""
        if owner_idx < self.n_macros:
            cx, cy = self.b.macro_positions[owner_idx].tolist()
            w, h = self._node_size(owner_idx)
            return float(cx - w / 2), float(cy - h / 2)
        # Port at port_positions[owner - n_macros]
        port_idx = owner_idx - self.n_macros
        x, y = self.b.port_positions[port_idx].tolist()
        return float(x), float(y)

    # ─── File writers ─── #
    def write(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_aux(out_dir)
        self._write_nodes(out_dir)
        self._write_nets(out_dir)
        self._write_pl(out_dir)
        self._write_scl(out_dir)
        self._write_wts(out_dir)

    def _write_aux(self, out_dir: Path):
        aux = out_dir / f"{self.design_name}.aux"
        aux.write_text(
            f"RowBasedPlacement : {self.design_name}.nodes "
            f"{self.design_name}.nets {self.design_name}.wts "
            f"{self.design_name}.pl {self.design_name}.scl\n"
        )

    def _write_nodes(self, out_dir: Path):
        path = out_dir / f"{self.design_name}.nodes"
        # Count terminals (fixed nodes + ports)
        n_terminals = 0
        for owner in range(self.n_nodes_total):
            if self._is_terminal(owner):
                n_terminals += 1

        lines = [
            "UCLA nodes 1.0",
            "",
            f"NumNodes : {self.n_nodes_total}",
            f"NumTerminals : {n_terminals}",
            "",
        ]
        # Convention: nodes first, terminals last (DREAMPlace expects this)
        for owner in range(self.n_nodes_total):
            if not self._is_terminal(owner):
                w, h = self._node_size(owner)
                lines.append(f"\t{self._node_name(owner)}\t{w:.6f}\t{h:.6f}")
        for owner in range(self.n_nodes_total):
            if self._is_terminal(owner):
                w, h = self._node_size(owner)
                lines.append(f"\t{self._node_name(owner)}\t{w:.6f}\t{h:.6f}\tterminal")
        path.write_text("\n".join(lines) + "\n")

    def _write_pl(self, out_dir: Path):
        path = out_dir / f"{self.design_name}.pl"
        lines = ["UCLA pl 1.0", ""]
        for owner in range(self.n_nodes_total):
            x_ll, y_ll = self._node_position(owner)
            tag = " /FIXED" if self._is_terminal(owner) else ""
            lines.append(f"\t{self._node_name(owner)}\t{x_ll:.6f}\t{y_ll:.6f}\t: N{tag}")
        path.write_text("\n".join(lines) + "\n")

    def _write_nets(self, out_dir: Path):
        path = out_dir / f"{self.design_name}.nets"
        n_pins_total = sum(p.shape[0] for p in self.b.net_pin_nodes)

        # Pre-build pin offset table: pin_offset_lookup[(owner, pin_idx)] -> (dx, dy)
        # For hard macros: from macro_pin_offsets
        # For soft macros / ports: (0, 0)
        def pin_offset(owner: int, pin_idx: int) -> Tuple[float, float]:
            if owner < self.n_hard:
                offsets = self.b.macro_pin_offsets[owner]
                if offsets.shape[0] == 0:
                    return 0.0, 0.0
                pin_idx = min(pin_idx, offsets.shape[0] - 1)
                dx, dy = offsets[pin_idx].tolist()
                return float(dx), float(dy)
            return 0.0, 0.0

        lines = [
            "UCLA nets 1.0",
            "",
            f"NumNets : {self.b.num_nets}",
            f"NumPins : {n_pins_total}",
            "",
        ]
        for nid, pins in enumerate(self.b.net_pin_nodes):
            arr = pins.tolist()
            lines.append(f"NetDegree : {len(arr)}\tnet_{nid}")
            for owner_idx, pin_idx in arr:
                owner = int(owner_idx)
                pidx = int(pin_idx)
                dx, dy = pin_offset(owner, pidx)
                # Pin direction "I" (input) — DREAMPlace ignores it for HPWL
                lines.append(
                    f"\t{self._node_name(owner)}\tI\t:\t{dx:.6f}\t{dy:.6f}"
                )
        path.write_text("\n".join(lines) + "\n")

    def _write_wts(self, out_dir: Path):
        path = out_dir / f"{self.design_name}.wts"
        lines = ["UCLA wts 1.0", ""]
        for nid in range(self.b.num_nets):
            w = float(self.b.net_weights[nid].item()) if nid < self.b.net_weights.shape[0] else 1.0
            lines.append(f"net_{nid}\t{w:.4f}")
        path.write_text("\n".join(lines) + "\n")

    def _write_scl(self, out_dir: Path):
        """
        SCL = site rows. Bookshelf needs at least one row spanning the canvas.
        We use a single coresite of 1×1 μm; rows tiled vertically.

        DREAMPlace will override row geometry from this; what matters is the
        canvas bounding box (xl, yl, xh, yh) is implied by the row coverage.
        """
        path = out_dir / f"{self.design_name}.scl"
        site_w = 1.0
        site_h = 1.0
        n_rows = max(int(math.ceil(self.ch / site_h)), 1)
        n_cols_per_row = max(int(math.ceil(self.cw / site_w)), 1)
        lines = ["UCLA scl 1.0", "", f"NumRows : {n_rows}", ""]
        for r in range(n_rows):
            y_lo = r * site_h
            lines.append("CoreRow Horizontal")
            lines.append(f"  Coordinate    :   {y_lo:.4f}")
            lines.append(f"  Height        :   {site_h:.4f}")
            lines.append(f"  Sitewidth     :   {site_w:.4f}")
            lines.append("  Sitespacing   :   1")
            lines.append("  Siteorient    :   1")
            lines.append("  Sitesymmetry  :   1")
            lines.append(f"  SubrowOrigin  :   0  NumSites  :  {n_cols_per_row}")
            lines.append("End")
        path.write_text("\n".join(lines) + "\n")


# ───────────────────────── Bookshelf .pl reader ──────────────────────────── #
def parse_bookshelf_pl(pl_path: Path) -> Dict[str, Tuple[float, float]]:
    """
    Parse a Bookshelf .pl file and return {node_name: (x_ll, y_ll)}.

    Format (after header):
        node_name x y : ORIENTATION [ /FIXED ]
    """
    out: Dict[str, Tuple[float, float]] = {}
    text = pl_path.read_text()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("UCLA"):
            continue
        # Tokens: name x y : orientation [/FIXED]
        toks = s.replace("\t", " ").split()
        if len(toks) < 3:
            continue
        name = toks[0]
        try:
            x = float(toks[1])
            y = float(toks[2])
        except ValueError:
            continue
        out[name] = (x, y)
    return out


# ───────────────────────── DREAMPlace runner ─────────────────────────────── #
class DreamPlaceWrapper:
    """
    Run real DREAMPlace as the global-placement step.

    Usage:
        dp = DreamPlaceWrapper(dreamplace_dir="/opt/DREAMPlace", use_gpu=True)
        full_pos = dp.run(benchmark, target_density=0.85)
        # full_pos is a [num_macros, 2] CPU torch.Tensor of CENTER positions.

    The wrapper writes a temporary Bookshelf design + JSON params, invokes
    DREAMPlace's `Placer.py` as a subprocess, parses the output .pl file,
    and converts back to (num_macros, 2) tensor in the benchmark's macro
    index order.

    DREAMPlace's legalization and detailed placement are turned OFF; we
    rely on the host pipeline's legalize + CD + LNS instead.
    """

    def __init__(
        self,
        dreamplace_dir: str | None = None,
        use_gpu: bool = True,
        keep_tempdir: bool = False,
    ):
        if dreamplace_dir is None:
            dreamplace_dir = os.environ.get("DREAMPLACE_DIR", "/opt/DREAMPlace")
        self.dp_dir = Path(dreamplace_dir).resolve()
        if not self.dp_dir.exists():
            raise RuntimeError(
                f"DREAMPlace not found at {self.dp_dir}. "
                f"Set DREAMPLACE_DIR env var or pass dreamplace_dir=..."
            )
        if not (self.dp_dir / "dreamplace" / "Placer.py").exists():
            raise RuntimeError(
                f"{self.dp_dir} does not look like a DREAMPlace install "
                f"(missing dreamplace/Placer.py)."
            )
        self.use_gpu = use_gpu
        self.keep_tempdir = keep_tempdir

    def run(
        self,
        benchmark: Benchmark,
        target_density: float = 0.85,
        num_bins_x: int = 64,
        num_bins_y: int = 64,
        gp_iterations: int = 1000,
        learning_rate: float = 0.01,
        gamma: float = 4.0,
        density_weight: float = 8.0e-5,
        macro_halo_x: float = 0.0,
        macro_halo_y: float = 0.0,
        random_seed: int = 42,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Run DREAMPlace and return [num_macros, 2] tensor of CENTER positions.

        Args correspond to standard DREAMPlace JSON params (see DREAMPlace's
        params.json template). Defaults are tuned for IBM ICCAD04 benchmarks.

        Hyperparameter grid that produced v13's gains can be applied here:
            target_density ∈ {0.70, 0.85, 0.95}
            density_weight ∈ {3e-5, 8e-5, 2e-4}
            gamma          ∈ {2.0, 4.0, 8.0}
        Sweep externally (DREAMTuna/Optuna style).
        """
        ctx = tempfile.TemporaryDirectory()
        tmp = Path(ctx.name)
        try:
            design_dir = tmp / "design"
            BookshelfWriter(benchmark, design_name="design").write(design_dir)

            params_path = tmp / "params.json"
            result_dir = tmp / "results"
            self._write_params(
                design_dir, params_path, result_dir,
                target_density=target_density,
                num_bins_x=num_bins_x, num_bins_y=num_bins_y,
                gp_iterations=gp_iterations,
                learning_rate=learning_rate,
                gamma=gamma,
                density_weight=density_weight,
                macro_halo_x=macro_halo_x,
                macro_halo_y=macro_halo_y,
                random_seed=random_seed,
            )

            self._invoke(params_path, verbose=verbose)

            # DREAMPlace writes <design>.gp.pl in result_dir/<design_name>/
            pl_path = result_dir / "design" / "design.gp.pl"
            if not pl_path.exists():
                # Fallback: search for any .pl file under result_dir
                pls = list(result_dir.rglob("*.pl"))
                if not pls:
                    raise RuntimeError(
                        f"DREAMPlace produced no .pl file in {result_dir}"
                    )
                pl_path = pls[-1]
            return self._parse_back(pl_path, benchmark)
        finally:
            if not self.keep_tempdir:
                ctx.cleanup()

    def _write_params(
        self, design_dir: Path, out_path: Path, result_dir: Path,
        target_density: float, num_bins_x: int, num_bins_y: int,
        gp_iterations: int, learning_rate: float, gamma: float,
        density_weight: float, macro_halo_x: float, macro_halo_y: float,
        random_seed: int,
    ):
        params = {
            "aux_input": str(design_dir / "design.aux"),
            "lef_input": [],
            "def_input": "",
            "verilog_input": "",
            "gpu": int(bool(self.use_gpu)),
            "num_bins_x": num_bins_x,
            "num_bins_y": num_bins_y,
            "global_place_stages": [
                {
                    "num_bins_x": num_bins_x,
                    "num_bins_y": num_bins_y,
                    "iteration": gp_iterations,
                    "learning_rate": learning_rate,
                    "wirelength": "weighted_average",
                    "optimizer": "nesterov",
                    "Llambda_density_weight_iteration": 1,
                    "Lsub_iteration": 1,
                }
            ],
            "target_density": target_density,
            "density_weight": density_weight,
            "gamma": gamma,
            "RePlAce_ref_hpwl": 350000,
            "RePlAce_LOWER_PCOF": 0.95,
            "RePlAce_UPPER_PCOF": 1.05,
            "stop_overflow": 0.1,
            "use_bb": 1,
            "macro_place_flag": 1,
            "macro_halo_x": macro_halo_x,
            "macro_halo_y": macro_halo_y,
            "global_place_flag": 1,
            "legalize_flag": 0,           # we do our own
            "detailed_place_flag": 0,     # we do our own
            "routability_opt_flag": 0,
            "timing_opt_flag": 0,
            "random_seed": random_seed,
            "result_dir": str(result_dir),
            "plot_flag": 0,
            "num_threads": 8,
        }
        out_path.write_text(json.dumps(params, indent=2))

    def _invoke(self, params_path: Path, verbose: bool):
        cmd = [
            sys.executable,
            str(self.dp_dir / "dreamplace" / "Placer.py"),
            str(params_path),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(self.dp_dir) + os.pathsep + env.get("PYTHONPATH", "")
        )
        # OpenMP threads control
        env.setdefault("OMP_NUM_THREADS", "8")
        try:
            res = subprocess.run(
                cmd, env=env, check=True, capture_output=not verbose,
                cwd=str(self.dp_dir),
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else "(no stderr)"
            stdout = e.stdout.decode() if e.stdout else "(no stdout)"
            raise RuntimeError(
                f"DREAMPlace failed (exit {e.returncode}):\n"
                f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
            ) from e

    def _parse_back(self, pl_path: Path, benchmark: Benchmark) -> torch.Tensor:
        """Parse DREAMPlace's output .pl back into [num_macros, 2] CENTER positions."""
        name_to_pos = parse_bookshelf_pl(pl_path)
        n_macros = benchmark.num_macros
        out = benchmark.macro_positions.clone()  # default to current pos for missing
        for owner_idx in range(n_macros):
            if owner_idx < benchmark.num_hard_macros:
                name = f"hard_{owner_idx}"
            else:
                name = f"soft_{owner_idx - benchmark.num_hard_macros}"
            if name not in name_to_pos:
                continue
            x_ll, y_ll = name_to_pos[name]
            w, h = benchmark.macro_sizes[owner_idx].tolist()
            cx = x_ll + w / 2
            cy = y_ll + h / 2
            out[owner_idx, 0] = float(cx)
            out[owner_idx, 1] = float(cy)
        return out


# ───────────────────────── _eplace_global drop-in factory ────────────────── #
def make_eplace_global_fn(dp: DreamPlaceWrapper):
    """
    Return a function with the same signature as v14's `_eplace_global`,
    so the V14Placer pipeline can use real DREAMPlace transparently.

    Returned signature:
      _eplace_global(benchmark, device, n_iters=600, n_bins=64,
                     target_density=0.85, base_gamma_factor=4.0,
                     init_lr_frac=1e-3, add_filler=True, verbose=False,
                     net_weights=None, init_pos_real=None,
                     macro_halo_frac=0.0)
      -> (final_real_pos: torch.Tensor [num_macros, 2], state_dict)
    """

    def _eplace_global(
        benchmark, device,
        n_iters: int = 600,
        n_bins: int = 64,
        target_density: float = 0.85,
        base_gamma_factor: float = 4.0,
        init_lr_frac: float = 1e-3,
        add_filler: bool = True,
        verbose: bool = False,
        net_weights=None,           # ignored by DP path (could add via .wts)
        init_pos_real=None,          # ignored: DP starts fresh
        macro_halo_frac: float = 0.0,
    ):
        canvas_size = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        halo = macro_halo_frac * canvas_size

        # Apply per-net weight overrides via .wts is the cleanest way.
        # For the simple case we ignore net_weights here; if needed, the
        # caller can write a custom Bookshelf via BookshelfWriter and
        # dp.run pointing at its result.
        full_pos = dp.run(
            benchmark,
            target_density=target_density,
            num_bins_x=n_bins, num_bins_y=n_bins,
            gp_iterations=max(n_iters, 100),
            gamma=base_gamma_factor,
            macro_halo_x=halo, macro_halo_y=halo,
            random_seed=42,
            verbose=verbose,
        )
        return full_pos.cpu().float(), {}

    return _eplace_global
