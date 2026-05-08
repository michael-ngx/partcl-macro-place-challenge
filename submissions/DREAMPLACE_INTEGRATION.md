# DREAMPlace Integration Plan

When the VM with real DREAMPlace is ready, swap in real DREAMPlace as the
global placement (GP) step. The rest of our pipeline (legalize + soft-Jacobi
+ real-proxy CD + LNS) is plug-in compatible because it operates on
positions+sizes, not on GP internals.

## Current pipeline (v14)

```
Benchmark  →  ePlace global placement (my PyTorch ePlace)  →  legalize  →  CD  →  best
                ↑ swap this with real DREAMPlace
```

## Files we'd need to add

### 1. `submissions/v16/dreamplace_wrapper.py`

```python
"""Wrap real DREAMPlace as a drop-in for my _eplace_global function."""

import os, subprocess, tempfile, json
import numpy as np
import torch
from pathlib import Path

from macro_place.benchmark import Benchmark


class DreamPlaceWrapper:
    """
    Runs real DREAMPlace for global placement, returns (positions, info).

    Requires: DREAMPlace installed on the system (pip + manual build)
    or accessible via Docker (limbo018/dreamplace:cuda-latest).
    """

    def __init__(self, dreamplace_dir: str = "/opt/DREAMPlace",
                 use_gpu: bool = True,
                 target_density: float = 0.85,
                 num_bins_x: int = 64, num_bins_y: int = 64,
                 routability_opt_flag: bool = False):
        self.dp_dir = Path(dreamplace_dir)
        self.use_gpu = use_gpu
        self.target_density = target_density
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.routability_opt_flag = routability_opt_flag
        if not self.dp_dir.exists():
            raise RuntimeError(f"DREAMPlace not found at {dreamplace_dir}")

    def run(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Run real DREAMPlace and return [num_macros, 2] tensor of positions
        in benchmark's macro index order.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # 1. Convert protobuf benchmark to Bookshelf format
            self._write_bookshelf(benchmark, tmp / "design")
            # 2. Write DREAMPlace JSON params
            params_path = self._write_params(tmp / "design", benchmark, tmp / "params.json")
            # 3. Run DREAMPlace
            result_dir = tmp / "results"
            self._invoke_dreamplace(params_path, result_dir)
            # 4. Parse output .pl file back to a tensor
            return self._read_bookshelf_pl(result_dir / "design" / "design.gp.pl",
                                           benchmark)

    # ─── Bookshelf format converter (protobuf → Bookshelf) ─── #
    def _write_bookshelf(self, benchmark: Benchmark, out_dir: Path):
        """
        Write Bookshelf files (.aux, .nodes, .nets, .pl, .scl, .wts) from
        benchmark's protobuf data.

        Reference TILOS converter at:
        external/MacroPlacement/CodeElements/FormatTranslators/

        Spec for each file:
          - .aux: master file listing the others
          - .nodes: list of nodes with sizes; "terminal" keyword for fixed
          - .nets: list of nets with their pins (NetDegree, then pin lines)
          - .pl: initial placement (x y orientation /FIXED)
          - .scl: site row description (rows of placement sites)
          - .wts: net weights (default 1)

        IMPORTANT: pin offsets for hard macros must be transformed to
        Bookshelf's (relative to macro center + half_dim) convention.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        # ... (TODO when VM is ready)
        raise NotImplementedError("Bookshelf converter — to implement on VM")

    def _read_bookshelf_pl(self, pl_path: Path, benchmark: Benchmark) -> torch.Tensor:
        """Parse Bookshelf .pl format back to (num_macros, 2) tensor."""
        # ... (TODO)
        raise NotImplementedError

    def _write_params(self, design_dir: Path, benchmark: Benchmark,
                      out_path: Path) -> Path:
        """Write DREAMPlace params.json with our target density / bin grid."""
        params = {
            "aux_input": str(design_dir / "design.aux"),
            "lef_input": [],
            "def_input": "",
            "verilog_input": "",
            "gpu": int(self.use_gpu),
            "num_bins_x": self.num_bins_x,
            "num_bins_y": self.num_bins_y,
            "global_place_stages": [{
                "num_bins_x": self.num_bins_x,
                "num_bins_y": self.num_bins_y,
                "iteration": 1000,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
            }],
            "target_density": self.target_density,
            "density_weight": 8.0e-5,
            "gamma": 4.0,
            "RePlAce_ref_hpwl": 350000,
            "RePlAce_LOWER_PCOF": 0.95,
            "RePlAce_UPPER_PCOF": 1.05,
            "stop_overflow": 0.1,
            "use_bb": 1,
            "macro_place_flag": 1,
            "macro_halo_x": 0,
            "macro_halo_y": 0,
            "global_place_flag": 1,
            "legalize_flag": 0,  # WE handle legalization; turn off DREAMPlace's
            "detailed_place_flag": 0,
            "result_dir": str(out_path.parent / "results"),
            "routability_opt_flag": int(self.routability_opt_flag),
            "random_seed": 42,
        }
        out_path.write_text(json.dumps(params, indent=2))
        return out_path

    def _invoke_dreamplace(self, params_path: Path, result_dir: Path):
        """Run DREAMPlace and capture its output."""
        cmd = [
            "python", str(self.dp_dir / "dreamplace" / "Placer.py"),
            str(params_path),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.dp_dir}:{env.get('PYTHONPATH', '')}"
        subprocess.run(cmd, env=env, check=True, capture_output=True)


def make_dreamplace_eplace_global(dp_wrapper: DreamPlaceWrapper):
    """
    Returns a callable with the same signature as my _eplace_global,
    so V14Placer.place() can use real DREAMPlace transparently.
    """
    def _ep_global(benchmark, device, n_iters=600, n_bins=64,
                    target_density=0.85, **kwargs):
        # Override target_density per call (multi-td loop)
        dp_wrapper.target_density = target_density
        full_pos_t = dp_wrapper.run(benchmark)
        return full_pos_t.cpu(), {}
    return _ep_global
```

### 2. `submissions/v16/placer.py`

Same as v14, but at the top:

```python
USE_REAL_DREAMPLACE = os.environ.get("DREAMPLACE_DIR") is not None

if USE_REAL_DREAMPLACE:
    from .dreamplace_wrapper import DreamPlaceWrapper, make_dreamplace_eplace_global
    _DP = DreamPlaceWrapper(os.environ["DREAMPLACE_DIR"])
    _eplace_global = make_dreamplace_eplace_global(_DP)
# else: use my PyTorch _eplace_global (defined below, same as v14)
```

This keeps v14's pipeline intact and lets us swap GP at runtime via env var.

## Format conversion details

The challenge eval feeds `netlist.pb.txt` (Google's protobuf-text format).
DREAMPlace consumes Bookshelf or LEF/DEF. We need a converter.

The IBM ICCAD04 benchmarks **originally come in Bookshelf format** —
they're at `http://vlsicad.eecs.umich.edu/BK/ICCAD04bench/`. TILOS converted
them to protobuf for their flow. So one option is to **download the original
Bookshelf files** rather than write a converter.

Steps if we go that route:
1. `wget http://vlsicad.eecs.umich.edu/BK/ICCAD04bench/ICCAD04.tar.gz`
2. Extract `ibm01.aux`, `ibm01.nodes`, etc.
3. Use Bookshelf → DREAMPlace directly, skipping conversion
4. Need to map DREAMPlace's output node names back to benchmark's tensor
   indices (the protobuf and Bookshelf netlists have the same nodes but
   possibly different ordering)

Time estimate for this approach: 3-5 hours.

If we write our own converter:
- Reading protobuf: already done (loader.py uses plc_client_os.py)
- Writing Bookshelf: need to handle hard macros, soft macros (with pins
  at center), ports, hierarchical pin offsets
- Time estimate: 1-2 days

**Recommendation: download the original Bookshelf benchmarks.** Simpler
and matches what real DREAMPlace papers benchmark on.

## Tier-2 (NG45) integration

For NG45 designs, the workflow is:
1. NG45 designs are in LEF/DEF format already (in TILOS submodule)
2. DREAMPlace can consume LEF/DEF directly
3. After DREAMPlace, parse `<design>.gp.def` back to a torch tensor

## Hyperparameter sweep with real DREAMPlace

DREAMTuna (#3 leaderboard at 1.22) uses Optuna to sweep DREAMPlace params.
Once we have real DREAMPlace plugged in, we can:
1. Pick 5-10 hyperparameters: target_density, gamma, density_weight,
   learning_rate, num_bins, etc.
2. Run Bayesian optimization over them per benchmark
3. Cache results so subsequent runs reuse the best config

Time estimate: 1 day for the sweep harness, 2-3 days of search per benchmark.

## Expected impact

With v14's pipeline using real DREAMPlace:
- ePlace step: 30-100× faster (CUDA), better quality (validated operators)
- Expected score improvement: 5-10% (1.36 → 1.22-1.30 range)
- Combined with hyperparameter sweep (DREAMTuna-style): 1.20-1.25 range

That would put us in **top 5** on the leaderboard.

## Concrete handoff

When the VM is ready, the user can:
1. `pip install` DREAMPlace, set `DREAMPLACE_DIR`
2. Implement the `_write_bookshelf` and `_read_bookshelf_pl` methods in
   `dreamplace_wrapper.py` (or download original Bookshelf files)
3. Test: `python -c "from submissions.v16.dreamplace_wrapper import DreamPlaceWrapper; ..."`
4. Run: `uv run evaluate submissions/v16/placer.py --all` — same harness
5. Compare: should beat v14's 1.3629 average meaningfully
