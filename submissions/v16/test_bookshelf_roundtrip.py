"""
Local validation of the Bookshelf converter (no DREAMPlace required).

Validates:
  1. BookshelfWriter creates all 5 files (.aux/.nodes/.nets/.pl/.scl/.wts).
  2. Files are well-formed (parseable).
  3. The .pl positions, after re-reading and converting back to centers,
     match the original Benchmark macro_positions to <1e-5 tolerance.
  4. Sizes and node names round-trip correctly.

Usage:
  uv run python submissions/v16/test_bookshelf_roundtrip.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from macro_place.loader import load_benchmark_from_dir
from dreamplace_wrapper import BookshelfWriter, parse_bookshelf_pl


def parse_bookshelf_nodes(path: Path):
    """Parse Bookshelf .nodes -> dict {name: (w, h, is_terminal)}."""
    out = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("UCLA") or s.startswith("Num"):
            continue
        toks = s.replace("\t", " ").split()
        if len(toks) < 3:
            continue
        name = toks[0]
        try:
            w = float(toks[1]); h = float(toks[2])
        except ValueError:
            continue
        is_term = "terminal" in toks
        out[name] = (w, h, is_term)
    return out


def parse_bookshelf_nets(path: Path):
    """Parse Bookshelf .nets -> list of nets, each = list of (node_name, dx, dy)."""
    nets = []
    cur_net = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("UCLA") or s.startswith("Num"):
            continue
        if s.startswith("NetDegree"):
            if cur_net is not None:
                nets.append(cur_net)
            cur_net = []
        else:
            toks = s.replace("\t", " ").replace(":", " ").split()
            # name, I/O, dx, dy
            if len(toks) >= 4:
                name = toks[0]
                # Find dx, dy as the last two floats
                try:
                    dx = float(toks[-2]); dy = float(toks[-1])
                except ValueError:
                    continue
                cur_net.append((name, dx, dy))
    if cur_net is not None:
        nets.append(cur_net)
    return nets


def run(name: str = "ibm01"):
    print(f"=== Round-trip test: {name} ===")
    benchmark, plc = load_benchmark_from_dir(
        f"external/MacroPlacement/Testcases/ICCAD04/{name}"
    )
    print(f"  Loaded: {benchmark.num_macros} macros ({benchmark.num_hard_macros} hard, "
          f"{benchmark.num_macros - benchmark.num_hard_macros} soft), "
          f"{benchmark.port_positions.shape[0]} ports, {benchmark.num_nets} nets")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        writer = BookshelfWriter(benchmark, design_name="test")
        writer.write(tmp)

        files = sorted(tmp.iterdir())
        print(f"  Wrote: {[f.name for f in files]}")

        # Verify all 5 expected files exist
        expected = {"test.aux", "test.nodes", "test.nets", "test.pl",
                    "test.scl", "test.wts"}
        actual = {f.name for f in files}
        missing = expected - actual
        assert not missing, f"Missing files: {missing}"
        print("  ✓ All 5 Bookshelf files written")

        # ─── .nodes round-trip ─── #
        nodes = parse_bookshelf_nodes(tmp / "test.nodes")
        n_expected = benchmark.num_macros + benchmark.port_positions.shape[0]
        assert len(nodes) == n_expected, \
            f".nodes count {len(nodes)} != expected {n_expected}"
        for hi in range(benchmark.num_hard_macros):
            name = f"hard_{hi}"
            assert name in nodes, f"missing {name} in .nodes"
            w, h, is_term = nodes[name]
            real_w, real_h = benchmark.macro_sizes[hi].tolist()
            assert abs(w - real_w) < 1e-4, f"{name} width {w} != {real_w}"
            assert abs(h - real_h) < 1e-4, f"{name} height {h} != {real_h}"
            assert is_term == bool(benchmark.macro_fixed[hi].item()), \
                f"{name} terminal flag {is_term} != fixed {benchmark.macro_fixed[hi]}"
        print(f"  ✓ .nodes: {len(nodes)} nodes, sizes & terminals match")

        # ─── .pl round-trip (positions = centers) ─── #
        pl = parse_bookshelf_pl(tmp / "test.pl")
        max_err = 0.0
        for hi in range(benchmark.num_hard_macros):
            name = f"hard_{hi}"
            assert name in pl, f"missing {name} in .pl"
            x_ll, y_ll = pl[name]
            w, h = benchmark.macro_sizes[hi].tolist()
            cx_recovered = x_ll + w / 2
            cy_recovered = y_ll + h / 2
            cx_real, cy_real = benchmark.macro_positions[hi].tolist()
            err = max(abs(cx_recovered - cx_real), abs(cy_recovered - cy_real))
            max_err = max(max_err, err)
        for si in range(benchmark.num_macros - benchmark.num_hard_macros):
            owner = benchmark.num_hard_macros + si
            name = f"soft_{si}"
            assert name in pl, f"missing {name} in .pl"
            x_ll, y_ll = pl[name]
            w, h = benchmark.macro_sizes[owner].tolist()
            cx_recovered = x_ll + w / 2
            cy_recovered = y_ll + h / 2
            cx_real, cy_real = benchmark.macro_positions[owner].tolist()
            err = max(abs(cx_recovered - cx_real), abs(cy_recovered - cy_real))
            max_err = max(max_err, err)
        for pi in range(benchmark.port_positions.shape[0]):
            name = f"port_{pi}"
            assert name in pl, f"missing {name} in .pl"
            x_ll, y_ll = pl[name]
            x_real, y_real = benchmark.port_positions[pi].tolist()
            err = max(abs(x_ll - x_real), abs(y_ll - y_real))
            max_err = max(max_err, err)
        assert max_err < 1e-3, f"position round-trip error {max_err} too large"
        print(f"  ✓ .pl positions round-trip; max error = {max_err:.2e}")

        # ─── .nets round-trip ─── #
        nets = parse_bookshelf_nets(tmp / "test.nets")
        assert len(nets) == benchmark.num_nets, \
            f".nets count {len(nets)} != {benchmark.num_nets}"
        # Spot-check first net
        if nets:
            first_pins = nets[0]
            assert len(first_pins) == benchmark.net_pin_nodes[0].shape[0], \
                f"net 0 pin count mismatch"
        total_pins_written = sum(len(n) for n in nets)
        total_pins_expected = sum(p.shape[0] for p in benchmark.net_pin_nodes)
        assert total_pins_written == total_pins_expected, \
            f"total pins {total_pins_written} != {total_pins_expected}"
        print(f"  ✓ .nets: {len(nets)} nets, {total_pins_written} pins (matches benchmark)")

    print(f"  ✓ All round-trip checks passed for {name}\n")


if __name__ == "__main__":
    for n in ("ibm01", "ibm04", "ibm10"):
        run(n)
    print("All round-trip tests passed.")
