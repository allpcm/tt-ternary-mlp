#!/usr/bin/env python3
"""
Gate-level switching activity for tt_um_pcm_ternary_mlp, split by phase.

The post-synthesis counterpart to toggle_report.py, and the one to quote: it
measures the netlist LibreLane actually produced, after clock-tree synthesis
and after the 431 hold buffers and 645 timing-repair buffers it inserted.
Measured on run 40146396240:

    group                              idle/load      inference
    wmem[319:0] weight register          116,434              0
    xreg[15:0] input register              1,476              0
    all other nets in the tile         2,674,392      2,965,513
    TOTAL (tile nets)                  2,792,302      2,965,513

Zero of 2,965,513 inference toggles are in the weight register. All 320
weight nets survive synthesis by name and are individually tracked.

Note on VCD aliasing: fst2vcd emits one identifier per net, but a net can
appear under many scopes - every FILLER cell shares VPWR/VGND - so 77,502
names collapse to 9,767 ids. An id therefore maps to a SET of names;
toggles are counted once per id and attributed by checking all aliases.
Keying on a single name silently loses the registers to a filler scope.

Input is the gate-level dump from the gl_test CI job, which is FST despite
being named tb.vcd (the test Makefile passes -fst):

    fst2vcd tb.fst -o tb.real.vcd        # fst2vcd ships with GTKWave
    python3 tools/gl_toggle_report.py tb.real.vcd
"""
import sys
from collections import defaultdict

path = sys.argv[1]
aliases = defaultdict(set)
widths, scope = {}, []

with open(path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("$scope"):
            scope.append(line.split()[2])
        elif line.startswith("$upscope"):
            scope and scope.pop()
        elif line.startswith("$var"):
            t = line.split()
            if len(t) >= 6 and t[-1] == "$end":
                w, sid = int(t[2]), t[3]
                nm = " ".join(t[4:-1])
                if nm.endswith("]") and "[" in nm:
                    nm = nm[:nm.rindex("[")].strip()
                aliases[sid].add(".".join(scope + [nm.lstrip("\\")]))
                widths[sid] = w
        elif line.startswith("$enddefinitions"):
            break

def has(sid, suffix):
    return any(n.endswith(suffix) for n in aliases[sid])

st_ids = [s for s in aliases if has(s, ".state")]
print(f"{len(aliases)} unique nets, {sum(len(v) for v in aliases.values())} "
      f"names; phase markers: {len(st_ids)}", file=sys.stderr)

def ham(p, c, w):
    p, c = p.rjust(w, "0"), c.rjust(w, "0")
    n = max(len(p), len(c))
    return sum(1 for a, b in zip(p.rjust(n, "0"), c.rjust(n, "0")) if a != b)

last, st_val, active = {}, {}, False
tog = {False: defaultdict(int), True: defaultdict(int)}

with open(path) as f:
    defs = True
    for line in f:
        line = line.rstrip("\n")
        if defs:
            if line.startswith("$enddefinitions"):
                defs = False
            continue
        if not line or line[0] == "#":
            continue
        c0 = line[0]
        if c0 in "bB":
            parts = line[1:].split(None, 1)
            if len(parts) != 2:
                continue
            val, sid = parts[0], parts[1].strip()
        elif c0 in "01xXzZ":
            val, sid = c0, line[1:].strip()
        else:
            continue
        if sid not in aliases:
            continue
        if sid in last:
            tog[active][sid] += ham(last[sid], val, widths[sid])
        last[sid] = val
        if sid in st_ids:
            st_val[sid] = val
            active = any(v == "1" for v in st_val.values())

in_tile = lambda s: any("user_project" in n for n in aliases[s])
is_wmem = lambda s: in_tile(s) and has(s, ".wmem")
is_xreg = lambda s: in_tile(s) and has(s, ".xreg")

def agg(b, pred):
    return sum(v for s, v in b.items() if pred(s))

print(f"\n{'group':34s} {'idle/load':>14s} {'inference':>14s}")
print("-" * 64)
for label, pred in [("wmem[319:0] weight register", is_wmem),
                    ("xreg[15:0] input register", is_xreg),
                    ("all other nets in the tile",
                     lambda s: in_tile(s) and not is_wmem(s) and not is_xreg(s))]:
    print(f"{label:34s} {agg(tog[False], pred):14,d} {agg(tog[True], pred):14,d}")

ti, ta = agg(tog[False], in_tile), agg(tog[True], in_tile)
print(f"{'TOTAL (tile nets)':34s} {ti:14,d} {ta:14,d}")
w = agg(tog[True], is_wmem)
print(f"\nweight-register nets: {sum(1 for s in aliases if is_wmem(s))}")
if ta:
    print(f"weight register share of gate-level inference activity: "
          f"{w / ta:.4%}  ({w:,d} of {ta:,d})")
