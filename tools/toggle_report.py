#!/usr/bin/env python3
"""
Switching-activity report from an RTL VCD.

Answers one question: how much of the flip-flop activity in this tile is the
weight register, and how much of that happens during inference rather than
during weight load? That ratio is the weight-stationary argument, and it is
the number worth quoting - unlike absolute milliwatts, it needs no liberty
file and no PDK, so it is reproducible by anyone with Icarus.

    cd test && make clean && make FST=      # plain VCD, not FST
    python3 ../tools/toggle_report.py tb.vcd
"""

import re
import sys
from collections import defaultdict


def parse(path):
    """Return (signals, events). signals: id -> (name, width).
    events: list of (time, id, value) in file order."""
    signals, events = {}, []
    scope, t = [], 0
    var_re = re.compile(r"\$var\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)(?:\s*\[[^\]]*\])?\s+\$end")

    with open(path) as f:
        in_defs = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if in_defs:
                if line.startswith("$scope"):
                    scope.append(line.split()[2])
                elif line.startswith("$upscope"):
                    scope and scope.pop()
                elif line.startswith("$var"):
                    m = var_re.match(line)
                    if m:
                        w, sid, nm = int(m.group(1)), m.group(2), m.group(3)
                        signals[sid] = (".".join(scope + [nm]), w)
                elif line.startswith("$enddefinitions"):
                    in_defs = False
                continue
            if line[0] == "#":
                t = int(line[1:])
            elif line[0] in "bB":
                val, sid = line[1:].split(None, 1)
                events.append((t, sid.strip(), val))
            elif line[0] in "rR":
                pass
            elif line[0] in "01xXzZ":
                events.append((t, line[1:].strip(), line[0]))
    return signals, events


def bit_transitions(prev, cur, width):
    """Hamming distance between two VCD scalar/vector values, X/Z counted as
    a change. VCD elides leading zeros, so left-pad first."""
    p = prev.rjust(width, "0" if prev[:1] not in "xXzZ" else prev[0])
    c = cur.rjust(width, "0" if cur[:1] not in "xXzZ" else cur[0])
    n = max(len(p), len(c))
    p, c = p.rjust(n, "0"), c.rjust(n, "0")
    return sum(1 for a, b in zip(p, c) if a != b)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tb.vcd"
    signals, events = parse(path)
    print(f"{len(signals)} signals, {len(events)} value changes\n")

    last = {}
    toggles = defaultdict(int)
    for t, sid, val in events:
        if sid not in signals:
            continue
        name, width = signals[sid]
        if sid in last:
            toggles[name] += bit_transitions(last[sid], val, width)
        last[sid] = val

    width_of = {n: w for n, w in signals.values()}
    total = sum(toggles.values())
    clk = sum(v for k, v in toggles.items() if k.endswith(".clk"))

    def group(pred):
        return sum(v for k, v in toggles.items() if pred(k))

    wmem = group(lambda k: ".wmem" in k)
    xreg = group(lambda k: ".xreg" in k)
    acc = group(lambda k: ".acc" in k or ".hid" in k)
    ctrl = group(lambda k: any(s in k for s in (".state", ".step", ".bitcnt",
                                                ".done", ".busy", ".class_idx")))

    print(f"{'group':28s} {'bit-toggles':>12s}  {'share':>7s}")
    print("-" * 52)
    for label, v in [("wmem (320-bit weights)", wmem),
                     ("xreg (16-bit input)", xreg),
                     ("accumulators + hidden", acc),
                     ("control / FSM / outputs", ctrl),
                     ("clk", clk)]:
        print(f"{label:28s} {v:12,d}  {v / total:6.1%}")
    print(f"{'TOTAL (all signals)':28s} {total:12,d}")

    print("\ntop 15 signals by bit-toggles")
    for name, v in sorted(toggles.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {name:46s} w={width_of.get(name, 1):3d}  {v:9,d}")

    # ------------------------------------------------ phase split ----------
    # The tile is either shifting weights in or running an inference. Those
    # are the two power regimes and they are nothing alike, so a single
    # average over the run would hide the whole point.
    state_id = next((sid for sid, (n, _) in signals.items()
                     if n.endswith(".state")), None)
    if state_id is None:
        return

    def is_active(v):
        return any(c == "1" for c in v)          # state != S_IDLE (2'd0)

    last2, active = {}, False
    phase = {True: defaultdict(int), False: defaultdict(int)}
    for t, sid, val in events:
        if sid not in signals:
            continue
        name, width = signals[sid]
        if sid in last2:
            phase[active][name] += bit_transitions(last2[sid], val, width)
        last2[sid] = val
        if sid == state_id:
            active = is_active(val)

    print("\nby phase (state == S_IDLE is weight-load/idle, else inference)")
    print(f"{'group':28s} {'idle/load':>12s} {'inference':>12s}")
    print("-" * 54)
    for label, pred in [("wmem (320-bit weights)", lambda k: ".wmem" in k),
                        ("xreg (16-bit input)", lambda k: ".xreg" in k),
                        ("accumulators + hidden",
                         lambda k: ".acc" in k or ".hid" in k),
                        ("everything else",
                         lambda k: not (".wmem" in k or ".xreg" in k
                                        or ".acc" in k or ".hid" in k))]:
        a = sum(v for k, v in phase[False].items() if pred(k))
        b = sum(v for k, v in phase[True].items() if pred(k))
        print(f"{label:28s} {a:12,d} {b:12,d}")
    ta = sum(phase[False].values())
    tb_ = sum(phase[True].values())
    print(f"{'TOTAL':28s} {ta:12,d} {tb_:12,d}")
    w_inf = sum(v for k, v in phase[True].items() if ".wmem" in k)
    if tb_:
        print(f"\nweight register share of inference activity: "
              f"{w_inf / tb_:.4%}  ({w_inf:,d} of {tb_:,d} bit-toggles)")


if __name__ == "__main__":
    main()
