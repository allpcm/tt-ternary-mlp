"""
Cocotb port of tb.v for tt_um_pcm_ternary_mlp.

Same vectors, same golden model, same protocol. The one rule carried over from
the Verilog bench and worth repeating because breaking it cost a day in
September:

    ALL STIMULUS IS DRIVEN ON THE FALLING EDGE.

The DUT samples every control line on the rising edge. Driving on the rising
edge is a race, and in a gate-level netlist with real cell delays it produces
intermittent mismatches that look like logic bugs and are not.

Vectors are generated in-process by importing golden.py rather than reading
weights.bits / vectors.bits, so CI has no file-ordering dependency and the
golden model cannot silently drift out of sync with its own output files.

    make                 RTL
    make GATES=yes       gate level (needs gate_level_netlist.v from GDS.zip)
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

import golden as G

# uio_in control bit positions - must match the localparams in the Verilog
C_WSHIFT = 0
C_WDATA = 1
C_START = 2
C_INLOAD = 3

CLK_NS = 20  # 50 MHz, the silicon target


def _clock(dut):
    """Clock(..., unit=) in cocotb 2.x, units= in 1.x. Support both."""
    try:
        return Clock(dut.clk, CLK_NS, unit="ns")
    except TypeError:
        return Clock(dut.clk, CLK_NS, units="ns")


def bit(sig, n):
    """Bit n of an 8-bit output, as an int. Raises on X/Z, which is the point:
    a resolved-to-zero X in a gate-level run is a silent false pass."""
    s = str(sig.value)        # str() works on cocotb 1.x and 2.x; .binstr is deprecated
    c = s[len(s) - 1 - n]
    if c not in "01":
        raise AssertionError(f"{sig._name}[{n}] is '{c}' (full value {s})")
    return int(c)


def field(sig, hi, lo):
    s = str(sig.value)        # str() works on cocotb 1.x and 2.x; .binstr is deprecated
    chunk = s[len(s) - 1 - hi: len(s) - lo]
    if any(c not in "01" for c in chunk):
        raise AssertionError(f"{sig._name}[{hi}:{lo}] is '{chunk}'")
    return int(chunk, 2)


async def reset(dut):
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)


async def load_weights(dut, bits):
    """Shift the 320-bit image in MSB-first, matching golden.pack ordering:
    wmem <= {wmem[318:0], w_data}, so the first bit clocked in lands at
    wmem[319]."""
    uio = 0
    for b in reversed(bits):
        await FallingEdge(dut.clk)
        uio = (1 << C_WSHIFT) | (b << C_WDATA)
        dut.uio_in.value = uio
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0


async def load_input(dut, x16):
    """Two bytes, high then low. xreg <= {xreg[7:0], ui_in} twice gives
    xreg == x16, so xreg[i] is input i."""
    await FallingEdge(dut.clk)
    dut.ui_in.value = (x16 >> 8) & 0xFF
    dut.uio_in.value = 1 << C_INLOAD
    await FallingEdge(dut.clk)
    dut.ui_in.value = x16 & 0xFF
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0
    dut.ui_in.value = 0


async def run_inference(dut):
    """Pulse start, wait for done on uo_out[7]. Returns the measured latency in
    clock cycles, which must be exactly 26 for every input."""
    await FallingEdge(dut.clk)
    dut.uio_in.value = 1 << C_START
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0

    cycles = 0
    busy_seen = False
    while True:
        await RisingEdge(dut.clk)
        cycles += 1
        if bit(dut.uio_out, 4):          # busy
            busy_seen = True
        if bit(dut.uo_out, 7):           # done on the dedicated output
            # done must also appear on uio[5], per info.yaml. Checking both
            # catches a pinout/documentation drift, which is what the
            # 19 Oct fix to uio_out was.
            assert bit(dut.uio_out, 5) == 1, "done on uo_out[7] but not uio[5]"
            assert busy_seen, "busy never asserted during inference"
            return cycles
        assert cycles < 100, "done never asserted within 100 cycles"


def make_vectors():
    """Regenerate exactly what golden.py emits: NSETS weight sets, NVEC inputs
    each, same seed. Returns [(bits, [(x16, cls)])]."""
    random.seed(G.SEED)
    sets = []
    for _ in range(G.NSETS):
        w1 = [[G.ternary() for _ in range(G.N_HID)] for _ in range(G.N_IN)]
        w2 = [[G.ternary() for _ in range(G.N_OUT)] for _ in range(G.N_HID)]
        bits = G.pack(w1, w2)
        vecs = []
        for v in range(G.NVEC):
            density = [0.15, 0.5, 0.85][v % 3]
            x = [1 if random.random() < density else 0 for _ in range(G.N_IN)]
            cls, _, _ = G.infer(w1, w2, x)
            x16 = sum(x[i] << i for i in range(G.N_IN))
            vecs.append((x16, cls))
        sets.append((bits, vecs))
    return sets


@cocotb.test()
async def test_ternary_mlp(dut):
    cocotb.start_soon(_clock(dut).start())

    sets = make_vectors()
    total = sum(len(v) for _, v in sets)
    dut._log.info(f"{len(sets)} weight sets x {len(sets[0][1])} vectors "
                  f"= {total}, seed {G.SEED}")

    npass = 0
    latencies = set()

    for s, (bits, vecs) in enumerate(sets):
        await reset(dut)
        await load_weights(dut, bits)

        for v, (x16, expect) in enumerate(vecs):
            await load_input(dut, x16)
            cycles = await run_inference(dut)
            latencies.add(cycles)
            got = field(dut.uio_out, 7, 6)

            assert got == expect, (
                f"set {s} vec {v}: x=0x{x16:04x} expected class {expect}, "
                f"got {got}")
            npass += 1

    dut._log.info(f"{npass}/{total} vectors match the golden model")

    # The architectural claim: fixed latency, no data-dependent branching.
    # If this set has more than one element, something is scheduling at
    # runtime and the whole pitch is wrong.
    assert latencies == {26}, (
        f"latency is not a fixed 26 cycles: observed {sorted(latencies)}")
    dut._log.info("latency fixed at 26 cycles across all vectors")
