![gds](../../workflows/gds/badge.svg) ![docs](../../workflows/docs/badge.svg) ![test](../../workflows/test/badge.svg)

# Ternary MLP Inference Tile

A 16-8-4 neural network classifier on a 2×2 Tiny Tapeout tile. Weight-stationary,
no multiplier, no scheduler, fixed 26-cycle latency.

Built as a teaching artifact for a high school pre-engineering program: the design
is small enough that a student can read the Verilog beside the Python reference and
see the same operation twice, and the claims it makes about itself are all
measurable from this repository.

**[Full datasheet and I/O protocol → `docs/info.md`](docs/info.md)**

---

## What it does

Sixteen binary inputs, eight binary hidden units, four output classes. Every weight
is −1, 0 or +1, and all 160 of them live in flip-flops beside the arithmetic that
uses them. A binary activation times a ternary weight is one of {−1, 0, +1}, so
every multiply-accumulate collapses to an increment, a decrement, or nothing. There
is no multiplier on the die.

The shipped demonstration network classifies 4×4 binary glyphs — `BAR`, `COL`, `X`
and `BOX`. Three of the four carry exactly eight lit pixels, so pixel count alone
cannot separate them; the network has to learn position.

Weights are **not** baked into the silicon. They shift in at runtime over
`uio[1]`, 320 bits, so the same chip runs any network you train for it.

## Verified results

From CI run `40150872179` (LibreLane 3.0.14, sky130A, open_pdks `8afc834`):

| | |
|---|---|
| Tile size / clock | 2×2, 50 MHz |
| Latency | 26 cycles, fixed, input-independent |
| Synthesis | 1,834 cells, 22,700 µm² |
| Post-PnR standard cells | 3,587 instances, 31,266 µm² |
| Die area / utilization | 75,602 µm² / 43.1% |
| Flip-flops | 419, of which 320 are the weight register |
| Worst setup slack | 6.58 ns of a 20 ns period (ss, 100 °C, 1.60 V) |
| Worst hold slack | +0.106 ns (ff, −40 °C, 1.95 V) |
| Setup/hold TNS and WNS | 0.0 at every corner |
| DRC (route and magic) | 0 |
| Tiny Tapeout precheck | 15 / 15 |
| Verilator `-Wall` | clean |
| Testbench | 200 vectors, RTL and gate-level, 2/2 passing |

### The weight-stationary claim, measured

The point of a weight-stationary design is that stored weights do not switch while
the array computes. That is checkable rather than assertable, so this repo checks
it — on the netlist LibreLane produced, after clock-tree synthesis and after 431
hold buffers and 645 timing-repair buffers were inserted:

| group | idle / weight load | inference |
|---|---|---|
| `wmem[319:0]` weight register | 116,434 | **0** |
| `xreg[15:0]` input register | 1,476 | 0 |
| all other nets in the tile | 2,674,392 | 2,965,513 |

Zero of 2,965,513 inference-phase bit-toggles are in the weight register — 320 of
the design's 419 flip-flops, roughly 6,800 µm², completely static during compute.

Two honest qualifications. `w1_row` and `w2_row` still toggle during inference;
those are the combinational muxes *reading* rows out of the register, so the
storage is static but the read path is not. And **absolute power in milliwatts is
not yet measured** — `tools/power.tcl` is written and waiting on an annotated
OpenSTA run. LibreLane's own `power__total` metric assumes default switching
activity and is meaningless for a design whose largest register never toggles;
it is deliberately not quoted here.

### What the demonstration network achieves

| | accuracy |
|---|---|
| Clean prototypes | 100% |
| Up to 2 flipped pixels | 100% |
| Up to 3 flipped pixels | 97.7% |
| Up to 4 flipped pixels | 95.1% |

33 of 160 weights are non-zero — the trained network is 79% sparse.

## Running it locally

Needs [Icarus Verilog](https://steveicarus.github.io/iverilog/) and Python 3.11+.

```bash
cd test
pip install -r requirements.txt
make                      # RTL simulation, 200 vectors
make GATES=yes            # gate-level, needs gate_level_netlist.v from GDS.zip
```

Expect `TESTS=2 PASS=2 FAIL=0` and `latency fixed at 26 cycles`.

A standalone Icarus bench that needs no Python environment lives in `sim/`; see
`sim/README.md`. Both benches must agree, and the cocotb one is authoritative
because it is what gates the GDS build.

### Train your own network

```bash
python3 tools/train_glyphs.py
```

Edit the `GLYPHS` table to your own four shapes. Training runs in three stages: a
float surrogate with a sigmoid standing in for the hard step so gradients exist,
TWN magnitude ternarization, then coordinate descent against the *exact* integer
model the silicon implements. Only that last stage optimizes what the hardware
actually computes, which is why there is no train-to-deploy gap to discover during
bring-up. The script writes `tools/trained_weights.py`; copy the arrays into
`test/golden.py` as `TRAINED_W1` / `TRAINED_W2` and the whole bench revalidates
against them unchanged.

### Reproduce the activity measurement

```bash
cd test && make clean && make FST=          # plain VCD rather than FST
python3 ../tools/toggle_report.py tb.vcd    # RTL

# gate level, from the gl_test CI artifact
fst2vcd tb.fst -o tb.real.vcd               # fst2vcd ships with GTKWave
python3 tools/gl_toggle_report.py tb.real.vcd
```

The CI artifact named `tb.vcd` is actually FST, because the test Makefile passes
`-fst`. Convert it first.

## Using it from a microcontroller

The tile is a peripheral, not a system. It has no clock generator and no boot
sequence — a host drives `clk`, holds the reset, shifts in the weights, feeds an
activation vector and reads back a class. That makes it a clean target for an
Arduino, a Pico, or the RP2040 already on the Tiny Tapeout demo board.

The host loop is four steps and no interrupts:

1. hold `rst_n` low for a few clocks, release
2. clock in 320 weight bits on `uio[1]` with `uio[0]` high, MSB first
3. write the activation high byte then low byte on `ui[7:0]` with `uio[3]` high
4. pulse `uio[2]`, clock 26 times, read the class from `uio[7:6]`

Because the host supplies the clock, it can run the chip arbitrarily slowly —
or single-step it — which makes the internal state observable on a logic
analyzer or just an LED. A student can watch `busy` rise, count 26 clocks, and
see `done` go high.

**Budget about 17 I/O lines**: `clk`, `rst_n`, eight activation bits, four
control bits, and three to read back. An Uno has roughly that many only if you
sacrifice D0/D1, which costs you USB serial for debugging — a Mega, Nano Every
or Pico is the comfortable choice.

### The same code drives an FPGA

`src/tt_um_pcm_ternary_mlp.v` is ordinary portable Verilog. It synthesizes for a
Spartan-7 in Vivado with no changes, so the identical host sketch can talk to an
FPGA board today and to the fabricated tile when silicon returns. The wiring and
the protocol are the constant; only what sits on the other end of the ribbon
cable changes.

## Layout

```
src/     tt_um_pcm_ternary_mlp.v   the design
test/    cocotb bench + golden.py  what CI runs; authoritative
sim/     standalone Icarus bench   no Python needed
tools/   train_glyphs.py           three-stage ternary training
         toggle_report.py          RTL switching activity
         gl_toggle_report.py       gate-level switching activity
         power.tcl                 OpenSTA annotated power (not yet run)
docs/    info.md                   datasheet and I/O protocol
```

## Pinout summary

`ui[7:0]` activation byte · `uo[6:0]` 7-segment · `uo[7]` done ·
`uio[0]` weight shift enable · `uio[1]` weight data · `uio[2]` start ·
`uio[3]` activation load · `uio[4]` busy · `uio[5]` done · `uio[7:6]` class index

Full protocol in [`docs/info.md`](docs/info.md).

## Author

Phillip Moore — CTE Pre-Engineering, Foreman College and Career Academy, Chicago.

## About Tiny Tapeout

[Tiny Tapeout](https://tinytapeout.com) makes it practical to get a digital design
manufactured on a real chip. See the [FAQ](https://tinytapeout.com/faq/) and the
[digital design lessons](https://tinytapeout.com/digital_design/).

Licensed under Apache 2.0 — see [LICENSE](LICENSE).
