# Ternary MLP Inference Tile

## How it works

This tile runs a small neural network entirely on-chip. All 160 weights live in
flip-flops next to the arithmetic that uses them. They are loaded once over a
serial chain at power-up and never fetched again. During inference the only
things crossing the die boundary are the 16-bit input vector going in and a
class index coming out.

The network is 16 binary inputs to 8 hidden units to 4 output classes. Weights
are ternary: each is -1, 0, or +1.

Three design commitments are worth calling out, because they are the reason the
tile is this small:

**There is no multiplier.** A binary activation times a ternary weight is one of
{-1, 0, +1}. So every multiply-accumulate in the network reduces to "maybe add
one, maybe subtract one, maybe do nothing." Each accumulator is an up/down
counter, not an adder feeding a multiplier.

**There is no runtime scheduler and no data-dependent branching.** Control is a
4-bit counter and a four-state machine. Every inference takes exactly 26 cycles
regardless of what the input is. The schedule was fixed when the design was
compiled.

**Memory and compute are not separated.** The weight row for the current step is
selected by a mux tree and broadcast to all eight accumulators at once. Nothing
waits on a memory access, because there is no memory access.

### Cycle-by-cycle

| Cycles | Phase | What happens |
|---|---|---|
| 1 | IDLE to L1 | Layer-1 accumulators cleared |
| 16 | L1 | One input bit per cycle; all 8 hidden accumulators update in parallel |
| 1 | ACT | Hidden layer binarized (`hid[j] = acc1[j] > 0`); layer-2 accumulators cleared |
| 8 | L2 | One hidden bit per cycle; all 4 output accumulators update in parallel |
| — | — | Argmax is combinational; result is valid when `done` rises |

26 cycles total. At 50 MHz that is 520 ns per inference.

### Measured build

CI run 40150872179, LibreLane 3.0.14, sky130A, open_pdks 8afc834.

| | |
|---|---|
| Synthesis | 1,834 cells, 22,700 µm² |
| Post-PnR standard cells | 3,587 instances, 31,266 µm² |
| Die area (2×2) / utilization | 75,602 µm² / 41.4% |
| Flip-flops | 419, of which 320 are the weight register |
| Worst setup slack | 6.58 ns of 20 ns (ss, 100 °C, 1.60 V) |
| Worst hold slack | +0.106 ns (ff, −40 °C, 1.95 V) |
| Setup/hold TNS, WNS | 0.0 at every corner |
| DRC route / magic | 0 / 0 |
| Tiny Tapeout precheck | 15 / 15 |
| Verilator `-Wall`, inferred latches | clean, 0 |

Of the 8,912 µm² of sequential area, roughly 6,800 µm² is the 320-bit weight
register — about 76% of all flip-flop area, holding data that does not change
while the tile computes.

### Measured power

Annotated OpenSTA against the real vector set, sky130A, tt 25 °C 1.80 V:

| group | total | share |
|---|---|---|
| Sequential | 887.9 µW | 52.1% |
| Clock | 489.9 µW | 28.7% |
| Combinational | 327.7 µW | 19.2% |
| **Total** | **1.706 mW** | |

82.5% of that is internal power, 17.5% switching, leakage negligible at 46 nW.
The figure averages the whole test — roughly 1,600 cycles of weight shifting and
2,600 of inference — so it is not inference-only. At 520 ns per classification,
energy per inference is approximately 0.9 nJ.

The weight register holds all 320 of its bits static through inference and
contributes no switching power, yet the design still draws 1.7 mW, because every
one of the 419 flip-flops is clocked each cycle and a flip-flop dissipates
internal power on a clock edge regardless of whether its input changed. The
netlist contains no clock-gating cells. Gating the weight register outside its
load phase is the obvious next saving and is not implemented here.

## How to test

1. Assert `rst_n` low for several cycles, then release.
2. Shift in 320 weight bits: put each bit on `uio[1]`, hold `uio[0]` high, one
   clock per bit. Send the most significant bit of the weight image first.
3. Load the activation vector: put the high byte on `ui[7:0]` with `uio[3]`
   high for one clock, then the low byte for one clock.
4. Pulse `uio[2]` high for one clock to start.
5. Wait 26 clocks. `uo[7]` and `uio[5]` go high. The predicted class appears on
   `uio[7:6]` and as a digit on the 7-segment display.

`test/golden.py` generates weight images and input vectors along with the
expected class, computed by an independent Python reference implementation.

## External hardware

None required. The 7-segment display on the Tiny Tapeout demo board shows the
predicted class directly. The RP2040 on the demo board can drive the weight
load and activation input over its GPIO.
