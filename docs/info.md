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
