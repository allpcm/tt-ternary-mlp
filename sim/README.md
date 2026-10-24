# Standalone Verilog bench

`tb_standalone.v` is the original self-contained Icarus bench. It is NOT what
Tiny Tapeout CI runs — that is `../test/` (cocotb). This one is kept because it
needs no Python environment and is useful for a quick check on a strange machine.

    python3 golden.py          # writes weights.bits, vectors.bits
    mkdir -p out
    iverilog -g2012 -o out/tb.vvp ../src/tt_um_pcm_ternary_mlp.v tb_standalone.v
    vvp out/tb.vvp

Both benches must agree. If they diverge, the cocotb one is authoritative
because it is what gates the GDS build.
