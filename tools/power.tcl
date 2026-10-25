# Annotated power for tt_um_pcm_ternary_mlp.
#
# Reports power under the REAL vector set rather than a default activity
# factor. Build OpenSTA and run this end-to-end with:
#
#     bash tools/build_opensta_power.sh
#
# which compiles CUDD and OpenSTA from source, fetches the sky130 liberty
# files at the exact open_pdks hash this design was built with, writes this
# file with the discovered paths, and runs it.
#
# Inputs, all from one CI run so they describe the same netlist:
#   gl/tt_um_pcm_ternary_mlp.v     gate netlist   (tt_submission/)
#   gl/tt_um_pcm_ternary_mlp.sdc   constraints    (GDS_logs .../final/sdc/)
#   gl/tb.real.vcd                 activity       (gatelevel_test_results/)
#
# The gl_test artifact named tb.vcd is FST, because the test Makefile passes
# -fst. Convert it first: fst2vcd tb.fst -o tb.real.vcd
#
# Result on run 40146396240 (sky130A, tt 25C 1v80):
#
#   Sequential      887.9 uW   52.1%
#   Clock           489.9 uW   28.7%
#   Combinational   327.7 uW   19.2%
#   Total          1705.6 uW          (82.5% internal, 17.5% switching)
#
# Read that split before quoting the total. The weight register holds 320
# bits static through inference and contributes no switching power, yet the
# design still draws 1.7 mW: every one of the 419 flip-flops is clocked each
# cycle, and a flip-flop dissipates internal power on a clock edge whether or
# not D changed. There is no clock-gating cell in the netlist. Zero data
# toggles is not zero power.
#
# The total averages the whole run (~1,600 cycles of weight load, ~2,600 of
# inference), so it is not an inference-only figure.

read_liberty LIBERTY_PATH
read_verilog gl/tt_um_pcm_ternary_mlp.v
link_design tt_um_pcm_ternary_mlp
read_sdc gl/tt_um_pcm_ternary_mlp.sdc

# The DUT instance inside the testbench is tb/user_project (see test/tb.v).
read_vcd -scope tb/user_project gl/tb.real.vcd

puts "\n########## ANNOTATED POWER ##########"
report_power -digits 4

puts "\n########## TIMING SANITY ##########"
report_checks -path_delay max -digits 3
report_checks -path_delay min -digits 3
exit
