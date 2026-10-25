# Annotated power for tt_um_pcm_ternary_mlp.
#
# Reports power under the REAL vector set rather than a 0.5 default activity
# factor. That distinction is the whole point: this design's weight register
# is 320 of its 419 flip-flops and never toggles during inference, so a
# default-activity estimate overstates it badly. LibreLane's own
# power__total metric (1.86 mW) is exactly such an estimate - do not quote it.
#
# Needs OpenSTA plus the sky130 liberty files, which are not in the
# tt_submission artifact. Easiest source is the LibreLane container, which
# already has both:
#
#   docker run --rm -it -v "$PWD":/work -w /work \
#     ghcr.io/librelane/librelane:latest \
#     sta -no_splash tools/power.tcl
#
# Inputs expected in the working directory:
#   gl/tt_um_pcm_ternary_mlp.v   gate-level netlist (tt_submission/)
#   gl/tt_um_pcm_ternary_mlp.sdc constraints        (GDS_logs run dir)
#   gl/tb.vcd                    activity dump      (gatelevel_test_results/)
#
# The shipped tb.vcd is FST despite the name, because the test Makefile
# passes -fst. Convert it first, or regenerate a true VCD:
#   fst2vcd gl/tb.vcd -o gl/tb.real.vcd
#   # or: cd test && make clean && make GATES=yes FST=

set pdk_lib $::env(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/lib
set corner  sky130_fd_sc_hd__tt_025C_1v80.lib

read_liberty $pdk_lib/$corner
read_verilog gl/tt_um_pcm_ternary_mlp.v
link_design  tt_um_pcm_ternary_mlp
read_sdc     gl/tt_um_pcm_ternary_mlp.sdc

# Scope must name the DUT instance inside the testbench, not the testbench.
# In test/tb.v the instance is `user_project`, so the scope is tb/user_project.
read_vcd -scope tb/user_project gl/tb.real.vcd

puts "\n=== power, annotated from the cocotb vector set ==="
report_power

puts "\n=== power by hierarchy ==="
report_power -instances [get_cells *]

puts "\n=== timing sanity, should match the LibreLane run ==="
report_checks -path_delay max -digits 3
report_checks -path_delay min -digits 3
