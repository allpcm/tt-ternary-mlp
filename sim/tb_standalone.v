`default_nettype none
`timescale 1ns / 1ps

module tb;  // standalone: generates its own clock, not for cocotb

    localparam W_BITS = 320;
    localparam NSETS  = 5;
    localparam NVEC   = 20;

    reg        clk = 0;
    reg        rst_n = 0;
    reg        ena = 1;
    reg  [7:0] ui_in = 0;
    reg  [7:0] uio_in = 0;
    wire [7:0] uo_out;
    wire [7:0] uio_out;
    wire [7:0] uio_oe;

    // control bit positions inside uio_in
    localparam C_WSHIFT = 0;
    localparam C_WDATA  = 1;
    localparam C_START  = 2;
    localparam C_INLOAD = 3;

    tt_um_pcm_ternary_mlp dut (
        .ui_in   (ui_in),
        .uo_out  (uo_out),
        .uio_in  (uio_in),
        .uio_out (uio_out),
        .uio_oe  (uio_oe),
        .ena     (ena),
        .clk     (clk),
        .rst_n   (rst_n)
    );

    always #5 clk = ~clk;   // 100 MHz in sim; silicon target is 50 MHz

    // test data
    reg [0:0]  wbits [0:NSETS*W_BITS-1];
    reg [17:0] vecs  [0:NSETS*NVEC-1];

    integer s, v, b;
    integer pass = 0, fail = 0;
    integer cyc_start, cyc_done, cyc_count = 0;
    reg [15:0] x;
    reg [1:0]  expect_cls, got_cls;

    // free-running cycle counter so we can report real latency
    always @(posedge clk) cyc_count <= cyc_count + 1;

    task do_reset;
        begin
            rst_n = 0; uio_in = 0; ui_in = 0;
            repeat (4) @(posedge clk);
            rst_n = 1;
            @(posedge clk);
        end
    endtask

    // All stimulus is driven on the negative edge so the DUT samples stable
    // values on the positive edge. Driving on the same edge the DUT clocks is
    // a race and will produce intermittent, misleading mismatches.
    task load_weights(input integer set_idx);
        integer i;
        begin
            for (i = 0; i < W_BITS; i = i + 1) begin
                @(negedge clk);
                uio_in[C_WDATA]  = wbits[set_idx*W_BITS + i];
                uio_in[C_WSHIFT] = 1'b1;
            end
            @(negedge clk);
            uio_in[C_WSHIFT] = 1'b0;
            uio_in[C_WDATA]  = 1'b0;
        end
    endtask

    task load_input(input [15:0] xv);
        begin
            @(negedge clk);
            ui_in = xv[15:8];
            uio_in[C_INLOAD] = 1'b1;
            @(negedge clk);
            ui_in = xv[7:0];
            @(negedge clk);
            uio_in[C_INLOAD] = 1'b0;
            ui_in = 8'd0;
        end
    endtask

    task run_inference;
        integer guard;
        begin
            @(negedge clk);
            uio_in[C_START] = 1'b1;
            cyc_start = cyc_count;
            @(negedge clk);
            uio_in[C_START] = 1'b0;
            guard = 0;
            while (uo_out[7] !== 1'b1 && guard < 100) begin
                @(posedge clk);
                guard = guard + 1;
            end
            cyc_done = cyc_count;
        end
    endtask

    initial begin
        $dumpfile("out/tb.vcd");
        $dumpvars(0, tb);

        $readmemb("weights.bits", wbits);
        $readmemb("vectors.bits", vecs);

        $display("");
        $display("=== tt_um_pcm_ternary_mlp :: RTL vs golden model ===");

        for (s = 0; s < NSETS; s = s + 1) begin
            do_reset;
            load_weights(s);

            for (v = 0; v < NVEC; v = v + 1) begin
                x          = vecs[s*NVEC + v][17:2];
                expect_cls = vecs[s*NVEC + v][1:0];

                load_input(x);
                run_inference;

                got_cls = uio_out[7:6];

                if (got_cls === expect_cls) begin
                    pass = pass + 1;
                end else begin
                    fail = fail + 1;
                    $display("  MISMATCH set=%0d vec=%0d x=%b exp=%0d got=%0d",
                             s, v, x, expect_cls, got_cls);
                end

                if (s == 0 && v == 0)
                    $display("  inference latency: %0d cycles (start edge to done)",
                             cyc_done - cyc_start);
            end
        end

        $display("");
        $display("  vectors run : %0d", pass + fail);
        $display("  passed      : %0d", pass);
        $display("  failed      : %0d", fail);
        $display("");
        if (fail == 0)
            $display("  RESULT: PASS - RTL matches golden model bit for bit");
        else
            $display("  RESULT: FAIL");
        $display("");

        $finish;
    end

endmodule

`default_nettype wire
