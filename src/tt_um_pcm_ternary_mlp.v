/*
 * tt_um_pcm_ternary_mlp
 *
 * A weight-stationary, statically-scheduled ternary-weight MLP inference tile.
 *
 * Architecture notes (the whole pedagogical point):
 *   - All 160 weights live in on-chip flip-flops. They are loaded once over a
 *     serial chain and never fetched again. No memory crosses the die boundary
 *     during inference; only the input vector goes in and a class index comes out.
 *   - Weights are ternary {-1, 0, +1} and activations are binary {0, 1}. The
 *     product of a binary activation and a ternary weight is therefore one of
 *     {-1, 0, +1}, so the "multiply-accumulate" collapses to a conditional
 *     increment/decrement. There is no multiplier anywhere in this design.
 *   - Control is a counter and a 4-state FSM. There is no runtime scheduler and
 *     no data-dependent branching: every inference takes exactly 26 cycles
 *     regardless of input. The schedule is fixed at compile time.
 *
 * Network: 16 binary inputs -> 8 hidden (binarized) -> 4 output classes
 *
 * Author: Phillip Moore
 * License: Apache-2.0
 */

`default_nettype none

module tt_um_pcm_ternary_mlp (
    input  wire [7:0] ui_in,    // dedicated inputs  - activation bytes
    output wire [7:0] uo_out,   // dedicated outputs - 7-segment + done
    input  wire [7:0] uio_in,   // IOs: input path   - control
    output wire [7:0] uio_out,  // IOs: output path  - status + class
    output wire [7:0] uio_oe,   // IOs: enable       - 1 = drive output
    input  wire       ena,      // always 1 when powered - unused
    input  wire       clk,
    input  wire       rst_n
);

    // ---------------------------------------------------------------- params
    localparam N_IN    = 16;
    localparam N_HID   = 8;
    localparam N_OUT   = 4;
    localparam ACC1_W  = 6;                     // signed, holds -16..+16
    localparam ACC2_W  = 5;                     // signed, holds  -8..+8
    localparam W1_BITS = N_IN  * N_HID * 2;     // 256
    localparam W2_BITS = N_HID * N_OUT * 2;     // 64
    localparam W_BITS  = W1_BITS + W2_BITS;     // 320

    // --------------------------------------------------------------- control
    wire w_shift = uio_in[0];   // shift one weight bit in on this clk
    wire w_data  = uio_in[1];   // the weight bit
    wire start   = uio_in[2];   // begin inference
    wire in_load = uio_in[3];   // shift ui_in byte into the activation register

    localparam S_IDLE = 2'd0,
               S_L1   = 2'd1,
               S_ACT  = 2'd2,
               S_L2   = 2'd3;

    reg [1:0] state;
    reg [3:0] step;
    reg       done;

    // --------------------------------------------------- weight shift memory
    // One flat shift register. Bits [W1_BITS-1:0] are layer 1, laid out as 16
    // consecutive 16-bit rows (one row per input, 8 weights of 2 bits each).
    // Bits [W_BITS-1:W1_BITS] are layer 2, as 8 rows of 8 bits.
    reg [W_BITS-1:0] wmem;

    always @(posedge clk) begin
        if (!rst_n)
            wmem <= {W_BITS{1'b0}};
        else if (w_shift)
            wmem <= {wmem[W_BITS-2:0], w_data};
    end

    // ------------------------------------------------- activation input reg
    reg [N_IN-1:0] xreg;

    always @(posedge clk) begin
        if (!rst_n)
            xreg <= {N_IN{1'b0}};
        else if (in_load)
            xreg <= {xreg[N_IN-9:0], ui_in};
    end

    // ------------------------------------------------------- row selection
    // The only "addressing" in the design: pick this cycle's weight row.
    wire [N_HID*2-1:0] w1_row = wmem[ step * (N_HID*2) +: (N_HID*2) ];
    wire [N_OUT*2-1:0] w2_row = wmem[ W1_BITS + step[2:0] * (N_OUT*2) +: (N_OUT*2) ];

    wire x_bit   = xreg[step];
    reg  [N_HID-1:0] hid;
    wire h_bit   = hid[step[2:0]];

    // ------------------------------------------------------- accumulators
    reg [N_HID*ACC1_W-1:0] acc1;
    reg [N_OUT*ACC2_W-1:0] acc2;

    genvar j;

    // Layer 1: eight accumulators step in parallel, one input bit per cycle.
    generate
        for (j = 0; j < N_HID; j = j + 1) begin : gen_l1
            wire sgn = w1_row[j*2];
            wire nz  = w1_row[j*2+1];
            wire en  = (state == S_L1) && x_bit && nz;
            wire signed [ACC1_W-1:0] cur = acc1[j*ACC1_W +: ACC1_W];

            always @(posedge clk) begin
                if (!rst_n)
                    acc1[j*ACC1_W +: ACC1_W] <= {ACC1_W{1'b0}};
                else if (state == S_IDLE && start)
                    acc1[j*ACC1_W +: ACC1_W] <= {ACC1_W{1'b0}};
                else if (en)
                    acc1[j*ACC1_W +: ACC1_W] <= sgn ? cur - 1'b1 : cur + 1'b1;
            end
        end
    endgenerate

    // Layer 2: four accumulators, one hidden bit per cycle.
    generate
        for (j = 0; j < N_OUT; j = j + 1) begin : gen_l2
            wire sgn = w2_row[j*2];
            wire nz  = w2_row[j*2+1];
            wire en  = (state == S_L2) && h_bit && nz;
            wire signed [ACC2_W-1:0] cur = acc2[j*ACC2_W +: ACC2_W];

            always @(posedge clk) begin
                if (!rst_n)
                    acc2[j*ACC2_W +: ACC2_W] <= {ACC2_W{1'b0}};
                else if (state == S_ACT)
                    acc2[j*ACC2_W +: ACC2_W] <= {ACC2_W{1'b0}};
                else if (en)
                    acc2[j*ACC2_W +: ACC2_W] <= sgn ? cur - 1'b1 : cur + 1'b1;
            end
        end
    endgenerate

    // ------------------------------------------- hidden-layer binarization
    // hid[j] = (acc1[j] > 0). Sign bit clear and value nonzero.
    integer k;
    always @(posedge clk) begin
        if (!rst_n) begin
            hid <= {N_HID{1'b0}};
        end else if (state == S_ACT) begin
            for (k = 0; k < N_HID; k = k + 1)
                hid[k] <= ~acc1[k*ACC1_W + ACC1_W-1] & (|acc1[k*ACC1_W +: ACC1_W]);
        end
    end

    // ------------------------------------------------------------- the FSM
    // Fixed schedule: 1 + 16 + 1 + 8 cycles, every time, for every input.
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            step  <= 4'd0;
            done  <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        state <= S_L1;
                        step  <= 4'd0;
                        done  <= 1'b0;
                    end
                end
                S_L1: begin
                    if (step == N_IN - 1) begin
                        state <= S_ACT;
                        step  <= 4'd0;
                    end else begin
                        step <= step + 1'b1;
                    end
                end
                S_ACT: begin
                    state <= S_L2;
                    step  <= 4'd0;
                end
                S_L2: begin
                    if (step == N_HID - 1) begin
                        state <= S_IDLE;
                        done  <= 1'b1;
                    end else begin
                        step <= step + 1'b1;
                    end
                end
            endcase
        end
    end

    // -------------------------------------------------------------- argmax
    // Combinational compare tree over four signed accumulators.
    // Ties resolve to the lowest class index.
    wire signed [ACC2_W-1:0] o0 = acc2[0*ACC2_W +: ACC2_W];
    wire signed [ACC2_W-1:0] o1 = acc2[1*ACC2_W +: ACC2_W];
    wire signed [ACC2_W-1:0] o2 = acc2[2*ACC2_W +: ACC2_W];
    wire signed [ACC2_W-1:0] o3 = acc2[3*ACC2_W +: ACC2_W];

    wire                     a_gt = (o1 > o0);
    wire signed [ACC2_W-1:0] a_val = a_gt ? o1 : o0;
    wire [1:0]               a_idx = a_gt ? 2'd1 : 2'd0;

    wire                     b_gt = (o3 > o2);
    wire signed [ACC2_W-1:0] b_val = b_gt ? o3 : o2;
    wire [1:0]               b_idx = b_gt ? 2'd3 : 2'd2;

    wire [1:0] class_idx = (b_val > a_val) ? b_idx : a_idx;

    // ------------------------------------------------------- 7-segment out
    reg [6:0] seg;
    always @(*) begin
        case (class_idx)
            2'd0:    seg = 7'b0111111;  // 0
            2'd1:    seg = 7'b0000110;  // 1
            2'd2:    seg = 7'b1011011;  // 2
            default: seg = 7'b1001111;  // 3
        endcase
    end

    // --------------------------------------------------------------- pinout
    // Fixed 19 Oct 2026. The previous mapping was
    //   uio_out = {class_idx, 1'b0, done, busy, 3'b000};  uio_oe = 8'b1101_0000;
    // which put done on uio[4], left uio[5] tied low, and placed busy on
    // uio[3] with its output enable OFF - and uio[3] is already an input
    // (activation load enable), so busy could never have been driven there at
    // all. info.yaml documented uio[4] as busy and uio[5] as done, so the
    // published pinout did not describe the silicon. This mapping matches the
    // documentation and makes busy observable.
    //   uio[7:6] class index   uio[5] done   uio[4] busy   uio[3:0] inputs
    assign uo_out  = {done, seg};
    assign uio_out = {class_idx, done, (state != S_IDLE), 4'b0000};
    assign uio_oe  = 8'b1111_0000;

    // silence unused warnings
    wire _unused = &{ena, uio_in[7:4], 1'b0};

endmodule

`default_nettype wire
