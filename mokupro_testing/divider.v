module SequentialDividerSigned #(
    parameter WIDTH = 16
)(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  start,
    input  wire signed [WIDTH-1:0] numerator,
    input  wire signed [WIDTH-1:0] denominator,
    output reg  signed [WIDTH-1:0] quotient,
    output reg                   done
);
    reg [WIDTH-1:0] num_abs, den_abs;
    reg [WIDTH:0]   remainder;
    reg [WIDTH-1:0] qtemp;
    reg [$clog2(WIDTH+1)-1:0] bit_index;
    reg sign;
    reg busy;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            quotient   <= 0;
            remainder  <= 0;
            qtemp      <= 0;
            bit_index  <= 0;
            busy       <= 0;
            done       <= 0;
            sign       <= 0;
        end else begin
            done <= 0;

            if (start && !busy) begin
                // setup
                sign   <= numerator[WIDTH-1] ^ denominator[WIDTH-1];
                num_abs<= numerator[WIDTH-1] ? -numerator : numerator;
                den_abs<= denominator[WIDTH-1] ? -denominator : denominator;

                remainder <= 0;
                qtemp     <= 0;
                bit_index <= WIDTH;
                busy      <= 1;
            end else if (busy) begin
                remainder <= {remainder[WIDTH-1:0], num_abs[WIDTH-1]};
                num_abs   <= {num_abs[WIDTH-2:0], 1'b0};

                if (remainder >= den_abs) begin
                    remainder <= remainder - den_abs;
                    qtemp     <= {qtemp[WIDTH-2:0], 1'b1};
                end else begin
                    qtemp     <= {qtemp[WIDTH-2:0], 1'b0};
                end

                bit_index <= bit_index - 1;

                if (bit_index == 0) begin
                    quotient <= sign ? -$signed(qtemp) : $signed(qtemp);
                    busy     <= 0;
                    done     <= 1;
                end
            end
        end
    end
endmodule
