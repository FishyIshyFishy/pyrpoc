module CustomWrapper (
    input  wire Clk,
    input  wire Reset,
    input  wire [31:0] Sync,

    input  wire signed [15:0] InputA,
    input  wire signed [15:0] InputB,
    input  wire signed [15:0] InputC,
    input  wire signed [15:0] InputD,

    input  wire ExtTrig,

    output reg  signed [15:0] OutputA,
    output reg  signed [15:0] OutputB,
    output reg  signed [15:0] OutputC,
    output reg  signed [15:0] OutputD,

    output wire OutputInterpA,
    output wire OutputInterpB,
    output wire OutputInterpC,
    output wire OutputInterpD,

    input  wire [31:0] Control0,
    input  wire [31:0] Control1,
    input  wire [31:0] Control2,
    input  wire [31:0] Control3,
    input  wire [31:0] Control4,
    input  wire [31:0] Control5,
    input  wire [31:0] Control6,
    input  wire [31:0] Control7,
    input  wire [31:0] Control8,
    input  wire [31:0] Control9,
    input  wire [31:0] Control10,
    input  wire [31:0] Control11,
    input  wire [31:0] Control12,
    input  wire [31:0] Control13,
    input  wire [31:0] Control14,
    input  wire [31:0] Control15
);

    assign OutputInterpA = 1'b0;
    assign OutputInterpB = 1'b0;
    assign OutputInterpC = 1'b0;
    assign OutputInterpD = 1'b0;

    // Divider signals
    reg        start_div;
    wire       div_done;
    wire signed [15:0] div_out;

    // Instantiate sequential signed divider
    SequentialDividerSigned #(.WIDTH(16)) u_div (
        .clk(Clk),
        .reset(Reset),
        .start(start_div),
        .numerator(InputA),
        .denominator(InputB),
        .quotient(div_out),
        .done(div_done)
    );

    // FSM to trigger divider and latch result
    reg busy;
    reg signed [31:0] scaled_result;

    always @(posedge Clk or posedge Reset) begin
        if (Reset) begin
            busy          <= 0;
            start_div     <= 0;
            scaled_result <= 0;
            OutputA       <= 0;
            OutputB       <= 0;
            OutputC       <= 0;
            OutputD       <= 0;
        end else begin
            if (!busy) begin
                start_div <= 1;
                busy      <= 1;
            end else begin
                start_div <= 0;
            end

            if (div_done) begin
                // scale quotient by Control0
                scaled_result <= (div_out * $signed(Control0)) >>> 15;

                // saturate to 16-bit
                if (scaled_result > 32767)
                    OutputA <= 32767;
                else if (scaled_result < -32768)
                    OutputA <= -32768;
                else
                    OutputA <= scaled_result[15:0];

                // passthroughs
                OutputB <= InputA;
                OutputC <= InputB;
                OutputD <= 0;

                busy <= 0;
            end
        end
    end

endmodule
