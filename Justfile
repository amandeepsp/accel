# accel — RISC-V CFU accelerator task runner

set dotenv-load := false

oss_cad_bin := "/home/amandeeps/Projects/oss-cad-suite/bin"
cfu_rows := "8"
cfu_cols := "8"
cfu_store_depth := "512"
cfu_in_width := "8"
cfu_acc_width := "32"
port := "/dev/ttyUSB1"

# Default: list available recipes
default:
    @just --list

# Generate Verilog CFU from Amaranth
verilog:
    uv run python -m hardware.top \
        --cfu-rows {{ cfu_rows }} \
        --cfu-cols {{ cfu_cols }} \
        --cfu-store-depth {{ cfu_store_depth }} \
        --cfu-in-width {{ cfu_in_width }} \
        --cfu-acc-width {{ cfu_acc_width }}

# Build the LiteX SoC for Tang Nano 20K
hw-build: verilog
    env PATH={{ oss_cad_bin }}:{{ env_var('PATH') }} \
        uv run python -m litex_boards.targets.sipeed_tang_nano_20k \
        --build --toolchain apicula \
        --cpu-type vexriscv --cpu-variant full+cfu --cpu-cfu top.v \
        --cfu-rows {{ cfu_rows }} \
        --cfu-cols {{ cfu_cols }} \
        --cfu-store-depth {{ cfu_store_depth }} \
        --cfu-in-width {{ cfu_in_width }} \
        --with-cfu-led-debug

# Flash the bitstream to the board
hw-flash:
    env PATH={{ oss_cad_bin }}:{{ env_var('PATH') }} \
        uv run python -m litex_boards.targets.sipeed_tang_nano_20k --flash

# Reset the FPGA board
hw-reset:
    env PATH={{ oss_cad_bin }}:{{ env_var('PATH') }} \
        openFPGALoader --board tangnano20k --reset

# Build libaccel.so for host Python binding
libaccel:
    zig build libaccel -Dbuild-dir=build/sipeed_tang_nano_20k

# Build firmware targeting the hardware SoC
hw-firmware build-dir="build/sipeed_tang_nano_20k":
    zig build firmware \
        -Dbuild-dir={{ build-dir }} \
        -Dcfu-rows={{ cfu_rows }} \
        -Dcfu-cols={{ cfu_cols }} \
        -Dcfu-store-depth={{ cfu_store_depth }}

# Upload firmware via serial boot
hw-upload:
    uv run litex_term {{ port }} --kernel zig-out/bin/firmware.bin

# Upload firmware and automatically release the serial port when the transfer completes
hw-upload-once:
    uv run litex-upload-once {{ port }} zig-out/bin/firmware.bin \
        --reset-command 'just hw-reset' \
        --post-boot-timeout 12

# Run the end-to-end GEMM test against a board with firmware running
hw-e2e-gemm:
    env ACCEL_CFU_WORD_BITS=$(({{ cfu_rows }} * {{ cfu_in_width }})) \
        ACCEL_CFU_STORE_DEPTH_WORDS={{ cfu_store_depth }} \
        uv run accel-e2e-gemm {{ port }}

# Run a larger pipelined GEMM that spans multiple K chunks
hw-e2e-gemm-large:
    env ACCEL_CFU_WORD_BITS=$(({{ cfu_rows }} * {{ cfu_in_width }})) \
        ACCEL_CFU_STORE_DEPTH_WORDS={{ cfu_store_depth }} \
        uv run accel-e2e-gemm {{ port }} --m {{ cfu_rows }} --k $((({{ cfu_store_depth }} * 4 / {{ cfu_rows }}) + 128)) --n {{ cfu_cols }}

# JTAG reset → pause → serial-boot firmware → small strict e2e → large strict e2e (one after another).
# Reset clears SDRAM, so we always re-upload before the tests.
hw-e2e-after-reset:
    just hw-reset
    sleep 2
    just hw-upload-once
    just hw-e2e-gemm
    just hw-e2e-gemm-large

# Full hardware flow: build SoC → flash → build firmware → upload
hw-all: hw-build hw-flash hw-firmware hw-upload-once
