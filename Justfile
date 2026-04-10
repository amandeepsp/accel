# accel — RISC-V CFU accelerator task runner

set dotenv-load := false

oss_cad_bin := "/home/amandeeps/Projects/oss-cad-suite/bin"
cfu_rows := "8"
cfu_cols := "8"
cfu_store_depth := "512"
cfu_in_width := "8"
cfu_acc_width := "32"

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

# Run the driver against a real device
driver port="/dev/ttyUSB1":
    zig build run -- {{ port }}

# Build the shared library for the Python wrapper
python-lib:
    zig build python-lib

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

# Build firmware targeting the hardware SoC
hw-firmware build-dir="build/sipeed_tang_nano_20k":
    zig build firmware -Dbuild-dir={{ build-dir }}

# Upload firmware via serial boot
hw-upload port="/dev/ttyUSB1":
    uv run litex_term {{ port }} --kernel zig-out/bin/firmware.bin

# Full hardware flow: build SoC → flash → build firmware → upload
hw-all: hw-build hw-flash hw-firmware hw-upload
