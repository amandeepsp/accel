# accel — RISC-V Custom Function Unit Accelerator

Hardware ML accelerator built around a VexRiscv soft-core with a Custom Function Unit (CFU) running on a LiteX SoC

## Layout

| Directory    | Language       | Purpose                                       |
|-------------|----------------|-----------------------------------------------|
| `hardware/` | Python/Amaranth | CFU RTL — instruction definitions, SIMD MAC4  |
| `firmware/` | Zig            | Bare-metal firmware: UART link protocol, CFU driver |
| `driver/`   | Zig            | Host-side native driver (serial)               |
| `shared/`   | Zig            | Wire protocol + CFU inline asm (firmware & driver) |
| `top.v`     | Verilog (generated) | CFU Verilog output consumed by LiteX       |

## Build And Test On Tang Nano 20K

Prerequisites: `oss-cad-suite` on `PATH`.

```sh
just hw-all
```
