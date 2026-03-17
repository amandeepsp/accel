#!/usr/bin/env python3
"""Host-side client helpers for the CFU link protocol.

Supports two transports:
  - SimLink:    spawns litex_sim and talks over stdin/stdout pipes
  - SerialLink: talks to a real FPGA over a serial port

Usage:
    uv run python host/client.py --test                          # sim
    uv run python host/client.py --test --serial /dev/ttyUSB1   # real FPGA (flashes)
    uv run python host/client.py --test --serial /dev/ttyUSB1 --no-upload
    uv run python host/numpy_sim.py                              # library-style NumPy example
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager

from lib.seriallink import SerialLink
from lib.simlink import SimLink

DEFAULT_BAUD = 115200
DEFAULT_CFU = "top.v"
DEFAULT_FIRMWARE = "firmware/zig-out/bin/firmware.bin"


def create_link(
    *,
    serial: str | None = None,
    baud: int = DEFAULT_BAUD,
    cfu: str = DEFAULT_CFU,
    firmware: str = DEFAULT_FIRMWARE,
    verbose: bool = False,
    no_upload: bool = False,
):
    if serial:
        return SerialLink(
            serial,
            firmware=firmware,
            baudrate=baud,
            verbose=verbose,
            no_upload=no_upload,
        )
    return SimLink(cfu=cfu, firmware=firmware, verbose=verbose)


@contextmanager
def open_link(**kwargs):
    link = create_link(**kwargs)
    try:
        link.wait_for_ready()
        yield link
    finally:
        link.close()


def run_tests(link):
    if hasattr(link, "ping"):
        ok = link.ping()
        print(f"\n== Host-side link tests ==\n\n  ping() = {'PASS' if ok else 'FAIL'}")
        if not ok:
            return False
    else:
        print("\n== Host-side link tests ==\n")

    def check(a, b, expected, desc=""):
        result = link.mac4(a, b)
        ok = result == expected
        tag = "PASS" if ok else f"FAIL (got 0x{result:08x}, want 0x{expected:08x})"
        print(f"  mac4(0x{a:08x}, 0x{b:08x}) = 0x{result:08x}  {tag}  {desc}")
        return ok

    passed = total = 0

    for a, b, exp, desc in [
        (0x00000000, 0x00000000, 0x00000000, "zeros"),
        (0x00000000, 0x01010101, 0x00000200, "offset only"),
        (0x01010101, 0x01010101, 0x00000204, "1s * 1s"),
        (0x01020304, 0x04030201, 0x00000514, "mixed"),
    ]:
        total += 1
        if check(a, b, exp, desc):
            passed += 1

    print(
        f"\nResults: {passed}/{total}",
        "ALL PASSED" if passed == total else "SOME FAILED",
    )
    return passed == total


def parse_args():
    parser = argparse.ArgumentParser(description="CFU host client")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--serial", metavar="PORT", help="Serial port for real FPGA (e.g. /dev/ttyUSB1)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip firmware upload (assume already running)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Serial baud rate (default: {DEFAULT_BAUD})",
    )
    parser.add_argument("--cfu", default=DEFAULT_CFU)
    parser.add_argument("--firmware", default=DEFAULT_FIRMWARE)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.test:
        return 0

    with open_link(
        serial=args.serial,
        baud=args.baud,
        cfu=args.cfu,
        firmware=args.firmware,
        verbose=args.verbose,
        no_upload=args.no_upload,
    ) as link:
        return 0 if run_tests(link) else 1


if __name__ == "__main__":
    sys.exit(main())
