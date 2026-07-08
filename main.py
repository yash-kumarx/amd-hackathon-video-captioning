#!/usr/bin/env python3
"""AMD Hackathon Track 2 — video captioning entrypoint.

Reads /input/tasks.json, writes /output/results.json, always exits 0
(the failure ladder guarantees a complete, valid output file).
"""
import asyncio
import sys


def main() -> int:
    from pipeline.run import run
    try:
        return asyncio.run(run())
    except Exception as e:  # absolute last line of defense for the exit-0 contract
        print(f"FATAL (suppressed for exit-0 contract): {e}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
