#!/usr/bin/python3

__appname__ = "pyzx_analyze"
__author__ = "Joseph Lunderville <jlunderv@sfu.ca>"
__version__ = "0.1"

import argparse
from dataclasses import dataclass, replace, field
import logging
from pathlib import Path
import time

import pyzx as zx


logger = logging.getLogger(__appname__)


@dataclass
class Args:
    verbose: bool = False
    input: Path | None = None
    output: Path | None = None


arg_parser = argparse.ArgumentParser(
    description="Use PyZX to optimize a quantum circuit"
)
arg_parser.add_argument(
    "-v", "--verbose", action="store_true", help="verbose message output"
)
arg_parser.add_argument("input", metavar="INPUT", type=Path)

args: Args = arg_parser.parse_args(namespace=Args())

logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

args.input = args.input.resolve()

if not args.input.is_file() or args.input.suffix not in [".qc", ".qasm"]:
    arg_parser.error("Input is not a readable .qc or .qasm file.")

logger.info("Loading '%s'", args.input)
c: zx.Circuit = zx.Circuit.load(str(args.input)).to_basic_gates()
tcount = c.tcount()
logger.info("Success: %d T gates", tcount)

print('{ "gates": { "T": %d } }' % (tcount,))
