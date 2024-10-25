#!/usr/bin/python3

__appname__ = "bench_pyzx"
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
arg_parser.add_argument("output", metavar="OUTPUT", type=Path)

args: Args = arg_parser.parse_args(namespace=Args())

logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

args.input = args.input.resolve()
args.output = args.output.resolve()

if not args.input.is_file() or args.input.suffix not in [".qc", ".qasm"]:
    arg_parser.error("Input is not a readable .qc or .qasm file.")

if not args.output.parent.is_dir():
    arg_parser.error("Output location is not a directory.")
if args.output.suffix not in [".qc", ".qasm"]:
    arg_parser.error("Output must be named .qc or .qasm.")

logger.info("Loading '%s'", args.input)
c: zx.Circuit = zx.Circuit.load(str(args.input)).to_basic_gates()
logger.info("Success: %d T gates", c.tcount())
# qubits = c.qubits
g = c.to_graph()
start_time = time.time()
g = zx.simplify.teleport_reduce(g)
elapsed = time.time() - start_time
logger.info("After teleport_reduce (%.2fs): %d T gates", elapsed, zx.tcount(g))
c_opt = zx.Circuit.from_graph(g).split_phase_gates().to_basic_gates()
start_time = time.time()
c_opt = zx.optimize.basic_optimization(c_opt).to_basic_gates()
elapsed = time.time() - start_time
logger.info("After basic_optimization: %d T gates", c_opt.tcount())

if args.output:
    if args.output.suffix == ".qasm":
        logger.info("Writing out to '%s' as .qasm", args.output)
        args.output.open("w").write(c_opt.to_qasm())
    else:
        logger.info("Writing out to '%s' as .qc", args.output)
        args.output.open("w").write(c_opt.to_qc())

# if validate:
#     c_id = c.adjoint()
#     c_id.add_circuit(c_opt)
#     g = c_id.to_graph()
#     zx.simplify.full_reduce(g)
#     if g.num_vertices() == 2*len(g.inputs):
#         self.verified = "Y"
#     else: self.verified = "N"
# else: self.verified = "-"
