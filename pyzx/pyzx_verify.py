#!/usr/bin/python3

__appname__ = "pyzx_verify"
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
    ref_circuit: Path | None = None
    xform_circuit: Path | None = None


arg_parser = argparse.ArgumentParser(
    description="Use PyZX to optimize a quantum circuit"
)
arg_parser.add_argument(
    "-v", "--verbose", action="store_true", help="verbose message xform_circuit"
)
arg_parser.add_argument("ref_circuit", metavar="REF", type=Path)
arg_parser.add_argument("xform_circuit", metavar="XFORM", type=Path)

args: Args = arg_parser.parse_args(namespace=Args())

logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

args.ref_circuit = args.ref_circuit.resolve()
args.xform_circuit = args.xform_circuit.resolve()

valid_exts = set(
    [
        ".qc",
        ".qasm",
    ]
)

if not args.ref_circuit.is_file() or args.ref_circuit.suffix not in valid_exts:
    arg_parser.error("Reference is not a readable .qc or .qasm file.")

if not args.xform_circuit.is_file() or args.xform_circuit.suffix not in valid_exts:
    arg_parser.error("Transformed is not a readable .qc or .qasm file.")

args.ref_circuit = args.ref_circuit.resolve()
args.xform_circuit = args.xform_circuit.resolve()

logger.info("Loading reference circuit '%s'", args.ref_circuit)
c_ref: zx.Circuit = zx.Circuit.load(str(args.ref_circuit)).to_basic_gates()
logger.info("Success: %d T gates", c_ref.tcount())

logger.info("Loading transformed circuit '%s'", args.xform_circuit)
c_xform: zx.Circuit = zx.Circuit.load(str(args.xform_circuit)).to_basic_gates()
logger.info("Success: %d T gates", c_xform.tcount())

start_time = time.time()
c_id = c_ref.adjoint()
c_id.add_circuit(c_xform)
g = c_id.to_graph()
zx.simplify.full_reduce(g)
elapsed = time.time() - start_time
verified = g.num_vertices() == 2 * len(g.inputs())

logger.info(
    "Verification graph simplified in %.2fs; %s",
    elapsed,
    "transformation is valid" if verified else "results inconclusive",
)

if verified:
    print('{"verified": true}')
else:
    print('{"verified": false}')
