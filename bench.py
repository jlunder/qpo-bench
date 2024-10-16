#!/usr/bin/python

__appname__ = "bench"
__author__ = "Joseph Lunderville <jlunderv@sfu.ca>"
__version__ = "0.1"


import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
import itertools
import logging
import os
from pathlib import Path
import sys

from ninja import ninja_syntax as ns

logger = logging.getLogger(__appname__)


class Measurable(Enum):
    T_COUNT = auto()
    TIME = auto()


@dataclass(frozen=True, order=True)
class Resource:
    name: str
    qc_res: str | None
    qasm_res: str | None
    qasm3_res: str | None
    skip_subjects: set[str] = field(default_factory=set)


@dataclass(frozen=True, order=True)
class Benchmark:
    name: str
    measurables: set[Measurable]
    resources: list[Resource]
    time_limit: float | None
    memory_limit: float | None
    repeat_count: int


# The TestSubject is almost-but-not-quite the thing that runs the benchmark:
# we use ninja as our build backend, and this class generates syntax that
# instructs it on how to actually conduct the test.
class TestSubject:
    # Write a ninja snippet to run the actual test and collect output
    def emit_goal(
        self,
        writer: ns.Writer,
        output_path: Path,
        bench: Benchmark,
    ):
        pass


@dataclass
class Args:
    verbose: bool = False
    list_suites: bool = False
    suites: list[str] | None = None


def make_arg_parser():
    parser = argparse.ArgumentParser(description="Run a suite of benchmarks")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose message output"
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="output TBM path",
    )
    parser.add_argument(
        "suites", metavar="SUITE", nargs="*", help="list of benchmark suites"
    )

    return parser


arg_parser: argparse.ArgumentParser = make_arg_parser()


# Scan the bench/* folders for resource files i.e. circuits/programs

res_dir = Path("bench/resources")
resource_files = {
    d: {
        os.path.splitext(r)[0]: res_dir / d / r
        for r in os.listdir(res_dir / d)
        if not r.startswith(".")
    }
    for d in os.listdir(res_dir)
    if not d.startswith(".")
}

resource_names = set(itertools.chain(*map(dict.keys, resource_files.values())))
resources = {
    name: Resource(
        name,
        resource_files["qc"].get(name),
        resource_files["qasm"].get(name),
        resource_files["qasm3"].get(name),
    )
    for name in resource_names
}

circuit_tests = [
    "adder_8",
    "barenco_tof_10",
    "barenco_tof_3",
    "barenco_tof_4",
    "barenco_tof_5",
    "csla_mux_3",
    "csum_mux_9",
    "cycle_17_3",
    "fprenorm",
    "gf2^10_mult",
    "gf2^128_mult",
    "gf2^16_mult",
    "gf2^256_mult",
    "gf2^32_mult",
    "gf2^4_mult",
    "gf2^5_mult",
    "gf2^64_mult",
    "gf2^6_mult",
    "gf2^7_mult",
    "gf2^8_mult",
    "gf2^9_mult",
    "grover_5",
    "ham15-high",
    "ham15-low",
    "ham15-med",
    "hwb10",
    "hwb11",
    "hwb12",
    "hwb6",
    "hwb8",
    "mod5_4",
    "mod_adder_1024",
    "mod_adder_1048576",
    "mod_mult_55",
    "mod_red_21",
    "qcla_adder_10",
    "qcla_com_7",
    "qcla_mod_7",
    "qft_4",
    "rc_adder_6",
    "tof_10",
    "tof_3",
    "tof_4",
    "tof_5",
    "vbe_adder_3",
]

program_tests = [
    "grover",
    "if-simple",
    "loop-block",
    "loop-cycle",
    "loop-h",
    "loop-nested",
    "loop-nonlinear",
    "loop-null",
    "loop-simple",
    "loop-swap",
    "reset-simple",
    "rus",
]

benchmarks = {"popl25": None}


suites = Path()


class FeynmanTestSubject:
    def emit_goal(
        self,
        writer: ns.Writer,
        output_path: Path,
        bench: Benchmark,
    ):
        pass


class MlvoqcTestSubject:
    def emit_goal(
        self,
        writer: ns.Writer,
        output_path: Path,
        bench: Benchmark,
    ):
        pass


class QuartzTestSubject:
    def emit_goal(
        self,
        writer: ns.Writer,
        output_path: Path,
        bench: Benchmark,
    ):
        pass


class QuesoTestSubject:
    def emit_goal(
        self,
        writer: ns.Writer,
        output_path: Path,
        bench: Benchmark,
    ):
        pass


class QuizxTestSubject:
    def emit_goal(
        self,
        writer: ns.Writer,
        output_path: Path,
        bench: Benchmark,
    ):
        pass


subjects = {
"feynman": FeynmanTestSubject(),
"mlvoqc": MlvoqcTestSubject(),
"quartz": QuartzTestSubject(),
"queso": QuesoTestSubject(),
"quizx": QuizxTestSubject(),
}


def main(args: Args):
    if args.list_suites:
        pass
    else:
        if len(args.suites) < 1:
            arg_parser.error(
                "specify at least one suite; for a list of valid options, "
                + "use --list-suites"
            )
            sys.exit(2)
    return 0


if __name__ == "__main__":
    try:
        args: Args = arg_parser.parse_args(namespace=Args())
        logger.setLevel(logging.INFO if args.verbose else logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        res = main(args)

        sys.exit(res)

    except KeyboardInterrupt as e:  # Ctrl-C
        raise e

    except SystemExit as e:  # sys.exit()
        raise e

    except Exception as e:
        logger.exception("Failed with exception:")
        sys.exit(3)
