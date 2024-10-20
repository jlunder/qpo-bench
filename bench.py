#!/usr/bin/python

# See README.md for all sorts of useful info

__appname__ = "bench"
__author__ = "Joseph Lunderville <jlunderv@sfu.ca>"
__version__ = "0.1"


from datetime import datetime

start_time = datetime.now()


import argparse
from dataclasses import dataclass, replace
from enum import Enum, auto
import itertools
import logging
import os
from pathlib import Path
import shutil
from subprocess import Popen
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


@dataclass(frozen=True, order=True)
class Benchmark:
    name: str
    subjects: set[str]
    measurables: set[Measurable]
    resources: set[str]
    time_limit: float | None
    memory_limit: float | None
    repeat_count: int


@dataclass
class Args:
    verbose: bool = False
    list_suites: bool = False
    suites: list[str] | None = None

    run_ts: datetime = start_time
    run_id: str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_path: Path = Path(os.path.abspath(os.curdir))
    bench_bench: Path = Path("bench")
    bench_build: Path = Path("bench/build")
    # Relative to bench_build!
    bench_root: Path = Path("../..")
    res_dir: Path = Path("bench/resources")
    all_resources: dict[str, Resource] = None

    @staticmethod
    def make_parser():
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


@dataclass
class TestResultFiles:
    suite: str
    subject: str
    resource: str
    test: str
    run_id: int
    output: Path
    run_log: Path
    usage: Path


# The TestSubject is almost-but-not-quite the thing that runs the benchmark:
# we use ninja as our build backend, and this class generates syntax that
# instructs it on how to actually conduct the test.
class TestSubject:
    name: str

    # Write a ninja snippet to run the actual test and collect output; output
    # is a list of with the full paths of the optimized files
    def emit(
        self, args: Args, writer: ns.Writer, output_path: Path, bench: Benchmark
    ) -> list[str]:
        return []


class FeynmanTestSubject(TestSubject):
    name = "feynman"

    def emit(
        self, args: Args, writer: ns.Writer, output_path: Path, bench: Benchmark
    ) -> list[str]:
        subject_root = Path(
            os.path.normpath(
                args.run_path / args.bench_build / args.bench_root / self.name
            )
        )

        rule = ns.escape(self.name)

        build_rule = ns.escape("build_" + self.name)
        writer.rule(
            build_rule,
            "cd "
            + ns.escape_path(str(subject_root))
            + " && cabal build -v0 -O2 feynopt",
        )

        writer.build(self.name, build_rule)

        writer.rule(
            rule,
            "cd "
            + ns.escape_path(str(subject_root))
            + " && cabal exec -v0 -O2 feynopt -- -O2 $in > $out",
        )

        rule_qasm3 = ns.escape(self.name + "_qasm3")
        writer.rule(
            rule_qasm3,
            "cd "
            + ns.escape_path(str(subject_root))
            + " && cabal exec -v0 -O2 feynopt -- -O2 -qasm3 $in > $out",
        )

        targets = []
        for r in bench.resources:
            res = args.all_resources[r]
            qc = res.qc_res or res.qasm_res
            qp = res.qasm3_res
            if qc:
                out = os.path.normpath(
                    args.run_path / output_path / (r + "_opt" + os.path.splitext(qc)[1])
                )
                targets.append(out)
                writer.build([str(out)], rule, [str(qc)], [self.name])
            elif qp:
                out = os.path.normpath(
                    args.run_path / output_path / (r + "_opt" + os.path.splitext(qp)[1])
                )
                targets.append(out)
                writer.build([str(out)], rule_qasm3, [str(qp)], [self.name])
        return targets


"""
./run_voqc -f circuits/qasm/length_simplified_orig$i.qasm -o voqc_out/length_simplified_orig$i.qasm
"""


class MlvoqcTestSubject(TestSubject):
    name = "mlvoqc"

    def emit(
        self, args: Args, writer: ns.Writer, output_path: Path, bench: Benchmark
    ) -> list[str]:
        return []


"""
./run_quartz circuits/qasm/$X.qasm --eqset quartz/3_2_5_complete_ECC_set.json --output quartz_out/length_simplified_orig$i.qasm
"""


class QuartzTestSubject(TestSubject):
    name = "quartz"

    def emit(
        self, args: Args, writer: ns.Writer, output_path: Path, bench: Benchmark
    ) -> list[str]:
        return []


"""
java --enable-preview -cp queso/SymbolicOptimizer-1.0-SNAPSHOT-jar-with-dependencies.jar \
  Applier -c circuits/qasm/$X.qasm -g nam -r queso/rules_q3_s6_nam.txt -sr queso/rules_q3_s6_nam_symb.txt -t $QUESO_TIMEOUT_SEC -o queso_out -j "nam" > queso_out/$X
"""


class QuesoTestSubject(TestSubject):
    name = "queso"

    def emit(
        self, args: Args, writer: ns.Writer, output_path: Path, bench: Benchmark
    ) -> list[str]:
        subject_root = Path(
            os.path.normpath(
                args.run_path / args.bench_build / args.bench_root / self.name
            )
        )

        rule = ns.escape(self.name)
        queso_jar = str(subject_root / "target/QUESO-1.0-jar-with-dependencies.jar")

        build_rule = ns.escape("build_" + self.name)
        writer.rule(
            build_rule,
            "cd "
            + ns.escape_path(str(subject_root))
            + " && mvn package -Dmaven.test.skip",
        )

        writer.build(queso_jar, build_rule)

        nam_txt = str(subject_root / "rules_q3_s6_nam.txt")
        nam_symb_txt = str(subject_root / "rules_q3_s6_nam_symb.txt")
        writer.rule(
            rule,
            "cd "
            + ns.escape_path(str(subject_root))
            + " && cabal exec -v0 -O2 feynopt -- -O2 $in > $out",
        )

        rule_qasm3 = ns.escape(self.name + "_qasm3")
        writer.rule(
            rule_qasm3,
            "cd "
            + ns.escape_path(str(subject_root))
            + " && cabal exec -v0 -O2 feynopt -- -O2 -qasm3 $in > $out",
        )

        targets = []
        for r in bench.resources:
            res = args.all_resources[r]
            qc = res.qc_res or res.qasm_res
            qp = res.qasm3_res
            if qc:
                out = os.path.normpath(
                    args.run_path / output_path / (r + "_opt" + os.path.splitext(qc)[1])
                )
                targets.append(out)
                writer.build([str(out)], rule, [str(qc)], [self.name])
            elif qp:
                out = os.path.normpath(
                    args.run_path / output_path / (r + "_opt" + os.path.splitext(qp)[1])
                )
                targets.append(out)
                writer.build([str(out)], rule_qasm3, [str(qp)], [self.name])
        return targets


"""
./run_quizx circuits/qasm/$X.qasm > quizx_out/$X.qasm
"""


class QuizxTestSubject(TestSubject):
    name = "quizx"

    def emit(
        self, args: Args, writer: ns.Writer, output_path: Path, bench: Benchmark
    ) -> list[str]:
        subject_root = Path(
            os.path.normpath(
                args.run_path / args.bench_build / args.bench_root / self.name
            )
        )

        rule = ns.escape(self.name)

        build_rule = ns.escape("build_" + self.name)
        writer.rule(
            build_rule,
            "cd "
            + ns.escape_path(str(subject_root))
            + " && cabal build -v0 -O2 feynopt",
        )

        writer.build(self.name, build_rule)

        writer.rule(
            rule,
            "cd "
            + ns.escape_path(str(subject_root))
            + " && cabal exec -v0 -O2 feynopt -- -O2 $in > $out",
        )

        rule_qasm3 = ns.escape(self.name + "_qasm3")
        writer.rule(
            rule_qasm3,
            "cd "
            + ns.escape_path(str(subject_root))
            + " && cabal exec -v0 -O2 feynopt -- -O2 -qasm3 $in > $out",
        )

        targets = []
        for r in bench.resources:
            res = args.all_resources[r]
            qc = res.qc_res or res.qasm_res
            qp = res.qasm3_res
            if qc:
                out = os.path.normpath(
                    args.run_path / output_path / (r + "_opt" + os.path.splitext(qc)[1])
                )
                targets.append(out)
                writer.build([str(out)], rule, [str(qc)], [self.name])
            elif qp:
                out = os.path.normpath(
                    args.run_path / output_path / (r + "_opt" + os.path.splitext(qp)[1])
                )
                targets.append(out)
                writer.build([str(out)], rule_qasm3, [str(qp)], [self.name])
        return targets


subjects: dict[str, TestSubject] = {
    subject.name: subject
    for subject in [
        FeynmanTestSubject(),
        MlvoqcTestSubject(),
        # PyzxTestSubject(),
        QuartzTestSubject(),
        QuesoTestSubject(),
        QuizxTestSubject(),
        # ToptTestSubject(),
        # VvQcoTestSubject(),
    ]
}


all_circuits = [
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

all_programs = [
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


benchmark_suites = {
    "minimal": Benchmark(
        "minimal",
        subjects=set(["feynman"]),
        measurables=set([Measurable.T_COUNT, Measurable.TIME]),
        resources=set(["qft_4", "tof_4"] + ["if-simple", "loop-simple"]),
        memory_limit=None,
        time_limit=None,
        repeat_count=1,
    ),
    "popl25": Benchmark(
        "popl25",
        subjects=set(["feynman"]),
        measurables=set([Measurable.T_COUNT, Measurable.TIME]),
        resources=set(
            [
                "grover_5",
                "mod5_4",
                "vbe_adder_3",
                "csla_mux_3",
                "csum_mux_9",
                "qcla_com_7",
                "qcla_mod_7",
                "qcla_adder_10",
                "adder_8",
                "rc_adder_6",
                "mod_red_21",
                "mod_mult_55",
                "mod_adder_1024",
                "gf2^4_mult",
                "gf2^5_mult",
                "gf2^6_mult",
                "gf2^7_mult",
                "gf2^8_mult",
                "gf2^9_mult",
                "gf2^10_mult",
                "gf2^16_mult",
                "gf2^32_mult",
                "ham15-low",
                "ham15-med",
                "ham15-high",
                "hwb6",
                "qft_4",
                "tof_3",
                "tof_4",
                "tof_5",
                "tof_10",
                "barenco_tof_3",
                "barenco_tof_4",
                "barenco_tof_5",
                "barenco_tof_10",
            ]
            + [
                "fprenorm",
            ]
            + [
                "rus",
                "grover",
                "reset-simple",
                "if-simple",
                "loop-simple",
                "loop-h",
                "loop-nested",
                "loop-swap",
                "loop-nonlinear",
                "loop-null",
            ]
        ),
        memory_limit=None,
        time_limit=None,
        repeat_count=1,
    ),
}


def detect_run_path(args: Args):
    if not os.path.isdir(args.run_path / args.bench_build):
        logger.info("Didn't find bench_build '%s'", args.run_path / args.bench_build)
        alt_run_path = Path(os.path.abspath(os.path.dirname(sys.argv[0])))
        if os.path.isdir(alt_run_path / args.bench_build):
            logger.warning(
                "Don't seem to be running from bench project "
                + "root, using argv[0] root '%s' instead of CWD",
                alt_run_path,
            )
            args.run_path = alt_run_path


def scan_resources(args: Args):
    logger.info("Checking for bench_bench '%s'", args.run_path / args.bench_bench)
    if not os.path.isdir(args.run_path / args.bench_bench):
        raise Exception("Didn't find bench dir at '%s'" % (args.bench_bench,))
    norm_bench_root = Path(
        os.path.normpath(args.run_path / args.bench_build / args.bench_root)
    )
    if not os.path.isdir(norm_bench_root):
        raise Exception(
            "Didn't find bench project root dir at '%s'" % (args.bench_root,)
        )
    logger.info("Real bench_root is '%s'", norm_bench_root)

    # Make a shortlist of subjects which are actually required for this
    # particular suite before doing sanity checking -- this is QOL for
    # development, if a subject is broken because it's not implemented, you
    # probably would like to still be able to test other subjects
    subjects_used = set(
        sum([list(benchmark_suites[s].subjects) for s in args.suites], [])
    )
    # The sanity check right now just tests if there's a folder there
    logger.info("Looking for actually used subjects [%s]" % (", ".join(subjects_used),))
    for s in subjects_used:
        if not os.path.isdir(norm_bench_root / s):
            raise Exception("Didn't find subject dir at '%s'" % (args.bench_root / s,))

    # Scan the bench/* folders for resource files i.e. circuits/programs
    norm_res_dir = norm_bench_root / args.res_dir

    def list_resources(path: Path, ext_filter: str):
        if not os.path.isdir(path):
            logger.warning("Didn't find expected resource dir '%s'", path)
            return
        for filename in os.listdir(path):
            filename: str
            if not filename.startswith("."):
                base, ext = os.path.splitext(filename)
                full_path = path / filename
                if os.path.isfile(full_path) and ext == ext_filter:
                    yield base, full_path

    resource_files = {
        d: {n: p for n, p in list_resources(norm_res_dir / d, ext)}
        for d, ext in {"qc": ".qc", "qasm": ".qasm", "qasm3": ".qasm"}.items()
    }
    resource_names = set(itertools.chain(*map(dict.keys, resource_files.values())))
    args.all_resources = {
        name: Resource(
            name,
            resource_files["qc"].get(name),
            resource_files["qasm"].get(name),
            resource_files["qasm3"].get(name),
        )
        for name in resource_names
    }
    if len(args.all_resources) > 0:
        logger.info("Loaded resources:")
        for n, r in args.all_resources.items():
            logger.info("  %s: %s", n, r)
    else:
        logger.warning("Didn't find any resources!")


def run_benchmark(args: Args, b: Benchmark):
    build_folder = args.bench_build / b.name / args.run_id
    logger.info("Run ID '%s', building into folder '%s'", args.run_id, build_folder)
    if os.path.isdir(build_folder):
        raise Exception("Build folder '%s' already exists" % (build_folder,))
    try:
        os.makedirs(build_folder)
        w = ns.Writer(open(build_folder / "build.ninja", "wt"))
        targets = []
        for s in b.subjects:
            os.makedirs(build_folder / s)
            targets.extend(subjects[s].emit(args, w, build_folder / s, b))
        del w
    except:
        # We didn't get far enough along to bother saving the folder
        shutil.rmtree(build_folder, ignore_errors=True)
        raise
    p = Popen(["ninja"] + targets, cwd=build_folder)
    p.communicate()


def main(args: Args):
    if args.list_suites:
        print("Available benchmark suites:")
        for suite in sorted(benchmark_suites.keys()):
            print("  " + suite)
        return 0

    for suite in args.suites:
        run_benchmark(args, benchmark_suites[suite])
    return 0


if __name__ == "__main__":
    arg_parser: argparse.ArgumentParser = Args.make_parser()

    try:
        args: Args = arg_parser.parse_args(namespace=Args())

        logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        detect_run_path(args)
        scan_resources(args)

        if len(args.suites) < 1:
            arg_parser.error(
                "specify at least one suite; for a list of valid options, "
                + "use --list-suites"
            )
        for suite in args.suites:
            if not suite in benchmark_suites:
                arg_parser.error("unknown suite '" + suite + "'")

        res = main(args)

        sys.exit(res)

    except KeyboardInterrupt as e:  # Ctrl-C
        raise e

    except SystemExit as e:  # sys.exit()
        raise e

    except Exception as e:
        logger.exception("Failed with exception:")
        sys.exit(3)
