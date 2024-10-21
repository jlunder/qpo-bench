#!/usr/bin/python

# See README.md for all sorts of useful info

__appname__ = "bench"
__author__ = "Joseph Lunderville <jlunderv@sfu.ca>"
__version__ = "0.1"


from datetime import datetime

start_time = datetime.now()


import argparse
from dataclasses import dataclass, replace, field
from enum import Enum, auto
import itertools
import logging
import os
from pathlib import Path
import shutil
from subprocess import Popen
import sys
from typing import Callable

from ninja import ninja_syntax as ns


logger = logging.getLogger(__appname__)


@dataclass
class Args:
    verbose: bool = False
    list_benchmarks: bool = False
    benchmarks: list[str] | None = None

    run_ts: datetime = start_time
    bench_ts: str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_path: Path = Path(os.path.abspath(os.curdir))
    bench_bench: Path = Path("bench")
    bench_build: Path = Path("bench/build")
    # Relative to bench_build!
    bench_root: Path = Path("../..")
    res_dir: Path = Path("bench/resources")

    @staticmethod
    def make_parser():
        parser = argparse.ArgumentParser(description="Run a suite of benchmarks")
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="verbose message output"
        )
        parser.add_argument(
            "--list-benchmarks",
            action="store_true",
            help="output TBM path",
        )
        parser.add_argument(
            "benchmarks",
            metavar="BENCHMARK",
            nargs="*",
            help="list of benchmark suites",
        )

        return parser


args: Args | None = None


class Measurable(Enum):
    T_COUNT = auto()
    TIME = auto()
    MAX_MEMORY = auto()


@dataclass(frozen=True, order=True)
class Resource:
    name: str
    qc_res: str | None
    qasm_res: str | None
    qasm3_res: str | None


@dataclass(frozen=True, order=True)
class BenchmarkConfig:
    name: str  # Should just match the name in the Benchmark
    measurables: set[Measurable]
    resources: set[Resource]
    time_limit: float | None
    memory_limit: float | None
    repeat_count: int


@dataclass(frozen=True, order=True)
class TestResults:
    benchmark_name: str
    subject_name: str
    resource_name: str
    run_id: int
    opt_path: Path | None
    log_path: Path | None
    time_path: Path | None
    analysis_path: Path | None


# The TestSubject is almost-but-not-quite the thing that runs the benchmark:
# we use ninja as our build backend, and this class generates syntax that
# instructs it on how to actually conduct the test.
@dataclass(frozen=True, order=True)
class TestSubject:
    name: str
    subject_path: Path

    @staticmethod
    def make_subject_path(name: str):
        global args
        return Path(
            os.path.normpath(args.run_path / args.bench_build / args.bench_root / name)
        )

    # Write a ninja snippet to run the actual test and collect output; output
    # is a list of with the full paths of the optimized files
    def emit(
        self, writer: ns.Writer, out_path: Path, config: BenchmarkConfig
    ) -> list[TestResults]:
        return []

    @staticmethod
    def emit_time_rule(
        writer: ns.Writer,
        rule_name: str,
        escaped_command: str,
        time_file_var: str,
        **kw_args
    ):
        time_outputs = {
            "user": "%U",
            "system": "%S",
            "elapsed": "%e",
            "avgtext": "%X",
            "avgdata": "%D",
            "avgstack": "%p",
            "maxresident": "%M",
            "swaps": "%W",
            "status": "%x",
        }
        time_outputs_str = ", ".join(
            ['"%s": %s' % (k, v) for k, v in time_outputs.items()]
        )
        writer.rule(
            rule_name,
            "time -f '{"
            + ns.escape(time_outputs_str)
            + "}' -o '"
            + time_file_var
            + "' /bin/sh -c '"
            + escaped_command.replace("'", "'\\''")
            + "'",
            **kw_args
        )


@dataclass(frozen=True, order=True)
class Benchmark:
    name: str  # Should match the name in the config
    subjects: list[TestSubject]
    config: BenchmarkConfig


class FeynmanTestSubject(TestSubject):
    name = "feynman"

    def __init__(self):
        super().__init__(
            name=self.__class__.name,
            subject_path=TestSubject.make_subject_path(self.__class__.name),
        )

    def emit(
        self, writer: ns.Writer, out_path: Path, config: BenchmarkConfig
    ) -> list[TestResults]:
        build_rule = ns.escape("build_" + self.name)
        writer.rule(
            build_rule,
            "cd "
            + ns.escape_path(str(self.subject_path))
            + " && cabal build -O2 feynopt",
        )
        writer.build(self.name, build_rule)

        rule = ns.escape(self.name)
        TestSubject.emit_time_rule(
            writer,
            rule,
            "cd "
            + ns.escape_path(str(self.subject_path))
            + " && cabal exec -v0 feynopt -- -O2 '$in' > '$opt_file' 2> '$log_file'",
            "$time_file",
        )

        rule_qasm3 = ns.escape(self.name + "_qasm3")
        TestSubject.emit_time_rule(
            writer,
            rule_qasm3,
            "cd "
            + ns.escape_path(str(str(self.subject_path)))
            + " && cabal exec -v0 feynopt -- -O2 -qasm3 '$in' > '$opt_file' 2> '$log_file'",
            "$time_file",
        )

        targets = []
        for res in config.resources:
            qc = res.qc_res or res.qasm_res
            if qc:
                rule_used = rule
                input_path = qc
            elif res.qasm3_res:
                rule_used = rule_qasm3
                input_path = res.qasm3_res
            else:
                assert not "neither circuit nor program assigned to this resource?"
            opt_path = out_path / (res.name + "_opt" + os.path.splitext(input_path)[1])
            log_path = out_path / (res.name + "_stderr.log")
            time_path = out_path / (res.name + "_time.json")
            vars = {
                "opt_file": ns.escape_path(str(opt_path)),
                "log_file": ns.escape_path(str(log_path)),
                "time_file": ns.escape_path(str(time_path)),
            }
            writer.build(
                [str(opt_path), str(log_path), str(time_path)],
                rule_used,
                [str(input_path)],
                [self.name],
                variables=vars,
            )
            targets.append(
                TestResults(
                    benchmark_name=config.name,
                    subject_name=self.name,
                    resource_name=res.name,
                    run_id=0,
                    opt_path=opt_path,
                    log_path=log_path,
                    time_path=time_path,
                    analysis_path=None,
                )
            )
        return targets


class MlvoqcTestSubject(TestSubject):
    name = "mlvoqc"

    def __init__(self):
        super().__init__(
            name=self.__class__.name,
            subject_path=TestSubject.make_subject_path(self.__class__.name),
        )

    def emit(
        self, writer: ns.Writer, out_path: Path, config: BenchmarkConfig
    ) -> list[TestResults]:
        bench_bin = str(self.subject_path / "_build/default/bench_voqc.exe")
        build_rule = ns.escape("build_" + self.name)
        writer.rule(
            build_rule,
            "cd "
            + ns.escape_path(str(self.subject_path))
            + " && dune build bench_voqc.exe",
        )
        writer.build(bench_bin, build_rule)

        rule = ns.escape(self.name)
        TestSubject.emit_time_rule(
            writer,
            rule,
            "cd "
            + ns.escape_path(str(self.subject_path))
            + " && " + ns.escape_path(bench_bin) + " -f '$in' -o '$opt_file' 2>&1 > '$log_file'",
            "$time_file",
        )

        targets = []
        for res in config.resources:
            if res.qasm_res:
                rule_used = rule
                input_path = res.qasm_res
            else:
                continue
            opt_path = out_path / (res.name + "_opt" + os.path.splitext(input_path)[1])
            log_path = out_path / (res.name + "_stderr.log")
            time_path = out_path / (res.name + "_time.json")
            vars = {
                "opt_file": ns.escape_path(str(opt_path)),
                "log_file": ns.escape_path(str(log_path)),
                "time_file": ns.escape_path(str(time_path)),
            }
            writer.build(
                [str(opt_path), str(log_path), str(time_path)],
                rule_used,
                [str(input_path)],
                [self.name],
                variables=vars,
            )
            targets.append(
                TestResults(
                    benchmark_name=config.name,
                    subject_name=self.name,
                    resource_name=res.name,
                    run_id=0,
                    opt_path=opt_path,
                    log_path=log_path,
                    time_path=time_path,
                    analysis_path=None,
                )
            )
        return targets
    

"""
./run_quartz circuits/qasm/$X.qasm --eqset quartz/3_2_5_complete_ECC_set.json --output quartz_out/length_simplified_orig$i.qasm
"""


class QuartzTestSubject(TestSubject):
    name = "quartz"

    def __init__(self):
        super().__init__(
            name=self.__class__.name,
            subject_path=TestSubject.make_subject_path(self.__class__.name),
        )

    def emit(self, writer: ns.Writer, output_path: Path, bench: Benchmark) -> list[str]:
        return []


"""
java --enable-preview -cp queso/SymbolicOptimizer-1.0-SNAPSHOT-jar-with-dependencies.jar \
  Applier -c circuits/qasm/$X.qasm -g nam -r queso/rules_q3_s6_nam.txt -sr queso/rules_q3_s6_nam_symb.txt -t $QUESO_TIMEOUT_SEC -o queso_out -j "nam" > queso_out/$X
"""


class QuesoTestSubject(TestSubject):
    name = "queso"

    def __init__(self):
        super().__init__(
            name=self.__class__.name,
            subject_path=TestSubject.make_subject_path(self.__class__.name),
        )

    def emit(self, writer: ns.Writer, output_path: Path, bench: Benchmark) -> list[str]:
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

    def __init__(self):
        super().__init__(
            name=self.__class__.name,
            subject_path=TestSubject.make_subject_path(self.__class__.name),
        )

    def emit(self, writer: ns.Writer, output_path: Path, bench: Benchmark) -> list[str]:
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


subjects: dict[str, TestSubject] = {}

subject_ctors_by_name: dict[str, Callable] = dict(
    [
        (ctor.name, ctor)
        for ctor in [
            FeynmanTestSubject,
            MlvoqcTestSubject,
            # PyzxTestSubject,
            # QuartzTestSubject,
            # QuesoTestSubject,
            # QuizxTestSubject,
            # ToptTestSubject,
            # VvQcoTestSubject,
        ]
    ]
)


def make_subject(name: str) -> TestSubject:
    global subjects, subject_ctors_by_name

    if not name in subjects:
        # The sanity check right now just tests if there's a folder for the subject
        norm_subject_dir = os.path.normpath(
            args.run_path / args.bench_build / args.bench_root / name
        )
        if not os.path.isdir(norm_subject_dir):
            raise Exception(
                "Didn't find subject dir at '%s' (normalized from '%s')"
                % (
                    norm_subject_dir,
                    args.bench_build / args.bench_root / name,
                )
            )
        subjects[name] = subject_ctors_by_name[name]()
    return subjects[name]


resources: dict[str, Resource] = {}


def make_resource(name: str) -> Resource:
    global resources
    if not name in resources:
        norm_res_path = Path(
            os.path.normpath(
                args.run_path / args.bench_build / args.bench_root / args.res_dir
            )
        )
        qc_path = norm_res_path / "qc" / (name + ".qc")
        qasm_path = norm_res_path / "qasm" / (name + ".qasm")
        qasm3_path = norm_res_path / "qasm3" / (name + ".qasm")
        r = Resource(
            name,
            str(qc_path) if os.path.isfile(qc_path) else None,
            str(qasm_path) if os.path.isfile(qasm_path) else None,
            str(qasm3_path) if os.path.isfile(qasm3_path) else None,
        )
        # Sanity check
        if r.qc_res == None and r.qasm3_res == None and r.qasm3_res == None:
            raise Exception(
                "Didn't find any files for resource '%s' in '%s'"
                % (
                    name,
                    args.bench_build / args.bench_root / args.res_dir,
                )
            )
        resources[name] = r
    return resources[name]


def make_benchmark(
    name: str,
    subject_names: list[str],
    measurables: list[Measurable],
    resources: list[str],
    memory_limit: int | None = None,
    time_limit: int | None = None,
    repeat_count: int = 1,
) -> Benchmark:
    return Benchmark(
        name,
        subjects=set(map(make_subject, subject_names)),
        config=BenchmarkConfig(
            name,
            measurables=set(measurables),
            resources=set(map(make_resource, resources)),
            memory_limit=memory_limit,
            time_limit=time_limit,
            repeat_count=repeat_count,
        ),
    )


benchmark_ctors_by_name: dict[str, Callable] = {
    "minimal": lambda: make_benchmark(
        "minimal",
        ["feynman", "mlvoqc"],
        [Measurable.T_COUNT, Measurable.TIME, Measurable.MAX_MEMORY],
        ["qft_4", "tof_4"] + ["if-simple", "loop-simple"],
    ),
    "popl25": lambda: make_benchmark(
        "popl25",
        ["feynman"],
        [Measurable.T_COUNT, Measurable.TIME, Measurable.MAX_MEMORY],
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
        ],
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


def validate_paths(args: Args):
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


def run_benchmark(b: Benchmark):
    global args
    build_path = Path(
        os.path.normpath(args.run_path / args.bench_build / b.name / args.bench_ts)
    )
    logger.info("Run ID '%s', building into folder '%s'", args.bench_ts, build_path)
    if os.path.isdir(build_path):
        raise Exception("Build folder '%s' already exists" % (build_path,))
    try:
        os.makedirs(build_path)
        w = ns.Writer(open(build_path / "build.ninja", "wt"))
        targets: list[TestResults] = []
        for s in b.subjects:
            os.makedirs(build_path / s.name)
            targets.extend(s.emit(w, build_path / s.name, b.config))
        w.build("all", "phony", [str(t.opt_path) for t in targets])
        del w
    except:
        # We didn't get far enough along to bother saving the folder
        shutil.rmtree(build_path, ignore_errors=True)
        raise
    p = Popen(["ninja", "all"], cwd=build_path)
    p.communicate()


def main(args: Args):
    if args.list_benchmarks:
        print("Available benchmark suites:")
        for benchmark in sorted(benchmark_ctors_by_name.keys()):
            print("  " + benchmark)
        return 0

    for benchmark in args.benchmarks:
        run_benchmark(benchmark_ctors_by_name[benchmark]())
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
        validate_paths(args)

        if len(args.benchmarks) < 1:
            arg_parser.error(
                "specify at least one suite; for a list of valid options, "
                + "use --list-benchmarks"
            )
        for benchmark in args.benchmarks:
            if not benchmark in benchmark_ctors_by_name:
                arg_parser.error("unknown suite '" + benchmark + "'")

        res = main(args)

        sys.exit(res)

    except KeyboardInterrupt as e:  # Ctrl-C
        raise e

    except SystemExit as e:  # sys.exit()
        raise e

    except Exception as e:
        logger.exception("Failed with exception:")
        sys.exit(3)
