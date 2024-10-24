#!/usr/bin/python

# See README.md for all sorts of useful info

__appname__ = "bench"
__author__ = "Joseph Lunderville <jlunderv@sfu.ca>"
__version__ = "0.1"


from datetime import datetime

start_time = datetime.now()


import argparse
from dataclasses import dataclass, replace, field
from enum import Enum, StrEnum, auto
import itertools
import logging
import os
from pathlib import Path
import shutil
from subprocess import Popen
import sys
from typing import Callable, Iterable
import csv

from ninja import ninja_syntax as ns


logger = logging.getLogger(__appname__)


@dataclass
class Args:
    verbose: bool = False
    list_benchmarks: bool = False
    benchmarks: list[str] | None = None

    run_ts: datetime = start_time
    bench_ts: str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_path: Path = Path(os.curdir).absolute()
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


class Syntax(StrEnum):
    QC = auto()
    QASM = auto()
    QASM3 = auto()


@dataclass(frozen=True)
class Resource:
    name: str
    qc_res: str | None
    qasm_res: str | None
    qasm3_res: str | None


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str  # Should just match the name in the Benchmark
    measurables: set[Measurable]
    resources: set[Resource]
    time_limit: float | None
    memory_limit: float | None
    repeat_count: int


@dataclass(frozen=True)
class AnalysisResults:
    resource_name: str
    syntax: Syntax
    in_path: Path
    results_path: Path
    log_path: Path
    time_path: Path


@dataclass(frozen=True)
class VerifResults:
    resource_name: str
    syntax: Syntax
    ref_path: Path
    opt_path: Path
    results_path: Path
    log_path: Path
    time_path: Path


@dataclass(frozen=True)
class TestResults:
    benchmark_name: str
    subject_name: str
    resource_name: str
    run_id: int
    syntax: Syntax | None = None
    ref_path: Path | None = None
    opt_path: Path | None = None
    log_path: Path | None = None
    time_path: Path | None = None
    opt_analysis: AnalysisResults | None = None
    verif: VerifResults | None = None


# The TestSubject is almost-but-not-quite the thing that runs the benchmark:
# we use ninja as our build backend, and this class generates syntax that
# instructs it on how to actually conduct the test.
class TestSubject:
    name: str = None

    @property
    def subject_path(self) -> Path:
        global args
        return (args.run_path / self.path).resolve()

    def select_syntax(
        self, c: BenchmarkConfig, r: Resource, t: TestResults
    ) -> TestResults:
        return self.select_any_syntax(c, r, t)

    def select_any_syntax(
        self, c: BenchmarkConfig, r: Resource, t: TestResults
    ) -> TestResults:
        if r.qc_res:
            return replace(t, syntax=Syntax.QC, ref_path=Path(r.qc_res))
        elif r.qasm_res:
            return replace(t, syntax=Syntax.QASM, ref_path=Path(r.qasm_res))
        elif r.qasm3_res:
            return replace(t, syntax=Syntax.QASM3, ref_path=Path(r.qasm3_res))
        else:
            assert not "neither circuit nor program assigned to this resource?"
        return None

    def select_qc_syntax(
        self, c: BenchmarkConfig, r: Resource, t: TestResults
    ) -> TestResults:
        if r.qc_res:
            return replace(t, syntax=Syntax.QC, ref_path=Path(r.qc_res))
        return None

    def select_qasm_syntax(
        self, c: BenchmarkConfig, r: Resource, t: TestResults
    ) -> TestResults:
        if r.qasm_res:
            return replace(t, syntax=Syntax.QASM, ref_path=Path(r.qasm_res))
        return None

    # Write a ninja snippet to run the actual test and collect output; output
    # is a list of results where the formatted output will go
    def emit_test(self, w: ns.Writer, t: TestResults) -> list[TestResults]:
        pass


@dataclass(frozen=True)
class Benchmark:
    name: str  # Should match the name in the config
    subjects: list[TestSubject]
    config: BenchmarkConfig


class FeynmanTestSubject(TestSubject):
    path: Path = Path("feynman")

    opt_params: str

    def __init__(self, opt_params: str):
        self.opt_params = opt_params

    select_syntax = TestSubject.select_any_syntax

    def emit_test(self, w: ns.Writer, t: TestResults):
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_feynopt_qasm3" if t.syntax == Syntax.QASM3 else "bench_feynopt",
            [str(t.ref_path)],
            ["feynopt"],
            variables={
                "opt_params": self.opt_params,
                "opt_file": ns.escape_path(str(t.opt_path)),
                "log_file": ns.escape_path(str(t.log_path)),
                "time_file": ns.escape_path(str(t.time_path)),
            },
        )
        return t

    def emit_analysis(
        self,
        w: ns.Writer,
        resource_name: str,
        out_path: Path,
        syntax: Syntax,
        ref_path: Path,
    ) -> AnalysisResults:
        base = ref_path.stem
        analysis_path = out_path / f"{base}_{syntax}_analysis.json"
        log_path = out_path / f"{base}_{syntax}_analysis.log"
        time_path = out_path / f"{base}_{syntax}_analysis_time.json"
        vars = {
            "analysis_file": ns.escape_path(str(analysis_path)),
            "analysis_log_file": ns.escape_path(str(log_path)),
            "analysis_time_file": ns.escape_path(str(time_path)),
        }
        if syntax == Syntax.QASM3:
            w.build(
                [str(analysis_path), str(log_path), str(time_path)],
                "feyncount_qasm3_analyze",
                [str(ref_path)],
                ["feyncount"],
                variables=vars,
            )
        else:
            w.build(
                [str(analysis_path), str(log_path), str(time_path)],
                "feyncount_analyze",
                [str(ref_path)],
                ["feyncount"],
                variables=vars,
            )
        return AnalysisResults(
            resource_name, syntax, ref_path, analysis_path, log_path, time_path
        )

    def emit_verify(
        self, w: ns.Writer, out_path: Path, to_verify: list[TestResults]
    ) -> list[TestResults]:
        verified_result = []
        for result in to_verify:
            verify_log_path = out_path / (res.name + "_verify.log")
            verify_time_path = out_path / (res.name + "_verify_time.json")
            w.build(
                [str(verify_log_path), str(verify_time_path)],
                "feynver_verify",
                [str(result.opt_path)],
                ["feynver", str(result.ref_path)],
                variables={
                    "ref_file": ns.escape_path(str(result.ref_path)),
                    "log_file": ns.escape_path(str(verify_log_path)),
                    "time_file": ns.escape_path(str(verify_time_path)),
                },
            )
            verified_result.append(
                replace(
                    result,
                    verify_log_path=verify_log_path,
                    verify_time_path=verify_time_path,
                )
            )


class MlvoqcTestSubject(TestSubject):
    path: Path = Path("mlvoqc")

    @property
    def bench_bin_path(self) -> Path:
        return self.subject_path / "_build/default/bench_voqc.exe"

    select_syntax = TestSubject.select_qasm_syntax

    def emit_test(self, w: ns.Writer, t: TestResults) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_mlvoqc",
            [str(t.ref_path)],
            [str(self.bench_bin_path)],
            variables={
                "opt_file": ns.escape_path(str(t.opt_path)),
                "log_file": ns.escape_path(str(t.log_path)),
                "time_file": ns.escape_path(str(t.time_path)),
            },
        )
        return t


class QuartzTestSubject(TestSubject):
    path: Path = Path("quartz")

    @property
    def bench_quartz_bin_path(self) -> Path:
        return self.subject_path / "build/bench_quartz"

    @property
    def bench_quartz_ecc_set_path(self) -> Path:
        return self.subject_path / "eccset/Clifford_T_5_3_complete_ECC_set.json"

    select_syntax = TestSubject.select_qasm_syntax

    def emit_test(self, w: ns.Writer, t: TestResults) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_quartz",
            [str(t.ref_path)],
            [str(self.bench_quartz_bin_path), str(self.bench_quartz_ecc_set_path)],
            variables={
                "opt_file": ns.escape_path(str(t.opt_path)),
                "log_file": ns.escape_path(str(t.log_path)),
                "time_file": ns.escape_path(str(t.time_path)),
            },
        )
        return t


class QuesoTestSubject(TestSubject):
    path: Path = Path("queso")

    select_syntax = TestSubject.select_qc_syntax

    def emit_test(self, w: ns.Writer, t: TestResults) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_voqc",
            [str(t.ref_path)],
            [str(self.bench_bin_path)],
            variables={
                "opt_file": ns.escape_path(str(t.opt_path)),
                "log_file": ns.escape_path(str(t.log_path)),
                "time_file": ns.escape_path(str(t.time_path)),
            },
        )
        return t


class QuizxTestSubject(TestSubject):
    path: Path = Path("quizx")

    select_syntax = TestSubject.select_qc_syntax

    def emit_test(self, w: ns.Writer, t: TestResults) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_voqc",
            [str(t.ref_path)],
            [str(self.bench_bin_path)],
            variables={
                "opt_file": ns.escape_path(str(t.opt_path)),
                "log_file": ns.escape_path(str(t.log_path)),
                "time_file": ns.escape_path(str(t.time_path)),
            },
        )
        return t


subjects: dict[str, TestSubject] = {}

subject_ctors_by_name: dict[str, Callable] = {
    "feynman": (lambda: FeynmanTestSubject("-O2")),
    "feynman-apf": (lambda: FeynmanTestSubject("-apf")),
    "feynman-ppf": (lambda: FeynmanTestSubject("-ppf")),
    "mlvoqc": MlvoqcTestSubject,
    # "pyzx": PyzxTestSubject,
    "quartz": QuartzTestSubject,
    "queso": QuesoTestSubject,
    "quizx": QuizxTestSubject,
    # "topt": ToptTestSubject,
    # "vv-qco": VvQcoTestSubject,
}


def make_subject(name: str) -> TestSubject:
    global subjects, subject_ctors_by_name

    s = subjects.get(name)
    if s == None:
        s: TestSubject = subject_ctors_by_name[name]()
        s.name = name
        # The sanity check right now just tests if there's a folder for the subject
        if not s.subject_path.is_dir():
            raise Exception(f"Didn't find subject dir at '{s.subject_path}')")
        subjects[name] = s
    return s


resources: dict[str, Resource] = {}


def make_resource(name: str) -> Resource:
    global resources
    if not name in resources:
        norm_res_path = (args.run_path / args.res_dir).resolve()
        qc_path = norm_res_path / "qc" / f"{name}.qc"
        qasm_path = norm_res_path / "qasm" / f"{name}.qasm"
        qasm3_path = norm_res_path / "qasm3" / f"{name}.qasm"
        r = Resource(
            name,
            str(qc_path) if qc_path.is_file() else None,
            str(qasm_path) if qasm_path.is_file() else None,
            str(qasm3_path) if qasm3_path.is_file() else None,
        )
        # Sanity check
        if r.qc_res == None and r.qasm3_res == None and r.qasm3_res == None:
            raise Exception(
                f"Didn't find any files for resource '{name}' in "
                + f"'{args.bench_build / args.bench_root / args.res_dir}'"
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
        ["feynman", "feynman-ppf", "mlvoqc", "quartz"],
        [Measurable.T_COUNT, Measurable.TIME, Measurable.MAX_MEMORY],
        ["qft_4", "tof_4", "mod_adder_1024"] + ["if-simple", "loop-simple"],
    ),
    "popl25": lambda: make_benchmark(
        "popl25",
        [
            "feynman",
            "feynman-apf",
            "feynman-ppf",
            "mlvoqc",
            "quartz",  # ,"queso","pyzx", "vv-qco",
        ],
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
    if not (args.run_path / args.bench_build).is_dir():
        logger.info("Didn't find bench_build '%s'", args.run_path / args.bench_build)
        alt_run_path = Path(sys.argv[0]).parent.resolve()
        if (alt_run_path / args.bench_build).is_dir():
            logger.warning(
                "Don't seem to be running from bench project "
                + "root, using argv[0] root '%s' instead of CWD",
                alt_run_path,
            )
            args.run_path = alt_run_path


def validate_paths(args: Args):
    logger.info("Checking for bench_bench '%s'", args.run_path / args.bench_bench)
    if not (args.run_path / args.bench_bench).is_dir():
        raise Exception(f"Didn't find bench dir at '{args.bench_bench}'")
    norm_bench_root = args.run_path.resolve()
    if not norm_bench_root.is_dir():
        raise Exception(f"Didn't find bench project root dir at '{args.bench_root}'")
    logger.info("Real bench_root is '%s'", norm_bench_root)


@dataclass(frozen=True)
class DataRow:
    benchmark: str
    subject: str
    resource: str
    syntax: str
    ref_t_gates: int
    t_gates: int
    user_time: float
    sys_time: float
    elapsed_time: float
    max_resident: int
    status: int


import json


def read_analysis_results(analysis_path: Path) -> int | None:
    t_gates = None
    try:
        # This is carefully set up to explode spectacularly if the file is
        # empty or otherwise missing expected stuff, but not having the T
        # gate entry is normal if there are 0 T's, so that's defaulted
        res = json.load(open(analysis_path, "r"))
        t_gates = int(res["gates"].get("T", 0))
    except:
        pass
    return t_gates


def read_time_results(time_path: Path):
    user_time = None
    sys_time = None
    elapsed_time = None
    max_resident = None
    status = None
    try:
        res = json.load(open(time_path, "r"))
        user_time = float(res["user"])
        sys_time = float(res["system"])
        elapsed_time = float(res["elapsed"])
        max_resident = int(res["maxresident"])
        status = int(res["status"])
    except:
        pass
    return (user_time, sys_time, elapsed_time, max_resident, status)


def run_benchmark(b: Benchmark):
    global args
    build_path = (args.run_path / args.bench_build / b.name / args.bench_ts).resolve()
    logger.info("Run ID '%s', building into folder '%s'", args.bench_ts, build_path)
    if build_path.is_dir():
        raise Exception(f"Build folder '{build_path}' already exists")
    try:
        # Make Ninja build file
        os.makedirs(build_path)
        w = ns.Writer(open(build_path / "build.ninja", "wt"))
        # Include dependency targets
        w.variable(
            "bench_root",
            ns.escape_path(str(args.run_path.resolve())),
        )
        w.variable(
            "bench_bench",
            ns.escape_path(str((args.run_path / args.bench_bench).resolve())),
        )
        w.variable("build_path", ns.escape_path(str(build_path)))
        w.include((args.run_path / args.bench_bench / "ninja/common.ninja").resolve())

        # Add optimization build targets (this is the actual test runs)
        tests: list[TestResults] = []
        for s in b.subjects:
            out_path = build_path / s.name
            os.makedirs(out_path)
            for r in b.config.resources:
                t = TestResults(
                    run_id=0,
                    benchmark_name=b.name,
                    subject_name=s.name,
                    resource_name=r.name,
                )
                t = s.select_syntax(b.config, r, t)
                if t == None:
                    continue
                if b.config.repeat_count == 1:
                    t = replace(
                        t,
                        opt_path=out_path / f"{r.name}_opt{t.ref_path.suffix}",
                        log_path=out_path / f"{r.name}_opt.log",
                        time_path=out_path / f"{r.name}_opt_time.json",
                    )
                    tests.append(s.emit_test(w, t))
                else:
                    for i in range(b.config.repeat_count):
                        t = replace(
                            t,
                            run_id=i,
                            opt_path=out_path
                            / f"{r.name}_opt_{t.run_id}{t.ref_path.suffix}",
                            log_path=out_path / f"{r.name}_opt_{t.run_id}.log",
                            time_path=out_path / f"{r.name}_opt_{t.run_id}_time.json",
                        )
                        tests.append(s.emit_test(w, t))

        # Figure out which resources (refs) are used by the tests, and
        # add analysis targets for them -- we do this as a separate step
        # because we don't want to duplicate the ref analysis, typically one
        # ref analysis will be compared against multiple different
        # optimizations
        ref_build_path = build_path / "ref"
        os.makedirs(ref_build_path)
        a_sub: FeynmanTestSubject = make_subject("feynman")
        refs_analysis: list[AnalysisResults] = [
            a_sub.emit_analysis(w, resource_name, ref_build_path, syntax, ref_path)
            for ref_path, resource_name, syntax in sorted(
                set(((t.ref_path, t.resource_name, t.syntax) for t in tests))
            )
        ]

        # Make analysis targets for test results and annotate the test
        # results with them
        tests = [
            replace(
                t,
                opt_analysis=a_sub.emit_analysis(
                    w, t.resource_name, t.opt_path.parent, t.syntax, t.opt_path
                ),
            )
            for t in tests
        ]

        w.build(
            "all",
            "phony",
            [str(t.opt_path) for t in tests]
            + [
                str(t.opt_analysis.results_path)
                for t in tests
                if t.opt_analysis != None
            ]
            + [str(t.verif.results_path) for t in tests if t.verif != None]
            + [str(a.results_path) for a in refs_analysis],
        )
        del w
    except:
        # We didn't get far enough along to bother saving the folder
        shutil.rmtree(build_path, ignore_errors=True)
        raise

    # This is the main event! Now that the build is prepared, run Ninja
    p = Popen(["ninja", "all"], cwd=build_path)
    p.communicate()

    # The rest of this function is just parsing and collating all the loose
    # JSON, and formatting that as a (somewhat denormalized) CSV.
    rows: list[DataRow] = []
    ref_rows: dict[tuple[str, str], tuple[DataRow, AnalysisResults]] = {}

    for a in refs_analysis:
        t_gates = read_analysis_results(a.results_path)
        (_, _, _, _, status) = read_time_results(a.time_path)
        r = DataRow(
            b.name,
            "ref",
            a.resource_name,
            a.syntax,
            None,
            t_gates,
            None,
            None,
            None,
            None,
            status,
        )
        ref_rows[(a.resource_name, a.syntax)] = (r, a)
        rows.append(r)
    for t in tests:
        ref_t_gates = None
        ref = ref_rows.get((t.resource_name, t.syntax), None)
        if ref != None:
            ref_t_gates = ref[0].t_gates
        t_gates = read_analysis_results(
            t.opt_analysis.results_path if t.opt_analysis != None else None
        )
        (user_time, sys_time, elapsed_time, max_resident, status) = read_time_results(
            t.time_path
        )
        rows.append(
            DataRow(
                t.benchmark_name,
                t.subject_name,
                t.resource_name,
                t.syntax,
                ref_t_gates,
                t_gates,
                user_time,
                sys_time,
                elapsed_time,
                max_resident,
                status,
            )
        )

    cw = csv.writer(open(build_path / f"{b.name}_{args.bench_ts}.csv", "w"))
    cw.writerow(
        [
            "benchmark",
            "subject",
            "resource",
            "syntax",
            "reference t gates",
            "t gates",
            "user time (s)",
            "sys time (s)",
            "elapsed time (s)",
            "max resident (kiB)",
            "exit status",
        ]
    )
    for r in rows:
        cw.writerow(
            [
                r.benchmark,
                r.subject,
                r.resource,
                r.syntax,
                r.ref_t_gates,
                r.t_gates,
                r.user_time,
                r.sys_time,
                r.elapsed_time,
                r.max_resident,
                r.status,
            ]
        )
    del cw


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
                arg_parser.error(f"unknown suite '{benchmark}'")

        res = main(args)

        sys.exit(res)

    except KeyboardInterrupt as e:  # Ctrl-C
        raise e

    except SystemExit as e:  # sys.exit()
        raise e

    except Exception as e:
        logger.exception("Failed with exception:")
        sys.exit(3)
