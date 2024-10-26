#!/usr/bin/python3

# See README.md for all sorts of useful info

__appname__ = "bench"
__author__ = "Joseph Lunderville <jlunderv@sfu.ca>"
__version__ = "0.1"


from datetime import datetime

start_ts = datetime.now()


import argparse
from dataclasses import dataclass, replace
from enum import Enum, StrEnum, auto
from itertools import product
import logging
import os
from pathlib import Path
import shutil
from subprocess import Popen, PIPE
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

    bench_ts: str = start_ts.strftime("%Y-%m-%d_%H-%M-%S")
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
    QASM_CCZ = auto()
    QASM3 = auto()


@dataclass(frozen=True)
class Resource:
    name: str
    qc_res: str | None
    qasm_res: str | None
    qasm_ccz_res: str | None
    qasm3_res: str | None


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str  # Should just match the name in the Benchmark
    measurables: set[Measurable]
    resources: set[Resource]
    time_limit_s: float | None
    memory_limit_k: float | None
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
        if False:
            pass
        elif r.qc_res:
            return replace(t, syntax=Syntax.QC, ref_path=Path(r.qc_res))
        elif r.qasm_res:
            return replace(t, syntax=Syntax.QASM, ref_path=Path(r.qasm_res))
        elif r.qasm3_res:
            return replace(t, syntax=Syntax.QASM3, ref_path=Path(r.qasm3_res))
        else:
            assert not "neither circuit nor program assigned to this resource?"
        return None

    def select_circuit_syntax(
        self, c: BenchmarkConfig, r: Resource, t: TestResults
    ) -> TestResults:
        if r.qc_res:
            return replace(t, syntax=Syntax.QC, ref_path=Path(r.qc_res))
        elif r.qasm_res:
            return replace(t, syntax=Syntax.QASM, ref_path=Path(r.qasm_res))
        else:
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

    def select_qasm_ccz_syntax(
        self, c: BenchmarkConfig, r: Resource, t: TestResults
    ) -> TestResults:
        if r.qasm_ccz_res:
            return replace(t, syntax=Syntax.QASM_CCZ, ref_path=Path(r.qasm_ccz_res))
        return None

    def validate(self):
        if not self.subject_path.is_dir():
            raise Exception(f"Didn't find subject dir at '{self.subject_path}')")

    # Write a ninja snippet to run the actual test and collect output; output
    # is a list of results where the formatted output will go
    def emit_test(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> TestResults:
        pass

    def emit_feyncount_analyze(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> AnalysisResults:
        return emit_feyncount_analyze(
            w, t.resource_name, t.opt_path.parent, t.syntax, t.opt_path
        )

    def emit_pyzx_analyze(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> AnalysisResults:
        return emit_pyzx_analyze(
            w, t.resource_name, t.opt_path.parent, t.syntax, t.opt_path
        )

    def emit_analyze(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> AnalysisResults:
        return self.emit_feyncount_analyze(w, c, t)

    @staticmethod
    def test_vars(c: BenchmarkConfig, t: TestResults) -> dict[str, any]:
        return {
            "opt_file": ns.escape_path(str(t.opt_path)),
            "log_file": ns.escape_path(str(t.log_path)),
            "time_file": ns.escape_path(str(t.time_path)),
            "ulimit_time": c.time_limit_s,
            "ulimit_mem": c.memory_limit_k,
        }


@dataclass(frozen=True)
class Benchmark:
    name: str  # Should match the name in the config
    subjects: list[TestSubject]
    config: BenchmarkConfig


def emit_generic_analyze(
    w: ns.Writer,
    resource_name: str,
    out_path: Path,
    syntax: Syntax,
    in_path: Path,
    analyze_rule: str,
    analyze_deps: list[str],
) -> AnalysisResults:
    base = in_path.stem
    analysis_path = out_path / f"{base}_{syntax}_analysis.json"
    log_path = out_path / f"{base}_{syntax}_analysis.log"
    time_path = out_path / f"{base}_{syntax}_analysis_time.json"
    w.build(
        [str(analysis_path), str(log_path), str(time_path)],
        analyze_rule,
        [str(in_path)],
        analyze_deps,
        variables={
            "analysis_file": ns.escape_path(str(analysis_path)),
            "analysis_log_file": ns.escape_path(str(log_path)),
            "analysis_time_file": ns.escape_path(str(time_path)),
        },
    )
    return AnalysisResults(
        resource_name, syntax, in_path, analysis_path, log_path, time_path
    )


def emit_feyncount_analyze(
    w: ns.Writer, res: str, out: Path, syntax: Syntax, tgt: Path
) -> AnalysisResults:
    rule = "feyncount_analyze"
    deps = ["feyncount_analyze_deps"]
    return emit_generic_analyze(w, res, out, syntax, tgt, rule, deps)


def emit_feyncount_qasm3_analyze(
    w: ns.Writer, res: str, out: Path, syntax: Syntax, tgt: Path
) -> AnalysisResults:
    rule = "feyncount_qasm3_analyze"
    deps = ["feyncount_analyze_deps"]
    return emit_generic_analyze(w, res, out, syntax, tgt, rule, deps)


def emit_pyzx_analyze(
    w: ns.Writer, res: str, out: Path, syntax: Syntax, tgt: Path
) -> AnalysisResults:
    return emit_generic_analyze(w, res, out, syntax, tgt, "pyzx_analyze", [])


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
            ["feynver_verify_deps"],
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


class FeynmanTestSubject(TestSubject):
    path: Path = Path("feynman")

    opt_params: str

    def __init__(self, opt_params: str):
        super().__init__()
        self.opt_params = opt_params

    select_syntax = TestSubject.select_any_syntax

    def emit_test(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_feynopt_qasm3" if t.syntax == Syntax.QASM3 else "bench_feynopt",
            [str(t.ref_path)],
            ["feynopt_bench_deps"],
            variables=TestSubject.test_vars(c, t) | {"opt_params": self.opt_params},
        )
        return t


class MlvoqcTestSubject(TestSubject):
    path: Path = Path("mlvoqc")

    @property
    def bench_bin_path(self) -> Path:
        return self.subject_path / "_build/default/bench_voqc.exe"

    select_syntax = TestSubject.select_qasm_ccz_syntax

    def emit_test(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_mlvoqc",
            [str(t.ref_path)],
            ["mlvoqc_bench_deps"],
            variables=TestSubject.test_vars(c, t),
        )
        return t

    emit_analyze = TestSubject.emit_pyzx_analyze


class PyzxTestSubject(TestSubject):
    path: Path = Path("pyzx")

    select_syntax = TestSubject.select_qasm_syntax

    def emit_test(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_pyzx",
            [str(t.ref_path)],
            [],
            variables=TestSubject.test_vars(c, t),
        )
        return t

    emit_analyze = TestSubject.emit_pyzx_analyze


class PyzxToddTestSubject(TestSubject):
    path: Path = Path("pyzx")

    select_syntax = TestSubject.select_qasm_syntax

    def emit_test(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_pyzx_todd",
            [str(t.ref_path)],
            [],
            variables=TestSubject.test_vars(c, t),
        )
        return t


class FeynmanPyzxTestSubject(TestSubject):
    paths: list[Path] = [Path("feynman"), Path("pyzx")]

    @property
    def subject_paths(self) -> Path:
        global args
        return [(args.run_path / p).resolve() for p in self.paths]

    select_syntax = TestSubject.select_qc_syntax

    def validate(self):
        for p in self.subject_paths:
            if not p.is_dir():
                raise Exception(f"Didn't find subject dir at '{self.subject_path}')")

    def emit_test(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_feynopt_pyzx",
            [str(t.ref_path)],
            ["feynopt_bench_deps"],
            variables=TestSubject.test_vars(c, t),
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

    select_syntax = TestSubject.select_qasm_ccz_syntax

    def emit_test(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_quartz",
            [str(t.ref_path)],
            ["quartz_bench_deps"],
            variables=TestSubject.test_vars(c, t),
        )
        return t


class QuesoTestSubject(TestSubject):
    path: Path = Path("queso")

    queso_time_s: int
    queso_mem_k: int

    def __init__(self, queso_time_s: int = 45, queso_mem_k: int = 4 * 1024 * 1024):
        super().__init__()
        self.queso_time_s = queso_time_s
        self.queso_mem_k = queso_mem_k

    select_syntax = TestSubject.select_qasm_ccz_syntax

    def emit_test(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_queso",
            [str(t.ref_path)],
            ["queso_bench_deps"],
            variables=TestSubject.test_vars(c, t)
            | {
                "queso_time": self.queso_time_s,
                "queso_mem": self.queso_mem_k,
            },
        )
        return t


class QuizxTestSubject(TestSubject):
    path: Path = Path("quizx")

    select_syntax = TestSubject.select_qc_syntax

    def emit_test(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_voqc",
            [str(t.ref_path)],
            [str(self.bench_bin_path)],
            variables=TestSubject.test_vars(c, t),
        )
        return t


class VvQcoTestSubject(TestSubject):
    path: Path = Path("vv-qco")

    select_syntax = TestSubject.select_qc_syntax

    opt_alg: str

    def __init__(self, opt_alg: str):
        super().__init__()
        self.opt_alg = opt_alg

    def emit_test(
        self, w: ns.Writer, c: BenchmarkConfig, t: TestResults
    ) -> TestResults:
        w.build(
            [str(t.opt_path), str(t.log_path), str(t.time_path)],
            "bench_vv_qco",
            [str(t.ref_path)],
            [],
            variables=TestSubject.test_vars(c, t) | {"opt_alg": self.opt_alg},
        )
        return t


subjects: dict[str, TestSubject] = {}

subject_ctors_by_name: dict[str, Callable] = {
    "feynman": lambda: FeynmanTestSubject("-O2"),
    "feynman-apf": lambda: FeynmanTestSubject("-apf"),
    "feynman-qpf": lambda: FeynmanTestSubject("-qpf"),
    "feynman-ppf": lambda: FeynmanTestSubject("-ppf"),
    "feynman-pyzx": FeynmanPyzxTestSubject,
    "mlvoqc": MlvoqcTestSubject,
    "pyzx": PyzxTestSubject,
    "pyzx-todd": PyzxToddTestSubject,
    "quartz": QuartzTestSubject,
    "queso": lambda: QuesoTestSubject(45, 4 * 1024 * 1024),
    "quizx": QuizxTestSubject,
    # "topt": ToptTestSubject,
    "vv-qco-bbmerge": lambda: VvQcoTestSubject("bbmerge"),
    "vv-qco-fasttmerge": lambda: VvQcoTestSubject("fasttmerge"),
    "vv-qco-internalhopt": lambda: VvQcoTestSubject("internalhopt"),
    "vv-qco-tohpe": lambda: VvQcoTestSubject("tohpe"),
    "vv-qco-fasttodd": lambda: VvQcoTestSubject("fasttodd"),
}


def make_subject(name: str) -> TestSubject:
    global subjects, subject_ctors_by_name

    s = subjects.get(name)
    if s == None:
        s: TestSubject = subject_ctors_by_name[name]()
        s.name = name
        # The sanity check right now just tests if there's a folder for the subject
        s.validate()
        subjects[name] = s
    return s


resources: dict[str, Resource] = {}


def make_resource(name: str) -> Resource:
    global resources
    if not name in resources:
        norm_res_path = (args.run_path / args.res_dir).resolve()
        qc_path = norm_res_path / "qc" / f"{name}.qc"
        qasm_path = norm_res_path / "qasm" / f"{name}.qasm"
        qasm_ccz_path = norm_res_path / "qasm-ccz" / f"{name}.qasm"
        qasm3_path = norm_res_path / "qasm3" / f"{name}.qasm"
        r = Resource(
            name,
            str(qc_path) if qc_path.is_file() else None,
            str(qasm_path) if qasm_path.is_file() else None,
            str(qasm_ccz_path) if qasm_ccz_path.is_file() else None,
            str(qasm3_path) if qasm3_path.is_file() else None,
        )
        # Sanity check
        if r.qc_res == None and r.qasm_res == None and r.qasm3_res == None:
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
            memory_limit_k=memory_limit,
            time_limit_s=time_limit,
            repeat_count=repeat_count,
        ),
    )


popl25_subjects = [
    "feynman",
    "feynman-apf",
    "feynman-qpf",
    "feynman-pyzx",
    "mlvoqc",
    "pyzx",
    "pyzx-todd",
    "quartz",
    "queso",
    "vv-qco-bbmerge",
    "vv-qco-fasttmerge",
    "vv-qco-internalhopt",
    "vv-qco-tohpe",
    "vv-qco-fasttodd",
]

popl25_resources = (
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
)


popl25slow_resources = ["mod_adder_1024", "gf2^32_mult", "ham15-high", "grover"]
popl25quick_resources = [r for r in popl25_resources if r not in popl25slow_resources]


benchmark_ctors_by_name: dict[str, Callable] = {
    "popl25": lambda: make_benchmark(
        "popl25",
        popl25_subjects,
        [Measurable.T_COUNT, Measurable.TIME, Measurable.MAX_MEMORY],
        popl25_resources,
        memory_limit=8 * 1024 * 1024,
        time_limit=1800,
    ),
    "popl25quick": lambda: make_benchmark(
        "popl25quick",
        popl25_subjects,
        [Measurable.T_COUNT, Measurable.TIME, Measurable.MAX_MEMORY],
        popl25quick_resources,
        memory_limit=8 * 1024 * 1024,
        time_limit=10,
    ),
    "test-feynman-pyzx": lambda: make_benchmark(
        "test-feynman-pyzx",
        ["feynman-pyzx"],
        [Measurable.T_COUNT, Measurable.TIME, Measurable.MAX_MEMORY],
        ["qft_4", "tof_4", "mod_adder_1024"],
        memory_limit=8 * 1024 * 1024,
        time_limit=600,
    ),
    "minimal": lambda: make_benchmark(
        "minimal",
        [
            "feynman",
            # "feynman-ppf",
            # "mlvoqc",
            # "quartz",
            # "queso",
            # "feynman-pyzx",
            "vv-qco-fasttodd",
        ],
        [Measurable.T_COUNT, Measurable.TIME, Measurable.MAX_MEMORY],
        ["qft_4", "tof_4", "mod_adder_1024"] + ["if-simple", "loop-simple"],
        memory_limit=8 * 1024 * 1024,
        time_limit=60,
    ),
    "minimal-all": lambda: make_benchmark(
        "minimal-all",
        popl25_subjects,
        [Measurable.T_COUNT, Measurable.TIME, Measurable.MAX_MEMORY],
        ["qft_4", "tof_4", "mod_adder_1024"] + ["if-simple", "loop-simple"],
        memory_limit=8 * 1024 * 1024,
        time_limit=60,
    ),
}

benchmark_ctors_by_name |= {
    f"popl25{speed}-{subject}": lambda speed=speed, subject=subject: make_benchmark(
        f"popl25{speed}-{subject}",
        [subject],
        [Measurable.T_COUNT, Measurable.TIME, Measurable.MAX_MEMORY],
        popl25quick_resources if speed == "quick" else popl25slow_resources,
        memory_limit=8 * 1024 * 1024,
        time_limit=60 if speed == "quick" else 600,
    )
    for speed, subject in product(["slow", "quick"], popl25_subjects)
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
    hostname: str
    start_ts: str
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
    return {"t_gates": t_gates}


default_hostname = (
    Popen(["hostname", "-f"], stdout=PIPE)
    .communicate()[0]
    .decode(errors="ignore")
    .strip()
)
default_start_ts = start_ts.isoformat(timespec="seconds")


def read_time_results(time_path: Path) -> dict[str, any]:
    global default_hostname, default_start_ts
    try:
        loaded = json.load(open(time_path, "r"))
    except:
        pass
    time_required_keys = [
        ("hostname", default_hostname, str),
        ("start_ts", default_start_ts, str),
        ("user_time", None, float),
        ("sys_time", None, float),
        ("elapsed_time", None, float),
        ("max_resident", None, int),
        ("status", None, int),
    ]
    results = {}
    for k, d, f in time_required_keys:
        results[k] = d
        try:
            if k in loaded:
                results[k] = f(loaded[k])
        except:
            pass
    return results


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
                    tests.append(s.emit_test(w, b.config, t))
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
                        tests.append(s.emit_test(w, b.config, t))

        # Figure out which resources (refs) are used by the tests, and
        # add analysis targets for them -- we do this as a separate step
        # because we don't want to duplicate the ref analysis, typically one
        # ref analysis will be compared against multiple different
        # optimizations
        ref_build_path = build_path / "ref"
        os.makedirs(ref_build_path)
        refs_analysis: list[AnalysisResults] = []
        for ref_path, resource_name, syntax in sorted(
            set(((t.ref_path, t.resource_name, t.syntax) for t in tests))
        ):
            if syntax == Syntax.QASM3:
                a = emit_feyncount_qasm3_analyze(
                    w, resource_name, ref_build_path, syntax, ref_path
                )
            else:
                a = emit_feyncount_qasm3_analyze(
                    w, resource_name, ref_build_path, syntax, ref_path
                )
            refs_analysis.append(a)

        # Make analysis targets for test results and annotate the test
        # results with them
        def emit_analyze(t: TestResults):
            return subjects[t.subject_name].emit_analyze(w, b.config, t)

        tests = [replace(t, opt_analysis=emit_analyze(t)) for t in tests]

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
        results = {
            "benchmark": b.name,
            "subject": "ref",
            "resource": a.resource_name,
            "syntax": a.syntax,
            "ref_t_gates": None,
        }
        results |= read_time_results(a.time_path)
        results |= read_analysis_results(a.results_path)
        r = DataRow(**results)
        ref_rows[(a.resource_name, a.syntax)] = (r, a)
        rows.append(r)
    for t in tests:
        results = {
            "benchmark": b.name,
            "subject": t.subject_name,
            "resource": t.resource_name,
            "syntax": t.syntax,
        }
        results |= read_time_results(t.time_path)
        results |= read_analysis_results(
            t.opt_analysis.results_path if t.opt_analysis != None else None
        )
        ref = ref_rows.get((t.resource_name, t.syntax), None)
        results["ref_t_gates"] = ref[0].t_gates if ref != None else None
        rows.append(DataRow(**results))

    cw = csv.writer(open(build_path / f"{b.name}_{args.bench_ts}.csv", "w"))
    cols = [
        ("hostname", "hostname"),
        ("start time", "start_ts"),
        ("benchmark", "benchmark"),
        ("subject", "subject"),
        ("resource", "resource"),
        ("syntax", "syntax"),
        ("reference t gates", "ref_t_gates"),
        ("t gates", "t_gates"),
        ("user time (s)", "user_time"),
        ("sys time (s)", "sys_time"),
        ("elapsed time (s)", "elapsed_time"),
        ("max resident (kiB)", "max_resident"),
        ("exit status", "status"),
    ]
    cw.writerow([k for name, k in cols])
    for r in rows:
        cw.writerow([r.__dict__[k] for _, k in cols])
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
        logger.info("Default hostname: '%s'", default_hostname)
        logger.info("Default start time: '%s'", default_start_ts)

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
