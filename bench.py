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
    QC = "qc"
    QASM = "qasm"
    QASM3 = "qasm3"


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
class AnalysisResults:
    resource_name: str
    syntax: Syntax
    in_path: Path
    results_path: Path
    log_path: Path
    time_path: Path


@dataclass(frozen=True, order=True)
class VerifResults:
    resource_name: str
    syntax: Syntax
    ref_path: Path
    opt_path: Path
    results_path: Path
    log_path: Path
    time_path: Path


@dataclass(frozen=True, order=True)
class TestResults:
    benchmark_name: str
    subject_name: str
    resource_name: str
    run_id: int
    syntax: Syntax
    ref_path: Path
    opt_path: Path
    log_path: Path
    time_path: Path
    opt_analysis: AnalysisResults | None = None
    verif: VerifResults | None = None


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
        return (args.run_path / args.bench_build / args.bench_root / name).resolve()

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
        **kw_args,
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
        time_outputs_str = ", ".join([f'"{k}": {v}' for k, v in time_outputs.items()])
        writer.rule(
            rule_name,
            f"time -q -f '{{{ns.escape(time_outputs_str)}}}' "
            + f"-o '{time_file_var}' /bin/sh -c "
            + f"'{escaped_command.replace("'", "'\\''")}'",
            **kw_args,
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

    def emit_analysis_dependencies(self, writer: ns.Writer) -> None:
        writer.rule(
            "build_feyncount",
            f"cd {ns.escape_path(str(self.subject_path))} && "
            + "cabal build -O2 feyncount",
        )
        writer.build("feyncount", "build_feyncount")

        rule = ns.escape(self.name)
        TestSubject.emit_time_rule(
            writer,
            "feyncount_analyze",
            f"cd {ns.escape_path(str(self.subject_path))} && "
            + "cabal exec -v0 feyncount -O2 -- '$in' "
            + "> '$analysis_file' 2> '$analysis_log_file'",
            "$analysis_time_file",
        )

        rule = ns.escape(self.name)
        TestSubject.emit_time_rule(
            writer,
            "feyncount_qasm3_analyze",
            f"cd {ns.escape_path(str(self.subject_path))} && "
            + "cabal exec -v0 feyncount -O2 -- -qasm3 '$in' "
            + "> '$analysis_file' 2> '$analysis_log_file'",
            "$analysis_time_file",
        )

    def emit_analysis(
        self,
        writer: ns.Writer,
        resource_name: str,
        out_path: Path,
        syntax: Syntax,
        ref_path: Path,
    ) -> AnalysisResults:
        base = ref_path.stem
        analysis_path = out_path / f"{base}_{syntax}_analysis.json"
        log_path = out_path / f"{base}_{syntax}_analysis_stderr.log"
        time_path = out_path / f"{base}_{syntax}_analysis_time.json"
        vars = {
            "analysis_file": ns.escape_path(str(analysis_path)),
            "analysis_log_file": ns.escape_path(str(log_path)),
            "analysis_time_file": ns.escape_path(str(time_path)),
        }
        if syntax == Syntax.QASM3:
            writer.build(
                [str(analysis_path), str(log_path), str(time_path)],
                "feyncount_qasm3_analyze",
                [str(ref_path)],
                ["feyncount"],
                variables=vars,
            )
        else:
            writer.build(
                [str(analysis_path), str(log_path), str(time_path)],
                "feyncount_analyze",
                [str(ref_path)],
                ["feyncount"],
                variables=vars,
            )
        return AnalysisResults(
            resource_name, syntax, ref_path, analysis_path, log_path, time_path
        )

    def emit_verif_dependencies(self, writer: ns.Writer) -> None:
        writer.rule(
            "build_feynver",
            f"cd {ns.escape_path(str(self.subject_path))} && cabal build -O2 feynver",
        )
        writer.build("feynver", "build_feynver")

        rule = ns.escape(self.name)
        TestSubject.emit_time_rule(
            writer,
            "feynver_verify",
            f"cd {ns.escape_path(str(self.subject_path))} && "
            + "cabal exec -v0 feynver -O2 -- '$ref_file' '$opt_file' "
            + "> '$verif_result_file' 2> '$verif_log_file'",
            "$verif_time_file",
        )

    # def emit_verify(
    #     self, writer: ns.Writer, out_path: Path, to_verify: list[TestResults]
    # ) -> list[TestResults]:
    #     verified_result = []
    #     for result in to_verify:
    #         verify_log_path = out_path / (res.name + "_verify_stderr.log")
    #         verify_time_path = out_path / (res.name + "_verify_time.json")
    #         vars = {
    #             "ref_file": ns.escape_path(str(result.ref_path)),
    #             "log_file": ns.escape_path(str(verify_log_path)),
    #             "time_file": ns.escape_path(str(verify_time_path)),
    #         }
    #         writer.build(
    #             [str(verify_log_path), str(verify_time_path)],
    #             "feynver_verify",
    #             [str(result.opt_path)],
    #             ["feynver", str(result.ref_path)],
    #             variables=vars,
    #         )
    #         verified_result.append(
    #             replace(
    #                 result,
    #                 verify_log_path=verify_log_path,
    #                 verify_time_path=verify_time_path,
    #             )
    #         )

    def emit(
        self, writer: ns.Writer, out_path: Path, config: BenchmarkConfig
    ) -> list[TestResults]:
        writer.rule(
            "build_feynopt",
            f"cd {ns.escape_path(str(self.subject_path))} && "
            + "cabal build -O2 feynopt",
        )
        writer.build("feynopt", "build_feynopt")

        rule = ns.escape(self.name)
        TestSubject.emit_time_rule(
            writer,
            rule,
            f"cd {ns.escape_path(str(self.subject_path))} && "
            + "cabal exec -v0 -O2 feynopt -- -O2 '$in' "
            + "> '$opt_file' 2> '$log_file'",
            "$time_file",
        )

        rule_qasm3 = ns.escape(f"{self.name}_qasm3")
        TestSubject.emit_time_rule(
            writer,
            rule_qasm3,
            f"cd {ns.escape_path(str(self.subject_path))} && "
            + "cabal exec -v0 -O2 feynopt -- -O2 -qasm3 '$in' "
            + "> '$opt_file' 2> '$log_file'",
            "$time_file",
        )

        targets = []
        for res in config.resources:
            if res.qc_res:
                rule_used = rule
                ref_path = Path(res.qc_res)
                syntax = Syntax.QC
            elif res.qasm_res:
                rule_used = rule
                ref_path = Path(res.qasm_res)
                syntax = Syntax.QASM
            elif res.qasm3_res:
                rule_used = rule_qasm3
                ref_path = Path(res.qasm3_res)
                syntax = Syntax.QASM3
            else:
                assert not "neither circuit nor program assigned to this resource?"
            opt_path = out_path / f"{res.name}_opt{ref_path.suffix}"
            log_path = out_path / f"{res.name}_stderr.log"
            time_path = out_path / f"{res.name}_time.json"
            vars = {
                "opt_file": ns.escape_path(str(opt_path)),
                "log_file": ns.escape_path(str(log_path)),
                "time_file": ns.escape_path(str(time_path)),
            }
            writer.build(
                [str(opt_path), str(log_path), str(time_path)],
                rule_used,
                [str(ref_path)],
                ["feynopt"],
                variables=vars,
            )
            targets.append(
                TestResults(
                    benchmark_name=config.name,
                    subject_name=self.name,
                    resource_name=res.name,
                    run_id=0,
                    syntax=syntax,
                    ref_path=ref_path,
                    opt_path=opt_path,
                    log_path=log_path,
                    time_path=time_path,
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
            f"cd {ns.escape_path(str(self.subject_path))} && "
            + "dune build bench_voqc.exe",
        )
        writer.build(bench_bin, build_rule)

        rule = ns.escape(self.name)
        TestSubject.emit_time_rule(
            writer,
            rule,
            f"cd {ns.escape_path(str(self.subject_path))} && "
            + f"{ns.escape_path(bench_bin)} -f '$in' -o '$opt_file' "
            + "2>&1 > '$log_file'",
            "$time_file",
        )

        targets = []
        for res in config.resources:
            if res.qasm_res:
                rule_used = rule
                ref_path = Path(res.qasm_res)
                syntax = Syntax.QASM
            else:
                continue
            opt_path = out_path / f"{res.name}_opt{ref_path.suffix}"
            log_path = out_path / f"{res.name}_stderr.log"
            time_path = out_path / f"{res.name}_time.json"
            vars = {
                "opt_file": ns.escape_path(str(opt_path)),
                "log_file": ns.escape_path(str(log_path)),
                "time_file": ns.escape_path(str(time_path)),
            }
            writer.build(
                [str(opt_path), str(log_path), str(time_path)],
                rule_used,
                [str(ref_path)],
                [bench_bin],
                variables=vars,
            )
            targets.append(
                TestResults(
                    benchmark_name=config.name,
                    subject_name=self.name,
                    resource_name=res.name,
                    run_id=0,
                    syntax=syntax,
                    ref_path=ref_path,
                    opt_path=opt_path,
                    log_path=log_path,
                    time_path=time_path,
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
        subject_root = (
            args.run_path / args.bench_build / args.bench_root / self.name
        ).resolve()

        rule = ns.escape(self.name)
        queso_jar = str(subject_root / "target/QUESO-1.0-jar-with-dependencies.jar")

        build_rule = ns.escape(f"build_{self.name}")
        writer.rule(
            build_rule,
            f"cd {ns.escape_path(str(subject_root))} && "
            + "mvn package -Dmaven.test.skip",
        )

        writer.build(queso_jar, build_rule)

        nam_txt = str(subject_root / "rules_q3_s6_nam.txt")
        nam_symb_txt = str(subject_root / "rules_q3_s6_nam_symb.txt")
        writer.rule(
            rule,
            f"cd {ns.escape_path(str(subject_root))} && "
            + "cabal exec -v0 -O2 feynopt -- -O2 $in > $out",
        )

        rule_qasm3 = ns.escape(f"{self.name}_qasm3")
        writer.rule(
            rule_qasm3,
            f"cd {ns.escape_path(str(subject_root))} && "
            + "cabal exec -v0 -O2 feynopt -- -O2 -qasm3 $in > $out",
        )

        targets = []
        for r in bench.resources:
            res = args.all_resources[r]
            qc = res.qc_res or res.qasm_res
            qp = res.qasm3_res
            if qc:
                out = (
                    args.run_path / output_path / f"{r}_opt{os.path.splitext(qc)[1]}"
                ).resolve()
                targets.append(out)
                writer.build([str(out)], rule, [str(qc)], [self.name])
            elif qp:
                out = (
                    args.run_path / output_path / f"{r}_opt{os.path.splitext(qp)[1]}"
                ).resolve()
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
        subject_root = (
            args.run_path / args.bench_build / args.bench_root / self.name
        ).resolve()

        rule = ns.escape(self.name)

        build_rule = ns.escape("build_" + self.name)
        writer.rule(
            build_rule,
            f"cd {ns.escape_path(str(subject_root))} && "
            + "cabal build -v0 -O2 feynopt",
        )

        writer.build(self.name, build_rule)

        writer.rule(
            rule,
            f"cd {ns.escape_path(str(subject_root))} && "
            + "cabal exec -v0 -O2 feynopt -- -O2 $in > $out",
        )

        rule_qasm3 = ns.escape(self.name + "_qasm3")
        writer.rule(
            rule_qasm3,
            f"cd {ns.escape_path(str(subject_root))} && "
            + "cabal exec -v0 -O2 feynopt -- -O2 -qasm3 $in > $out",
        )

        targets = []
        for r in bench.resources:
            res = args.all_resources[r]
            qc = res.qc_res or res.qasm_res
            qp = res.qasm3_res
            if qc:
                out = (
                    args.run_path / output_path / (r + "_opt" + os.path.splitext(qc)[1])
                ).resolve()
                targets.append(out)
                writer.build([str(out)], rule, [str(qc)], [self.name])
            elif qp:
                out = (
                    args.run_path / output_path / (r + "_opt" + os.path.splitext(qp)[1])
                ).resolve()
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
        norm_subject_dir = (
            args.run_path / args.bench_build / args.bench_root / name
        ).resolve()
        if not norm_subject_dir.is_dir():
            raise Exception(
                f"Didn't find subject dir at '{norm_subject_dir}' (normalized "
                + f"from '{args.bench_build / args.bench_root / name}')"
            )
        subjects[name] = subject_ctors_by_name[name]()
    return subjects[name]


resources: dict[str, Resource] = {}


def make_resource(name: str) -> Resource:
    global resources
    if not name in resources:
        norm_res_path = (
            args.run_path / args.bench_build / args.bench_root / args.res_dir
        ).resolve()
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
    norm_bench_root = (args.run_path / args.bench_build / args.bench_root).resolve()
    if not norm_bench_root.is_dir():
        raise Exception(f"Didn't find bench project root dir at '{args.bench_root}'")
    logger.info("Real bench_root is '%s'", norm_bench_root)


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

        # Add optimization build targets (this is the actual test runs)
        targets: list[TestResults] = []
        for s in b.subjects:
            os.makedirs(build_path / s.name)
            targets.extend(s.emit(w, build_path / s.name, b.config))

        # Figure out which resources (refs) are used by the targets, and
        # add analysis targets for them -- we do this as a separate step
        # because we don't want to duplicate the ref analysis, typically one
        # ref analysis will be compared against multiple different
        # optimizations
        ref_build_path = build_path / "ref"
        os.makedirs(ref_build_path)
        a: FeynmanTestSubject = make_subject("feynman")
        a.emit_analysis_dependencies(w)
        refs_analysis: list[AnalysisResults] = [
            a.emit_analysis(w, resource_name, ref_build_path, syntax, ref_path)
            for ref_path, resource_name, syntax in sorted(
                set(((t.ref_path, t.resource_name, t.syntax) for t in targets))
            )
        ]

        # Make analysis targets for test results and annotate the test
        # results with them
        targets = [
            replace(
                t,
                opt_analysis=a.emit_analysis(
                    w, t.resource_name, t.opt_path.parent, t.syntax, t.opt_path
                ),
            )
            for t in targets
        ]

        w.build(
            "all",
            "phony",
            [str(t.opt_path) for t in targets]
            + [
                str(t.opt_analysis.results_path)
                for t in targets
                if t.opt_analysis != None
            ]
            + [str(t.verif.results_path) for t in targets if t.verif != None]
            + [str(a.results_path) for a in refs_analysis],
        )
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
