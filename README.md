This repository contains benchmarks allowing comparison of various quantum circuit/quantum program optimizers, mainly for the convenience of folks working on Feynman.

Please see [`doc/design.md`](doc/design.md) for more detailed information about how the benchmarking is done.

# Building And Running

## Repository Layout

| Folder Or File | Description |
|--|--|
| `/doc` | Folder containing general design information, and notes about tests and specific test subjects
| `/bench.py` | The script that does all the real work |
| `/bench` | Data and output folders for `bench.py` |
| `/bench/build` | Output folder for benchmarks, can be removed to clean the repository |
| `/bench/resources` | Source `.qc` and `.qasm` files used as input by the benchmark tests |
| `/<subject>` | Source and build folders for test subjects i.e. `feynman`, `mlvoqc`, `quartz`, `queso`, `quizx`... |

## Basic Instructions

Before running any benchmarks, you should build the programs being benchmarked. Their needs are very diverse, and I elected not to automate this at the moment because without containerizing the whole build it's challenging to automate installing build dependencies without polluting the user's environment. The programs being benchmarked have diverse build dependencies -- Feynman needs Haskell (specifically some vaguely recent version of GHC), others need variously Java, C++, OCaml, or Rust... you should consult the documentation for those programs for details.

For some test subjects, as of this writing QUESO and Quartz, it's necessary to generate rule files. Those packages include the tools to do this, but we don't automate the process. The rules should be checked into the repository, but if you want to regenerate these files there should be instructions in the subject folder.

For `bench.py` itself you will need the `ninja` package, but otherwise it just needs a recent 3.x version of python (I am developing with 3.10, older might also work and newer of course barring breaking changes).

Once the test subjects are built, run `python3 bench.py <suite>` where `<suite>` is something like `popl25`. For a list of current available options you can run `python3 bench.py --list-suites`. Output will be generated in `bench/build/<suite>/<timestamp>/<subject>/...`.

## Platform Compatibility

This system is currently maintained targeting Ubuntu 22.04. It will probably work on other unixes and e.g. MacOS with Homebrew or similar, but probably not on Windows (and if that's your platform, consider running it under WSL). If you want to port it, the Ninja generation will most likely need to be updated somewhat significantly, but the rest might just work.

# Acknowledgements

The various programs benchmarked are the work of their authors; see the documentation in the subtree folders for more information.

Scripts used to drive [VOQC](doc/mlvoqc.md), [Quartz](doc/quartz.md), and [QuiZX](doc/quizx.md) are adapted from the [Zenodo artifact](https://doi.org/10.5281/zenodo.10729070) for "The T-Complexity Costs of Error Correction for Control Flow in Quantum Computation" (Charles Yuan and Michael Corbin, [Proceedings of the ACM on Programming Languages, Volume 8, PLDI](https://doi.org/10.1145/3656397) and are credited to those authors.
