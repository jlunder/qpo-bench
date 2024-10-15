# Notes

As of 2024-10-14, looks like the latest (OCaml) version of VOQC is 0.3.0, which was published 2022 at https://github.com/inQWIRE/mlvoqc. The README suggests installing the pyvoqc wrapper at https://github.com/inQWIRE/pyvoqc, which was last updated around the same time, and maybe even suggests using it via Qiskit.

I don't see evidence that the authors of "The T-Complexity Costs of Error Correction for Control Flow in Quantum Computation" did this. I think they ran mlvoqc directly and I think that makes sense for our application also.

# Building mlvoqc

In Ubuntu 22.04 (probably newer would also work):
```
# as root:
apt install ocaml opam ocaml-dune

# as user:
opam init
# I accepted modification of .profile here, but :shrug:
opam switch create voqc 4.13.1
eval $(opam env --switch=voqc)
opam install dune  # build system
opam install openQASM zarith  # library dependencies

dune build bench_voqc.exe
```

Build product is at `_build/default/bench_voqc.exe`

FWIW opam itself says to 
run `eval $(opam env --switch=voqc)` and that's what I did, instead of just `eval $(opam env)` (per the instructions in the mlvoqc repository). I didn't run `opam install dune` either because I had alread installed dune via apt! Distro vs. upstream package management, it's a gas.

There is evidence the authors of the paper mentioned above also installed this exact version of the OCaml compiler (4.13.1). This also happens to be the default in Ubuntu 22.04 when I installed (2024-10-14).

**The run\_mlvoqc in the paper artifact doesn't exist in the current mlvoqc repository** -- the `example.ml` has been rewritten and that looks like where the binary was built from (after building everything, I was able to run it on `length_simplified_orig1.qasm` and it produced the same log messages as I got from running the benchmarks, FWIW).

I captured the file here: [`example.ml`](from-spire-paper-artifact/example.ml)

This is the file I renamed to `bench_voqc.ml`.

# Running mlvoqc

After you build, it's pretty simple:

```
```

# Instructions for pyvoqc

**Probably don't do this, see note at end.**

In ubuntu 22.04:
```
# as root:
apt install ocaml opam ocaml-dune
```

From the mlvoqc docs:
```
# as user:
# environment setup
opam init
eval $(opam env)

# install the OCaml version of VOQC
opam pin voqc https://github.com/inQWIRE/mlvoqc.git#mapping
```

-- `#mapping` should be a version number but it doesn't specify anywhere _which_ version to pin, I infer `#0.3.0`.

Anyway this point is where I figured out this was probably all ~~BS~~not the right direction anyway.

