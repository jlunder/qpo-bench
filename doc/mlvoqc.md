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
```

FWIW opam itself says to 
run `eval $(opam env --switch=voqc)` and that's what I did, instead of just `eval $(opam env)` (per the instructions in the mlvoqc repository). I didn't run `opam install dune` either because I had alread installed dune via apt! Distro vs. upstream package management, it's a gas.

There is evidence the authors of the paper mentioned above also installed this exact version of the OCaml compiler (4.13.1). This also happens to be the default in Ubuntu 22.04 when I installed (2024-10-14).

**The run\_mlvoqc in the paper artifact doesn't exist in the current mlvoqc repository** -- the example.ml has been rewritten and that looks like where the binary was built from (after building everything, I was able to run it on length_simplified_orig1.qasm and it produced the same log messages as I got from running the benchmarks, FWIW).

I quote the file here:
```
open Printf
open Voqc.Qasm
open Voqc.Main

(* Argument parsing *)
let f = ref ""
let o = ref ""
let light = ref false
let usage = "usage: " ^ Sys.argv.(0) ^ " -f string"
let speclist = [
    ("-f", Arg.Set_string f, ": input program");
    ("-o", Arg.Set_string o, ": output program");
    ("--light", Arg.Set light, ": use light optimization")
  ]
let () =
  Arg.parse
    speclist
    (fun x -> raise (Arg.Bad ("Bad argument : " ^ x)))
    usage;
if !f = "" then printf "ERROR: Input file (-f) required.\n" else

(* Read input file *)
let _ = printf "Reading input %s\n%!" !f in
let (c0, n) = read_qasm !f in
let _ = printf "Input circuit has %d gates and uses %d qubits.\n%!" (count_total c0) n in

(* Optimize *)
let c1 = if !light then optimize_nam_light c0 else optimize_nam c0 in
printf "After optimization, the circuit uses %d gates : { H : %d, X : %d, Rzq : %d, CX : %d }.\n%!"
          (count_total c1) (count_H c1) (count_X c1) (count_Rzq c1) (count_CX c1);

write_qasm c1 n !o
```
It's fairly trivial, but we should get permission from the authors if we're going to publish this in our artifact submission, which we should, so that people can run the comparisons.

**There is an error in the mlvoqc dune-project file:** the last line appears to have been modified to add a version requirement for zarith, so it looks like:
```
  zarith (>= 1.5)))
```
but there need to be parens around the line for it to parse with a version spec:
```
  (zarith (>= 1.5))))
```

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

