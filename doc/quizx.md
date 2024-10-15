# Building QuiZX

```
# as root
apt install cargo rustc

# as user
cargo build --bin bench_quizx
```

As seems to be a theme, QuiZX is intended as a library, and the file that does the optimization is specific to the paper -- I suspect it's [`quizx/src/bin/simp_and_extract.rs`](from-spire-paper-artifact/simp_and_extract.rs) from the artifact that's the entry point in this case. It's very similar to the original, but it's been modified from what's in the QuiZX repository to take a path argument..
