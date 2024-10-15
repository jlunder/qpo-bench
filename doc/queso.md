# Building QUESO

```
# as root
apt install maven openjdk-17-jdk-headless

# as user
mvn package -Dmaven.test.skip
```

The output .jar file is `target/QUESO-1.0-jar-with-dependencies.jar`.

You will need some rule files not in the QUESO repository -- per the Spire paper:

> Both Quartz and QUESO require a rule file and associated
> command-line arguments to be provided to the optimizer... For
> QUESO, we invoked the EnumeratorPrune -g nam -q 3 -s 6 flags to
> generate the rule files rules_q3_s6_nam.txt and
> rules_q3_s6_nam_symb.txt, which we then provided to the optimizer
> using the -r and -sr flags respectively, along with the -j "nam"
> flag.

It turns out the "EnumeratorPrune" class has been renamed "Synthesizer" -- the command line to do this generation is:

```
java --enable-preview -cp target/QUESO-1.0-jar-with-dependencies.jar \
  Synthesizer -g nam -q 3 -s 6
```

These files appear to be almost-but-not-quite the same as the ones from the Spire paper; for example, my rules_q3_s6_nam.txt is in a different order, but also, it's 3 lines shorter. I don't know why.

These are the ones from the Spire paper artifact:

[rules_q3_s6_nam.txt](from-spire-paper-artifact/rules_q3_s6_nam.txt)

[rules_q3_s6_nam_symb.txt](from-spire-paper-artifact/rules_q3_s6_nam_symb.txt)


# Running QUESO

After you build, there's a very long command line -- X is the input file name, OUT is the output folder, TIMEOUT is a timeout in seconds (which I gather isn't well respected). Also, "Applier" has been renamed "Optimizer":

```
java --enable-preview -cp target/QUESO-1.0-jar-with-dependencies.jar \
  Optimizer -c $X -g nam -r rules_q3_s6_nam.txt -sr rules_q3_s6_nam_symb.txt \
  -t $TIMEOUT -o $OUT -j "nam"

java --enable-preview -cp queso/QUESO-1.0-jar-with-dependencies.jar Applier -c ~/repos/feynman/benchmarks/qasm/adder_8.qasm -g nam -r rules_q3_s6_nam.txt -sr rules_q3_s6_nam_symb.txt -t 30 -o out -j "nam"
```

# Sus

I got test failures with the customary `mvn install`:
```
[ERROR] Failures: 
[ERROR]   OptimizerTest.testSymbRule2:179 expected: <s q2; 
cx q2, q1; 
t q1; 
cx q2, q3; 
cx q1, q2; 
h q3; 
cx q2, q3; 
cx q1, q2; 
> but was: <s q2; 
cx q2, q1; 
t q1; 
cx q2, q3; 
h q3; 
cx q1, q2; 
cx q2, q3; 
cx q1, q2; 
>
[ERROR]   SynthesizerTest.testRotationMerging:110 expected: <true> but was: <false>
```

