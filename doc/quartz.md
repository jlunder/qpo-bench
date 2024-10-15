# Building Quartz

It's a bit hard to tell what source was used to do the optimization for the benchmark -- in this case I suspect it's [`src/test/test_nam.cpp`](from-spire-paper-artifact/test_nam.cpp) that was used as the basis but again modified (at minimum, the src and dest gate sets are different from what's in the repository).

You will need these other data files -- per the paper:

> Both Quartz and QUESO require a rule file and associated
> command-line arguments to be provided to the optimizer... For
> Quartz, we used the provided gen_ecc_set tool to generate the
> rule file 3_2_5_complete_ECC_set.json, which we then provided
> to the optimizer using the --eqset flag. ...

[H_CZ_2_2_complete_ECC_set_modified.json](from-spire-paper-artifact/H_CZ_2_2_complete_ECC_set_modified.json)

[3_2_5_complete_ECC_set.json](from-spire-paper-artifact/3_2_5_complete_ECC_set.json)
