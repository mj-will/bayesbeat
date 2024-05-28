# Analyzing real data

You should be able to run the analyses using real data either directly or
using HTCondor/Slurm.

You will need to download the data and update paths etc in the ini file, though
you shouldn't need to download any equation/coefficients files.

## Running directly

To analyse a single ringdown run:

```
bayesbeat_run data_recover_analytic_general.ini --index 0
```

## Running with a scheduler

To analyse the ringdowns specified in `indices` run:

```
bayesbeat_build data_recovery_analytic_general.ini --submit
```

`--submit` is optional.

You will likely need to change the `Slurm` section in the ini file to
`HTCondor`. The main readme shows the fields you need.
