# BayesBeat

Bayesian analysis of ringdowns.

## Installation

### Using pip

`bayesbeat` can be installed directly using `pip`

```
pip install bayeseat
```

### Using conda

`bayesbeat` cannot yet be installed via `conda`. We do, however, provide `environment.yaml` which
can be used to install it:

```
conda env create environment.yaml
```

**Note:** for instructions on intall `conda` see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Running analyses

Before running an analysis, make sure you have activated the relevant environment.

Here is an example of how to the run analysis for a data file called `PyTotalAnalysis.mat` located in `data/`. The index determines which of ringdowns in the data file will be analyzed.

### Creating an ini file

Create an `.ini` file with a given name, e.g. `example.ini`

```
bayesbeat_create_ini example.ini
```

**Note:** if you plan to use a scheduler, e.g. HTCondor or Slurm to run the analyses you should append `--scheduler HTCondor` or `--scheduler Slurm` to the above command. This will add the relevant section.

Open the new ini file and set the values for the different fields.
You must specify `output` and `datafile`, the other settings will all have defaults that should work.
The most important are:

* `indices`: determines which ringdowns in the data file will be analyzed. Only used if running via Condor. If `None` or `'all'` all indices will be analysed. Otherwise, should be a list of integers (starting at 0).
* The parameters in the `Model` section. This will depend on the model being used.
* `n-pool`: the number of cores to use. We recommend setting this to at least 4.

The file should look something like this (this example uses HTCondor):

```
[General]
output = "outdir/"
label = "disk_0"
datafile = "data/PyTotalAnalysis.mat"
indices = [0]
seed = 1234
plot = True

[Data]
rescale-amplitude = False
maximum-amplitude = None

[Model]
name = GenericAnalyticGaussianBeam
equation_name = General_Equation_3_Terms.txt
photodiode-size = 1e-2
photodiode-gap = 0.25e-3
n-terms = 3
include-gap = True
beam_radius = 1e-3
x_offset = 0.0
rin_noise = True
prior_bounds = {"a_ratio": [0, 1], "tau_1": [290, 310], "tau_2": [140, 160], "dphi": [0, 3.141592654], "domega": [0.18, 0.22], "a_scale": [0, 10], "sigma_noise": [0, 10.0]}

[Analysis]
resume = True

[Sampler]
nlive = 1000
reset_flow = 8

[HTCondor]
request-disk = "2GB"
request-memory = "2GB"
request-cpus = 4
```

### Running with HTCondor or Slurm

The recommended way to use `bayesbeat` is on a cluster with HTCondor or Slurm installed.
This allows analyses to run in parallel rather than one-by-one on a local machine.
To run a local machine, see the section below.

When creating a ini file, add the `--scheduler` arguments with either `htcondor` or
`slurm`. This will add the relevant section to the file.

Once you have a create an ini file, the analyses can be prepared (built) and then submitted.
To do so run

```
bayesbeat_build example.ini
```

this will construct the relevant files which can be submitted using the command that is printed after the command has run.
The exact command will depend on which scheduler you are using.
Alternatively, if you run

```
bayesbeat_build example.ini --submit
```

the analysis will be built and submitted in a single step.

**Note:** if the output directory already exists, an error will be raised and the analysis will not be built or submitted. The `--overwrite` flag will ignore this but is not recommended as this can lead to data loss.

### Running on a local machine (without HTCondor or Slurm)

To run an analysis locally instead of via scheduler, use the following command

```
bayesbeat_run example.ini --index 0
```

where `--index` specifies which ringdown in the datafile to analyse.

**Note:** this ignores the value of `indices` in the ini file.

**Note:** it is not possible to analyse multiple ringdowns with a single call to `bayesbeat_run`.
