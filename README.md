# BayesBeat

Bayesian analysis of ringdowns.

**Warning:** the following instructions have only been tested on Linux and may be updated after testing on other platforms.

## Downloading the code

The recommended way to download the code is by cloning this repository. 

Alternatively, you can download the code as a Zip from [this URL](https://github.com/mj-will/bayesbeat/archive/refs/heads/main.zip).

## Installation

Before installing `nessai` you'll need to download and install `conda`. See the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). If running on computing cluster you will not have to do this step.

Then open a terminal, i.e. PowerShell, Command Line or equivalent.

### Creating a conda environment

We start by creating an environment from the environment file, this will install all the necessary packages:

```
conda env create -f environment.yml
```

The environment will be called `bayes-beat`.

**Note:** this can be very slow and on certain systems.

### Activating the environment

We must activate the environment to use the installed packages:

```
conda activate bayes-beat
```

You should now see `(bayes-beat)` in your terminal.


### Installing bayesbeat

Once you have activated the environment, install `bayesbeat` by running the following command in the root directory of the repository

```
pip install .
```

All the necessary dependencies should already be installed, so this should be quite quick.

## Running the analysis

Before running the analysis, make sure you have activated the environment. See the section above for instructions.

Here is an example of how to the run analysis for a data file called `PyTotalAnalysis.mat` located in `data/`. The index determines which of ringdowns in the data file will be analyzed.

### Creating an ini file

Create an `.ini` file with a given name, e.g. `example.ini`

```
bayesbeat_create_ini example.ini
```
Open the new ini file and set the values for the different fields. You must specify `output` and `datafile`, the other settings will all have defaults that should work.
The most important are:

* `indices`: determines which ringdowns in the data file will be analyzed. Only used if running via Condor. If `None` or `'all'` all indices will be analysed. Otherwise, should be a list of integers (starting at 0).
* `n-pool`: the number of cores to use. We recommend setting this to at least 4.

The file should look something like this:

```
[General]
output = "outdir/"
label = "disk_0"
datafile = "data/PyTotalAnalysis.mat"
indices = None
seed = 1234
plot = True

[Model]
rescale-amplitude = False
maximum-amplitude = None

[Analysis]
n-pool = 4
resume = True

[Sampler]
nlive = 1000
reset_flow = 8

[HTCondor]
request-disk = "2GB"
request-memory = "2GB"
request-cpus = None
```

### Running with HTCondor

The recommended way to use `bayesbeat` is on a cluster with HTCondor installed.
This allows analyses to run in parallel rather than one-by-one on a local machine.
To run a local machine, see the section below.

Once you have a create an ini file, the analyses can be prepared (built) and then submitted.
To do so run

```
bayesbeat_build example.ini
```

this will construct a condor DAG which can be submitted using the command that is printed after the command has run.
Alternatively, if you run

```
bayesbeat_build example.ini --submit
```

the analysis will be built and submitted in a single step.

**Note:** if the output directory already exists, an error will be raised and the analysis will not be built or submitted.

### Running on a local machine (without HTCondor)

To run an analysis locally instead of via HTCondor, use the following command

```
bayesbeat_run example.ini --index 0
```

where `--index` specifies which ringdown in the datafile to analyse.

**Note:** this ignores the value of `indices` in the ini file.

**Note:** it is not possible to analyse multiple ringdowns with a single call to `bayesbeat_run`.

