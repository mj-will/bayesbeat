# BayesBeat

Bayesian analysis of ringdowns.

**Warning:** the following instructions have only been tested on Linux and may be updated after testing on other platforms.

## Installation

Before installing `nessai` you'll need to download and install `conda`. See the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). If running on computing cluster you will not have to do this step.

Then open a terminal, i.e. PowerShell, Command Line or equivalent.

### Creating a conda environment

We start by creating an environment from the environment file, this will install all the necessary packages:

```
conda env create -f environment.yml
```

The environment will be called `bayes-beat`.

### Activating the environment

We must activate the environment to use the installed packages:

```
conda activate bayes-beat
```

You should now see `(bayes-beat)` in your terminal.

## Running the analysis

Before running the analysis, make sure you have activated the environment. See the section above for instructions.


### Running without HTCondor

Here is an example of how to the run analysis for a data file called `PyTotalAnalysis.mat` located in `data/`. The index determines which of ringdowns in the data file will be analysed.

```
python run_nessai.py --datafile=data/PyTotalAnalysis.mat --index=0 --outdir=outdir/ --rescale --n-pool=4
```

`--rescale` indicates that the code will rescale the maximum amplitude to be 1 and `--n-pool=4` defines the number of cores to use. You can configure the output directory by setting `outdir`.

### Running with HTCondor

You will also need to make sure the file `run_nessai.py` has the correct permissions, you'll only need to do this once.

```
chmod u+x run_nessai.py
```

To submit a run for a specific data index simply run the following on cluster with condor configured:

```
bash submit_run.sh <datafile> <index> <output directory>
```

where you should replace `<text>` with the relevant information.

You may need to add accounting tags if running on an LDG cluster.
