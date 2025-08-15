- **Title:** Snapshot Compressive Imaging with Foreground-Background Separation

# Main idea

**Message/hypothesis:** Assuming that a snapshot image can be decomposed into background +
foreground, such that

1. background sequence is low-rank;
2. the foreground is sparse and has patch similarity
   enables reconstructing sequences with higher quality and faster than DeSCI
   and is competitive with PnP approaches.

# Code Release

The codebase is structured as follows:

## Folder: Desci

In this folder there exists the main implementation of this project.

`automation.py`: Main entrance to the program. You can manually change the data
being used by modifying the file itself. There is an example on how to load and
generate a CASIA signal as well as the KOBE test signal. It automatically logs
outputs in a folder called `out/`.

```bash
python automation.py --help:
usage: automation.py [-h] [-r RHO] [-f FRAMES] [-b BLOCK] [-i ITERATIONS] [-p PATCH_SIZE] [-s STRIDE_RATIO] lambda_0 lambda_1 lambda_2 lambda_3

positional arguments:
  lambda_0
  lambda_1
  lambda_2
  lambda_3

options:
  -h, --help            show this help message and exit
  -r RHO, --rho RHO
  -f FRAMES, --frames FRAMES
  -b BLOCK, --block BLOCK
  -i ITERATIONS, --iterations ITERATIONS
  -p PATCH_SIZE, --patch_size PATCH_SIZE
  -s STRIDE_RATIO, --stride_ratio STRIDE_RATIO
```


For example: 

```bash
python automation.py 1 10 50 50 -b 0.5 -f 4 -i 500
```

`lr_sparse_admm.py`: Contains the implementation of ADMM and the entire
algorithm.

`utils/`: This folder contains multiple utility files including code about patch
forming, physics, dataloaders, as well as experimental features (mainly related
to blockmatching) that were not mentioned in the report. Also contains code that
was used for making visualizations of signals in the report.

`tests/`: Contains the unit tests written, as well as the scripts we used for
checking that the updates of the algorithm were correct.

`separation.py`: Includes experimentation done for separating background and
foreground with RPCA. This is the code used for generating some figures of the
report.

`rpca_plus_datafidelity.py`: This file includes the code for an experiment only
mentioned in the future work part of the report, where a robust PCA + data
fidelity term problem was solved with ADMM. When tweaking lamb0,lamb1,lamb2 it
can achieve very similar reconstruction performance, sometimes even faster than
our main method.

`run_lowpriority.sh`: Makes use of SLURM job arrays to perform a grid search
using params.txt. You can tweak to manually add extra flags. Uses the
`lowpriority` queue to run for free on Cirrus.

`params.txt`: The file from which parameters are read from for grid searches.
You can easily generate that through a list comprehension. Configure based on
help from `automation.py`

## Matlab-eval

`matlabgen.py`: Is used to generate multiple config files to use with the code
release of [DeSCI](https://github.com/liuyang12/DeSCI).

These files need to be placed in the repository from the official release of
DeSCI to be used:

`test_desci_custom.m`: A script modified from the repository that allowed
injection of our own files. Just change the `datapath` variable and run.

`run.sh`: SLURM script written to launch a job on cirrus running the above file.

## E2E

Contains an end to end model replicating the work of [DL-CACTI](https://pubs.aip.org/aip/app/article/5/3/030801/570340/Deep-learning-for-video-compressive-sensing).

Code is complete, needed training for multiple compression ratio pairs, which is
expensive, and it wasn't mentioned in the report.

## Documents

- This [document](admm-lr-plus-sparse/admm-lr-plus-sparse.pdf) formulates the
  reconstruction problem, applies ADMM to solve it, and details each step.
- `admm-lr-plus-sparse-old/` and `z-admm-nolonger-relevant/` contain
  formulations that were not used in the project.
