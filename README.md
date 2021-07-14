# Episimlab
[![Run tox tests](https://github.com/eho-tacc/episimlab/actions/workflows/tox.yml/badge.svg)](https://github.com/eho-tacc/episimlab/actions/workflows/tox.yml)

Episimlab is a framework for developing epidemiological models in a modular fashion. It provides a set of extensible, separable components that can be combined with user-written components, allowing for rapid development of reproducible disease-modeling pipelines.

## Highlights

* Build epidemiological models in a modular fashion:

```python
import xsimlab
import episimlab as esl
import my_module

my_model = xsimlab.Model({
    # Read a synthetic population from a CSV file
    'initialize_synthetic_population': esl.setup.SyntheticPopFromCSV,
    # Allow travel between zip codes
    'simulate_travel': esl.travel.TravelBetweenNodes,
    # A classical SEIR model
    'SEIR': esl.seir.SixteenCompartmentSEIR,
    # Add your own custom process! This process could modify
    # beta, the transmission coefficient in the SEIR model.
    'modify_beta': my_module.MyCustomProcess,
})
```

* **Fully integrated with the [`xarray-simlab`](https://xarray-simlab.readthedocs.io/) package -** Episimlab provides a library of [`xsimlab.process` ]() classes ("processes"), as well as a handful of commonly-used [`xsimlab.model`]()s ("models").
* **Extensible -** Users can quickly and easily develop their own process classes, either from scratch or based on an `episimlab` process, and include them in their models.
* **Any variable can be high-dimensional -** Want constant `beta` for all time points? Use the `ConstantBeta` process. Want `beta` with age and risk group structure? Use `AgeRiskBeta`. Want to add your own custom structure and read it from a CSV file? Quickly write your own process, and it will integrate with processes provided by `episimlab`. Better yet, push your new process in a [Pull Request](CONTRIBUTING.md) so that others can use it!
* **Good performance under the hood -** Frequently used processes - such as force of infection (FOI) calculation, SEIR disease progression, and travel between spatially separated nodes - are written in C-accelerated Python (Cython) with OpenMP support. This results in 100-1000X speed-up compared to the equivalent process written in pure Python.

## Installation

### Quick Start

1. Make a local clone of this repository, e.g. using `git clone`.
2. Install python dependencies:
```bash
pip install -r requirements.txt
pip install "dask[distributed]" "dask[dataframe]"
```
3. Install [GNU GSL](#install-gnu-gsl), which is necessary to run the C-accelerated model engine.
4. Install the Episimlab package using pip:
```bash
pip install .

# Alternatively, run the setup.py
python setup.py install
```

### Install GNU GSL

1. You might already have GSL installed on your system. To check, run `gsl-config` in the shell.
2. Install GSL
    * To install GSL on Mac OS X:
    ```bash
    # Install using HomeBrew
    brew install gsl

    # Check that gsl-config is in the $PATH
    gsl-config --version
    ```
    * To install on Ubuntu, use `apt-get`:
    ```bash
    apt-get install libgsl-dev
    ```
    * To install GSL 2.6 from source (for most Linux distributions):
    ```bash
    wget -O gsl-2.6.tar.gz ftp://ftp.gnu.org/gnu/gsl/gsl-2.6.tar.gz
    tar -xf gsl-2.6.tar.gz
    cd gsl-2.6
    ./configure
    make
    make check
    make install
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(realpath ./lib)"

    # Check that gsl-config is in the $PATH
    gsl-config --version
    ```
    * TACC HPC
        * GSL is already installed on most TACC HPC systems, and can be loaded using the [Lmod system](https://frontera-portal.tacc.utexas.edu/user-guide/admin/#using-modules-to-manage-your-environment):
        ```bash
        module load gsl/2.6
        gsl-config --version
        ```
        * Tested on Stampede2 and Frontera
3. For the most up-to-date instructions on how to install GSL, please consult the [GSL documentation](https://www.gnu.org/software/gsl/doc/html/).

### Troubleshooting

* [GNU GSL][1] is not installed
    * _Problem_: Attempting to install (e.g. with pip, conda, or setup.py) throws:
    ```
    Exception: GNU GSL does not appear to be installed; could not find `gsl-config` in $PATH. Please add `gsl-config` to your $PATH, or install GNU GSL if necessary. To install GSL on Mac OS X: brew install gsl.
    ```
    * _Fix_
        1. Ensure that you have [installed GNU GSL](#install-gnu-gsl).
        2. Check that `gsl-config` is in your `$PATH` and add it if necessary:
        ```bash
        # Should print location of GSL installation
        gsl-config --prefix

        # Append to PATH if necessary, and retry above command
        export PATH="$PATH:/path/to/gsl-2.6/bin"
        ```

* `libgsl.so` is not in the `$LD_LIBRARY_PATH`
    * _Problem_: Installation proceeds smoothly, but attempting to import Episimlab throws `ImportError`:
    ```
    ImportError: libgsl.so.25: cannot open shared object file: No such file or directory
    ```
    * _Fix_
        1. Ensure that you have [installed GNU GSL](#install-gnu-gsl).
        2. Check that your `$LD_LIBRARY_PATH` points to the `lib` directory in your installation of GSL. Append to the `$LD_LIBRARY_PATH` if necessary:
        ```bash
        # One of these paths should be /**/gsl*/lib
        echo $LD_LIBRARY_PATH

        # If the above env does not point to gsl lib directory, append to the LD_LIBRARY_PATH
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/path/to/gsl-2.6/lib"
        ```

### Install from [PyPI](https://pypi.org/)

_Work in progress_

### [Docker](https://www.docker.com/) Image

_Work in progress_

## Testing

Preferred testing environment runs poetry virtual env within tox.
1. Install [tox](https://tox.readthedocs.io/) and [poetry](https://python-poetry.org/)
2. Run tox from repository root:
```bash
# Default args
tox
# Pass args to pytest. In this case, we use 4-thread parallelism to run only the test_setup suite
tox -- tests/test_setup
```

[1]: https://www.gnu.org/software/gsl/
