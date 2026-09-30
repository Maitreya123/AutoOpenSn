# Installing OpenSn on macOS

Transcribed from `doc/source/install/install.rst` in the OpenSn checkout at
commit `2fd4a19ceade4f8da581a5029c68f474d3333ccb`, with the macOS branch of each
tab selected.

**Nothing in AutoOpenSn runs any of this.** These steps are written down so that
someone can follow them deliberately, on a machine they have chosen, after
reading them. Building OpenSn takes an afternoon and installs a compiler
toolchain, an MPI implementation, and PETSc, which is not something a workflow
tool should do on anyone's behalf.

You do not need OpenSn to use AutoOpenSn. The full test suite passes without it,
and studies run against recorded fixtures. You need it to produce new numbers.

## What you are installing

OpenSn ships two incompatible Python surfaces, and you have to choose:

- the **console application** `opensn`, where every class is preloaded into
  `__main__`, and
- the **module** `pyopensn`, which you `import` normally.

AutoOpenSn's templates are written for the **module**, because they are derived
from the regression scripts, which are too, and because the module is compatible
with `mpi4py`. `LocalMPIRunner` invokes `mpiexec -n N python script.py`, which
requires the module.

Note that the regression suite itself requires **both**, so if you want to run
it you need `-DOPENSN_WITH_PYTHON_MODULE=ON` on a console build.

## 1. Development tools

```shell
brew install gcc python git cmake open-mpi flex doxygen pandoc
export NPROC=$(sysctl -n hw.ncpu)
```

The documentation states these requirements:

| Requirement | Version |
| --- | --- |
| C++ compiler | `clang++` or `g++` with C++20 support |
| Python | 3.9 or newer, with `pip` and `pybind11` |
| CMake | 3.29 or newer |
| MPI | OpenMPI, MPICH, or MVAPICH |
| `flex` | required by PETSc's PTSCOTCH component |
| Pandoc, Doxygen | only to build the documentation |

CMake 3.29 is worth checking explicitly: `cmake --version`. The top-level
`CMakeLists.txt` requires it, and an older Homebrew CMake fails at configure
time with a message that does not say so plainly.

## 2. A virtual environment

Recommended by the OpenSn documentation, and worth doing: the module installs
into `site-packages`, and you want that to be removable.

```shell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pybind11
```

Activate it in every shell where you build or run OpenSn, including the one
where you run `autoopensn run --runner local`.

## 3. Clone

```shell
git clone https://github.com/Open-Sn/opensn.git
cd opensn
git checkout 2fd4a19ceade4f8da581a5029c68f474d3333ccb
```

The checkout is pinned deliberately. Every template and gold value in
AutoOpenSn is tied to that commit, and the PyOpenSn API has changed across
versions. If you build a different commit, the templates may still render, and
the scripts may still run, and the numbers may still be wrong.

## 4. Dependencies

This is the long step. CMake looks for compatible system installations and
downloads, builds, and installs whatever is missing.

```shell
mkdir build_deps && cd build_deps
cmake -DCMAKE_INSTALL_PREFIX=/path/to/dependencies /path/to/opensn/tools/dependencies
make -j
cd ..
rm -rf build_deps
```

Installed if not already present: PETSc 3.17+, Boost 1.86+, HDF5 1.14+,
VTK 9.3+, and Caliper 2.11+.

It generates an environment script. Source it before every build, and add it to
your shell startup file to make it stick:

```shell
source /path/to/dependencies/bin/set_opensn_env.sh
```

## 5. Build the module

```shell
pip install .
```

or, to get the extra packages the regression tests need:

```shell
pip install .[dev]
```

For the console application instead:

```shell
mkdir build && cd build
cmake ..
make -j$NPROC
```

To build both, which the regression suite requires:

```shell
cmake -DOPENSN_WITH_PYTHON_MODULE=ON ..
```

## 6. Verify

The OpenSn regression suite:

```shell
cd /path/to/opensn
test/run_tests -d test/python -j$NPROC -v 1 -w 3
```

Then AutoOpenSn's own check, which renders the Reed template and runs it for
real:

```shell
cd /path/to/AutoOpenSn
AUTOOPENSN_ALLOW_LOCAL_RUNS=1 pytest tests/test_runner.py -k real_opensn -v
```

That test is skipped unless the environment variable is set, and it skips
itself again if `pyopensn` is not importable. It runs the Reed problem at
default parameters, which is exactly `reed_balance.py`, so its gold values are
`Absorption=100.6178` and `OutFlow=0.3821562`.

## 7. Re-record the fixtures

Once a real run works, replace the hand-written fixtures. They are the one part
of this repository that is currently fiction, and the acceptance test becomes a
claim about physics the moment they are real.

```shell
autoopensn run tests/data/gmres_convergence.yaml --runner local
```

Then follow `tests/fixtures/README.md`.

## Alternatives to building

The OpenSn repository also provides:

- a Dockerfile under `distribution/docker`, with instructions in
  `distribution/docker/README.md`,
- a Spack package repository under `distribution/spack`, with instructions in
  `distribution/spack/README.md`.

Both are described in the upstream documentation as supported paths. Neither has
been tried for this project.
