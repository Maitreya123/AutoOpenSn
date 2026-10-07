# Running real OpenSn

AutoOpenSn does not need OpenSn. The full test suite passes without it, and the
1D scenarios compute for real in the reference solver. You need OpenSn to run
the other 39 scenarios, and to produce numbers that are OpenSn's rather than
this package's own.

There are two ways to get it, and the first is much cheaper than the second.

## 1. A cluster where it is already built (preferred)

Someone has already spent the afternoon compiling PETSc, VTK and HDF5. Use
their build.

At TAMU this is the NUEN **Orchard** cluster, whose node `class01`
(`nuen-orch-class001`) carries an `opensn/gcc/15` module. `RemoteRunner` targets
exactly this arrangement: it copies the rendered script over, loads the module
in a login shell, runs `mpiexec`, and brings stdout back for the ordinary
parser.

### Setting it up: one command

You need an Orchard account that includes class01 — the cluster administrator
creates it — and, off campus, the TAMU VPN connected. Then, from this
repository, after installing it:

```shell
scripts/setup_cluster.sh --netid <your NetID>
```

It does the rest, in order, and is safe to run again: every step checks
whether it is already done first, so a rerun after a dropped connection picks
up where it stopped. You type your NetID password once, when it installs your
key. It finishes by running the Reed problem on class01 through AutoOpenSn and
checking it against the value OpenSn's own regression suite records, so when it
says the cluster is ready, a real run has already succeeded.

| Step | What it does |
| --- | --- |
| 1 | Checks this machine can reach Orchard |
| 2 | Creates `~/.ssh/orchard_ed25519`, if there is no such key |
| 3 | Adds `orchard` and `class01` to `~/.ssh/config`, leaving existing entries alone |
| 4 | Installs the key on Orchard — the one password prompt |
| 5 | Records class01's host key, so automated runs are never asked to |
| 6 | Builds OpenSn on class01 into `~/opensn/build` — about two minutes |
| 7 | Installs IPython and matplotlib for your user on class01 |
| 8 | Runs Reed through AutoOpenSn and checks it against gold |

Options: `--jobs N` for fewer compile jobs on a busy node, `--local-only` for
steps 1 to 3. The steps below are what it does, for doing them by hand or
working out why one failed.

### Reaching the node, by hand

**A key**, because `RemoteRunner` runs with `BatchMode=yes` and never prompts. A
study of seven cases that stops on case one waiting for a password is worse
than one that refuses to start.

```shell
ssh-keygen -t ed25519 -N "" -f ~/.ssh/orchard_ed25519
```

**Two entries in `~/.ssh/config`.** Orchard's compute nodes are reachable only
through the front end, so the jump goes here, where `scp` and everything else
inherits it.

```
Host orchard
  HostName 128.194.17.172
  HostKeyAlias orchard.engr.tamu.edu
  User <netid>
  IdentityFile ~/.ssh/orchard_ed25519
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h-%p
  ControlPersist 10m

Host class01
  HostName nuen-orch-class001
  User <netid>
  ProxyJump orchard
  IdentityFile ~/.ssh/orchard_ed25519
  ControlMaster auto
  ControlPath ~/.ssh/cm-class01-%r@%h-%p
  ControlPersist 10m
```

By address rather than by name, because **the TAMU VPN's resolver does not
answer for `orchard.engr.tamu.edu`**: with the VPN connected, the name fails
with "Could not resolve hostname" while the address works. `HostKeyAlias` keeps
the host key filed under the name, so it still verifies if the address changes.

`ControlMaster` matters more than it looks. A study makes several SSH calls per
case, and the login node refuses bursts of new connections — running every
scenario once had it refuse nine in a row. With it, one connection carries
everything. `RemoteRunner` also retries a connection that failed before
anything ran, with back-off.

**The key on Orchard**, which asks for your password once:

```shell
ssh-copy-id -i ~/.ssh/orchard_ed25519.pub -o HostKeyAlias=orchard.engr.tamu.edu <netid>@128.194.17.172
```

**class01's host key**, recorded once, because `BatchMode` cannot answer the
"are you sure you want to continue connecting" question, and every automated
connection then fails with "Host key verification failed":

```shell
ssh -o StrictHostKeyChecking=accept-new class01 true
```

Home directories are shared between the login node and class01, so the key
installed on one works on both.

### Building OpenSn there

The module supplies the compiler, MPI and the other dependencies; OpenSn itself
you build once, into your home directory, which the login node and class01
share. Measured on class01: **1 minute 49 seconds at `-j32`, peak 12.9 GB**.

```shell
ssh class01
export MODULEPATH=/scratch-local/software/modulefiles:$MODULEPATH
module load opensn/gcc/15
module load python3/3.12.3          # after opensn: see below

git clone https://github.com/open-sn/opensn ~/opensn
cd ~/opensn
git checkout 2fd4a19ceade4f8da581a5029c68f474d3333ccb
mkdir build && cd build
cmake .. -DOPENSN_WITH_PYTHON_MODULE=ON
make -j32
```

Three departures from the generic build instructions, each of which produces a
build that compiles cleanly and then cannot run a single AutoOpenSn script:

- **`-DOPENSN_WITH_PYTHON_MODULE=ON`.** It defaults to `OFF`, which builds the
  console application only, and every template imports `pyopensn`. The shared
  build already on the node under `opensn/clang/21.1.0` was configured this
  way, which is why it cannot be used.
- **The pinned commit.** The templates are checked character for character
  against `2fd4a19`. The shared clang build is seven months older, and running
  the Reed problem on it gets through the solve and then fails on
  `ComputeBalanceTable`, the call that produces the regression gold values.
- **`python3/3.12.3` loaded after `opensn/gcc/15`.** The system Python is 3.6.8.
  The site's 3.12 is linked against `libmpi.so.12` and cannot start until the
  OpenSn module has put MPI on the library path; loaded first, it fails without
  a message and leaves the system Python in place.

`-j32` rather than the `-j64` the generic instructions suggest: it is all 32
physical cores, compilation gains little from hyperthreading, and it leaves the
other half of a shared machine to whoever else is on it.

The build puts the compiled extension at
`build/pyopensn/__init__.cpython-312-x86_64-linux-gnu.so`, so it is the build
directory that goes on `PYTHONPATH`. `RemoteRunner` does that, and the module
loads, in the order above, on every run.

**Two Python packages**, for your user only:

```shell
python3 -m pip install --user ipython matplotlib     # with both modules loaded
```

Every tutorial ends with a Jupyter-only shutdown block that imports IPython, and
about ten plot with matplotlib. The site Python has neither, so without them a
run prints its answer and then crashes. They are installed rather than the
templates edited, because editing a template would break its
character-for-character match with the upstream tutorial.

**One thing that cannot be fixed from here**: the site Python 3.12 was built
without `_ctypes`, so `keigen_to_transient` and
`source_step_steady_to_transient_to_steady` fail on import. It needs the
cluster's Python rebuilt with `libffi`, which is the administrator's to do.
Every other scenario runs.

`RemoteRunner.config()` records the module list in the cache key, so results
from two toolchains are never conflated.

class01 is shared. 32 physical cores, 128 GB, and other people on it — so keep
`num_procs` proportionate and do not leave sweeps running unattended. For many
long runs, ask the administrator to put them through Orchard's batch queues
instead.

### Checking it works

Every scenario has been run this way, once, at its defaults: 37 of 41 reproduce
OpenSn's recorded answers exactly, and none gives a wrong one. The other four,
and the bugs the first pass turned up, are in
[cluster_verification.md](cluster_verification.md).

```shell
autoopensn run study.yaml --runner remote --host class01
```

`RemoteRunner.preflight()` runs first and separates the two failures that look
alike: it could not connect, or it connected and the module did not load. The
second prints what it tried, because the usual cause is a module name that has
moved.

### When something fails

Every one of these was hit for real while setting class01 up.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Operation timed out` reaching Orchard | Off campus without the VPN | Connect the TAMU VPN |
| `Could not resolve hostname orchard.engr.tamu.edu` | The VPN's resolver | Use the address, as in the config above |
| `Permission denied (publickey,password)` with the right password | No account on the login node yet | Ask the administrator |
| `Host key verification failed` | class01's key never recorded | `ssh -o StrictHostKeyChecking=accept-new class01 true` |
| `Connection refused` or `Connection closed by UNKNOWN port 65535` partway through | The login node throttling bursts of connections | `ControlMaster`, above; `RemoteRunner` retries these itself |
| `ModuleNotFoundError: No module named 'pyopensn'` | Built without the Python module, or modules loaded in the wrong order | Rebuild with `-DOPENSN_WITH_PYTHON_MODULE=ON`; load `opensn/gcc/15` before `python3/3.12.3` |
| A run prints its answer, then `No module named 'IPython'` | The packages above are missing | `pip install --user ipython matplotlib` |
| `No module named '_ctypes'` | The site Python's build | The administrator; two scenarios are affected |

`RemoteRunner.preflight()` separates the first few from the rest: it says
whether it could not connect, or connected and could not load OpenSn.

## 2. Building it yourself

Only worth it if you have no cluster. Follow the upstream instructions at
`doc/source/install/install.rst` in the OpenSn checkout, which are kept current
by the project and were previously transcribed here — a copy that could only go
stale.

Budget one to three hours and 12–18 GB, most of it PETSc and VTK. On macOS with
Homebrew the dependencies are `gcc python git cmake open-mpi flex doxygen
pandoc`. Then use `--runner local`, which invokes `mpiexec` on this machine.

## Which Python surface

OpenSn ships two incompatible Python surfaces, and the choice matters:

- the **console application** `opensn`, where every class is preloaded into
  `__main__`, and
- the **module** `pyopensn`, which you `import` normally.

AutoOpenSn's templates are written for the **module**, because the regression
scripts they derive from are too, and because the module works with `mpi4py`.
Both `LocalMPIRunner` and `RemoteRunner` invoke `mpiexec -n N python script.py`,
which requires the module. Build with `-DOPENSN_WITH_PYTHON_MODULE=ON`.

The regression suite itself needs **both**, so a build intended to run it needs
the module flag on a console build.

## What AutoOpenSn will not do

Nothing in this package builds, installs, or submits anything on your behalf.
`RemoteRunner` copies a script to a directory you named and runs it; it creates
no modules, installs no software, and cleans up nothing it did not create.

A workflow tool that compiles a scientific code unattended on a shared machine
is a workflow tool nobody should trust.
