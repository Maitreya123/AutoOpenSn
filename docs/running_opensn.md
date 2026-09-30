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

### Reaching the node

Orchard's compute nodes are only reachable through the front end, so put the
jump in `~/.ssh/config` rather than doing it by hand. Then `scp` and everything
else inherits it:

```
Host orchard
  HostName orchard.engr.tamu.edu
  User <netid>
  IdentityFile ~/.ssh/orchard_ed25519

Host class01
  HostName nuen-orch-class001
  User <netid>
  ProxyJump orchard
  IdentityFile ~/.ssh/orchard_ed25519
```

Off campus, the TAMU VPN is required; `engr.tamu.edu` silently drops outside
traffic, so the symptom is a connection timeout rather than a refusal.

`RemoteRunner` runs with `BatchMode=yes` and never prompts, so **key-based login
is required**. Install the key once with
`ssh-copy-id -i ~/.ssh/orchard_ed25519.pub <netid>@orchard.engr.tamu.edu`. A
study of seven cases that stops on case one waiting for a password is worse than
one that refuses to start.

### Building OpenSn there

The module supplies the dependencies; OpenSn itself you build once, and with 32
cores it is minutes rather than hours:

```shell
ssh class01
export MODULEPATH=/scratch-local/software/modulefiles:$MODULEPATH
module load opensn/gcc/15

git clone https://github.com/open-sn/opensn
cd opensn && mkdir build && cd build
cmake .. && make -j64
```

`module avail` lists the alternatives; an `opensn/clang` module exists too.
`RemoteRunner.config()` records the module list in the cache key, so results
from two toolchains are never conflated.

class01 is shared. 32 physical cores, 128 GB, and other people on it — so keep
`num_procs` proportionate and do not leave sweeps running unattended. For many
long runs, ask the administrator to put them through Orchard's batch queues
instead.

### Checking it works

```shell
autoopensn run study.yaml --runner remote --host class01
```

`RemoteRunner.preflight()` runs first and separates the two failures that look
alike: it could not connect, or it connected and the module did not load. The
second prints what it tried, because the usual cause is a module name that has
moved.

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
