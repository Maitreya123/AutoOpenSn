#!/usr/bin/env bash
# Set up the TAMU Orchard cluster (node class01) so AutoOpenSn can run real
# OpenSn there. Run it from this repository, on your own machine, with the TAMU
# VPN connected:
#
#     scripts/setup_cluster.sh --netid <your NetID>
#
# It does everything that can be done unattended, in order, and is safe to run
# again: each step checks whether it is already done before doing it.
#
#   1. checks this machine can reach Orchard (the VPN is the usual culprit)
#   2. creates an SSH key for Orchard, if there is none
#   3. adds `orchard` and `class01` to ~/.ssh/config, if they are not there
#   4. installs the key on Orchard          <- asks for your NetID password once
#   5. accepts class01's host key, so automated runs are never asked
#   6. builds OpenSn on class01 at the commit AutoOpenSn's templates target,
#      with the Python module, into ~/opensn/build   (about two minutes)
#   7. installs IPython and matplotlib for your user on class01
#   8. runs the Reed problem through AutoOpenSn and checks it against the
#      value OpenSn's own regression suite records
#
# What it cannot do for you: connect the VPN, type your password, or create
# your Orchard account — ask whoever administers the cluster for that.
#
# The commit and module names are read from the autoopensn package rather than
# repeated here, so this script and the app cannot disagree about them. That
# means the package must be installed first: pip install -e ".[dev,api]".

set -euo pipefail
cd "$(dirname "$0")/.."

ORCHARD_IP="128.194.17.172"
ORCHARD_NAME="orchard.engr.tamu.edu"
CLASS01_NAME="nuen-orch-class001"
KEY="$HOME/.ssh/orchard_ed25519"
CONFIG="$HOME/.ssh/config"

NETID=""
JOBS=32
LOCAL_ONLY=0

usage() {
    sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --netid NETID   your TAMU NetID (required)
  --jobs N        parallel compile jobs on class01 (default 32: all physical
                  cores, half the logical ones, on a machine others share)
  --local-only    do steps 1-3 only, on this machine
  -h, --help      this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --netid) NETID="${2:-}"; shift 2 ;;
        --jobs) JOBS="${2:-}"; shift 2 ;;
        --local-only) LOCAL_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }
ok()   { printf '    \033[32mok\033[0m  %s\n' "$1"; }
note() { printf '    ..  %s\n' "$1"; }
fail() { printf '\n\033[31mstopped:\033[0m %s\n' "$1" >&2; exit 1; }

[ -n "$NETID" ] || fail "pass your NetID: scripts/setup_cluster.sh --netid <netid>"
case "$JOBS" in ''|*[!0-9]*) fail "--jobs must be a whole number" ;; esac

# The python that has autoopensn's dependencies installed. PYTHON overrides.
# Importing the package alone proves nothing here: this script runs from the
# repository root, where `import autoopensn` succeeds from the folder itself
# whether or not anything is installed. Its dependencies are what can be
# missing, so those are what is checked.
usable() { "$1" -c "import autoopensn.study, autoopensn.runner.remote" >/dev/null 2>&1; }
if [ -n "${PYTHON:-}" ]; then
    usable "$PYTHON" || fail "PYTHON=$PYTHON cannot import autoopensn and its dependencies.
    Install them into it with: $PYTHON -m pip install -e \".[dev,api]\""
else
    for candidate in python3 python3.12 python3.11 python; do
        if command -v "$candidate" >/dev/null 2>&1 && usable "$candidate"; then
            PYTHON="$candidate"; break
        fi
    done
    [ -n "${PYTHON:-}" ] || fail "no python here has autoopensn's dependencies. Install them first:
        pip install -e \".[dev,api]\"
    or point PYTHON at the interpreter that has them."
fi

SETTINGS=$("$PYTHON" -c '
from autoopensn import PINNED_OPENSN_COMMIT
from autoopensn.runner.remote import DEFAULT_MODULES, DEFAULT_MODULE_PATH, DEFAULT_SOURCE
print(PINNED_OPENSN_COMMIT, DEFAULT_MODULE_PATH, ",".join(DEFAULT_MODULES), DEFAULT_SOURCE)
') || fail "could not read the pinned commit and module names from autoopensn"
read -r PIN MODULE_PATH MODULES REMOTE_SOURCE <<< "$SETTINGS"
[ ${#PIN} -eq 40 ] || fail "autoopensn reported an unexpected commit: '$PIN'"

# --- 1 ------------------------------------------------------------------------
step 1 "Can this machine reach Orchard?"
reachable() {
    # Bounded: with the VPN down the packets are dropped, not refused, and an
    # unbounded connect waits over a minute before saying anything.
    if command -v nc >/dev/null 2>&1; then
        nc -z -w 6 "$ORCHARD_IP" 22 >/dev/null 2>&1
    else
        (exec 3<>"/dev/tcp/$ORCHARD_IP/22") 2>/dev/null
    fi
}
if reachable; then
    ok "Orchard answers on port 22"
else
    fail "cannot reach Orchard ($ORCHARD_IP, port 22).
    Off campus, connect the TAMU VPN first: engr.tamu.edu drops outside traffic
    silently, so this shows up as a timeout rather than a refusal."
fi

# --- 2 ------------------------------------------------------------------------
step 2 "SSH key"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
if [ -f "$KEY" ]; then
    ok "already have $KEY"
else
    # No passphrase: AutoOpenSn runs ssh unattended and never prompts. Anyone
    # who can read your home directory can use this key, so keep it private.
    ssh-keygen -q -t ed25519 -N "" -f "$KEY" -C "autoopensn-orchard-$(date +%Y%m%d)"
    ok "created $KEY"
fi

# --- 3 ------------------------------------------------------------------------
step 3 "~/.ssh/config entries"
touch "$CONFIG" && chmod 600 "$CONFIG"
has_host() { grep -Eq "^[Hh]ost[[:space:]]+(.*[[:space:]])?$1([[:space:]]|\$)" "$CONFIG"; }
if has_host orchard; then
    ok "'Host orchard' already present; left as it is"
else
    cat >> "$CONFIG" <<EOF

# Orchard front end (TAMU NUEN). Added by AutoOpenSn's setup_cluster.sh.
# By address, because the TAMU VPN's resolver does not answer for
# $ORCHARD_NAME; HostKeyAlias keeps the host key filed under the name.
Host orchard
  HostName $ORCHARD_IP
  HostKeyAlias $ORCHARD_NAME
  User $NETID
  IdentityFile $KEY
  ServerAliveInterval 60
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h-%p
  ControlPersist 10m
EOF
    ok "added 'Host orchard'"
fi
if has_host class01; then
    ok "'Host class01' already present; left as it is"
else
    cat >> "$CONFIG" <<EOF

# class01, reachable only through orchard. Added by setup_cluster.sh.
# ControlMaster reuses one connection: a study makes several SSH calls per
# case, and the login node refuses bursts of new ones.
Host class01
  HostName $CLASS01_NAME
  User $NETID
  ProxyJump orchard
  IdentityFile $KEY
  ServerAliveInterval 60
  ControlMaster auto
  ControlPath ~/.ssh/cm-class01-%r@%h-%p
  ControlPersist 10m
EOF
    ok "added 'Host class01'"
fi

if [ "$LOCAL_ONLY" = 1 ]; then
    printf '\nLocal steps done (--local-only).\n'
    exit 0
fi

# --- 4 ------------------------------------------------------------------------
step 4 "Key installed on Orchard"
quiet_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }
if quiet_ssh orchard true; then
    ok "key login to Orchard works"
else
    [ -t 0 ] || fail "the key is not on Orchard yet, and installing it needs your password,
    which this script will only ask for in an interactive terminal. Run it from one."
    note "installing the key — enter your NetID password when asked"
    ssh-copy-id -i "$KEY.pub" -o HostKeyAlias="$ORCHARD_NAME" -o StrictHostKeyChecking=accept-new \
        "$NETID@$ORCHARD_IP" \
        || fail "ssh-copy-id did not succeed.
    If your password is refused although it is right, your account may not exist
    on the Orchard login node yet — ask the cluster administrator."
    quiet_ssh orchard true || fail "the key was installed but key login still fails"
    ok "key login to Orchard works"
fi

# --- 5 ------------------------------------------------------------------------
step 5 "class01 reachable without prompts"
# accept-new records class01's host key the first time. Without it, every
# automated connection fails with "Host key verification failed", because
# BatchMode cannot ask.
if quiet_ssh class01 true; then
    ok "class01 answers, and its host key is recorded"
else
    fail "reached Orchard but not class01. Your account may not include class01,
    or Orchard's home directory may not be shared with it."
fi

# --- 6, 7 ---------------------------------------------------------------------
step 6 "OpenSn on class01 at ${PIN:0:7}, with the Python module"
note "this is a shared machine: the load is checked first, and a busy node stops the build"

# The remote program arrives on stdin, so nothing in it is re-parsed by the
# login shell — ssh joins its arguments with spaces and the far side splits
# them again, which is how quoting goes wrong.
ssh -o BatchMode=yes class01 "bash -l -s -- $PIN $JOBS $MODULE_PATH $MODULES $REMOTE_SOURCE" <<'REMOTE'
set -euo pipefail
PIN="$1"; JOBS="$2"; MODULE_PATH="$3"; MODULES="$4"; SRC="$HOME/$5"

export MODULEPATH="$MODULE_PATH:${MODULEPATH:-}"
# In this order: the site Python is linked against MPI, which only the OpenSn
# module puts on the library path. The other way round, Python fails silently.
for module in ${MODULES//,/ }; do module load "$module"; done

changed=0
if [ -d "$SRC/.git" ]; then
    cd "$SRC"
    if [ "$(git rev-parse HEAD)" != "$PIN" ]; then
        if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
            echo "    !!  $SRC has local changes; not touching it. Move it aside and rerun." >&2
            exit 11
        fi
        git fetch -q origin
        git -c advice.detachedHead=false checkout -q "$PIN"
        changed=1
    fi
    echo "    ok  $SRC is at ${PIN:0:7}"
else
    git clone -q https://github.com/open-sn/opensn "$SRC"
    cd "$SRC"
    git -c advice.detachedHead=false checkout -q "$PIN"
    changed=1
    echo "    ok  cloned OpenSn into $SRC at ${PIN:0:7}"
fi

if [ "$changed" = 0 ] && ls "$SRC"/build/pyopensn/__init__*.so >/dev/null 2>&1; then
    echo "    ok  already built"
else
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$load" -v j="$JOBS" 'BEGIN { exit !(l + j > 64) }'; then
        echo "    !!  class01 is busy (load $load). Building now with -j$JOBS would oversubscribe it." >&2
        echo "        Try again later, or pass a smaller --jobs." >&2
        exit 10
    fi
    echo "    ok  load $load; building with -j$JOBS"
    mkdir -p "$SRC/build" && cd "$SRC/build"
    # The Python module is OFF by default, and every AutoOpenSn template
    # imports pyopensn. A default build compiles cleanly and runs nothing.
    if ! cmake .. -DOPENSN_WITH_PYTHON_MODULE=ON -DPython_EXECUTABLE="$(command -v python3)" > cmake.log 2>&1; then
        echo "    !!  cmake failed; the end of $SRC/build/cmake.log:" >&2
        tail -15 cmake.log >&2; exit 12
    fi
    start=$(date +%s)
    if ! make -j"$JOBS" > make.log 2>&1; then
        echo "    !!  the build failed; first errors from $SRC/build/make.log:" >&2
        grep -m5 -iE "error[: ]" make.log >&2 || tail -15 make.log >&2; exit 13
    fi
    echo "    ok  built in $(( $(date +%s) - start ))s"
fi

echo
printf '\033[1m[7] Python packages the tutorials use\033[0m\n'
# Every tutorial ends with a Jupyter-only shutdown block that imports IPython,
# and some plot with matplotlib. Without them a run prints its answer and then
# crashes. Installed for this user only, under ~/.local.
python3 -m pip install --user --quiet --disable-pip-version-check ipython matplotlib
# Importing pyopensn prints OpenSn's own start and end banners. Only this
# script's lines are shown when it works; everything is shown when it does not,
# so a failure is never filtered away.
if ! report=$(PYTHONPATH="$SRC/build:${PYTHONPATH:-}" python3 - 2>&1 <<'PY'
import IPython, matplotlib, pyopensn
print(f"    ok  pyopensn imports; IPython {IPython.__version__}, matplotlib {matplotlib.__version__}")
try:
    import ctypes  # noqa: F401
except ImportError:
    print("    ..  this Python has no _ctypes: keigen_to_transient and")
    print("        source_step_steady_to_transient_to_steady cannot run until the")
    print("        cluster's Python is rebuilt with libffi. Everything else can.")
PY
); then
    echo "    !!  the check on class01 failed:" >&2
    printf '%s\n' "$report" >&2
    exit 14
fi
printf '%s\n' "$report" | grep -E '^    ' || true
REMOTE

# --- 8 ------------------------------------------------------------------------
step 8 "A real run, checked against OpenSn's own answer"
"$PYTHON" - <<'PY'
import sys
from pathlib import Path
from autoopensn import PINNED_OPENSN_COMMIT
from autoopensn.parse.gold import compare_to_gold
from autoopensn.runner import RemoteRunner, RunRequest, RunnerError, run_directory
from autoopensn.templates import load_template, render_run_script

runner = RemoteRunner()
try:
    runner.preflight()
except RunnerError as exc:
    sys.exit(f"    !!  preflight failed: {exc}")
template = load_template("reed_1d")
request = RunRequest(
    case_id="reed_1d",
    script=render_run_script(template, {}, pack_commit=PINNED_OPENSN_COMMIT,
                             spec_name="setup_check", case_id="reed_1d"),
    directory=run_directory(Path("runs"), "setup_check", "reed_1d"),
    template="reed_1d", num_procs=1, timeout_seconds=300,
)
result = runner.run(request)
gold = compare_to_gold(result.stdout, template, template.defaults())
if not result.ok or not gold.passed:
    sys.exit(f"    !!  the check run did not match: {result.failure_reason() or gold.reason}")
for check in gold.checks:
    print(f"    ok  {check['key']} {check['actual']}  (OpenSn's recorded value: {check['expected']})")
print(f"    ok  simulation time on class01: {result.wall_time:.2f}s")
PY

printf '\n\033[32mclass01 is ready.\033[0m Studies started from the web page run there\n'
printf 'whenever the VPN is connected; without it they fall back to the built-in solver.\n'
