# SPDX-License-Identifier: BSD-2-Clause
"""MPI x OpenMP launch planning for the native engine.

A launch is described by a :class:`LaunchLayout` (MPI ranks x OpenMP threads
per rank).  :func:`mpiexec_command` turns it into an ``mpiexec`` command line
with an explicit, disjoint per-rank CPU binding: with Open MPI 5 the binding is
``--map-by ppr:N:package:pe=T --bind-to core``, which gives every rank ``T``
whole physical cores inside one package.  :func:`plan_binding` validates the
layout against the detected :class:`CpuTopology` before anything is started,
and :func:`verify_binding` launches a tiny probe under the same arguments and
checks that the observed CPU masks are pairwise disjoint and ``T`` cores wide.

:func:`launch_environment` fixes the OpenMP contract (``OMP_NUM_THREADS``,
``OMP_PROC_BIND=close``, ``OMP_PLACES=cores``, no dynamic teams, one active
level) and pins third-party BLAS pools to one thread. The thread wait policy
can be overridden in the launch environment.
:func:`run_bounded_process` runs a command in its own process group with a
timeout and cooperative cancellation.  :func:`configure_resources` records the
explicit operator-workspace and address-space limits of a run.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
from time import monotonic
from typing import Any, Literal, Mapping, Sequence

from .authoring import ResourceRow

BindingPolicy = Literal["core", "none"]

_SYSFS_CPU = Path("/sys/devices/system/cpu")
_MAXIMUM_INT64 = 2**63 - 1
_MAXIMUM_INT32 = 2**31 - 1


class LaunchError(ValueError):
    """A launch layout, binding, environment or resource request is invalid."""


def _positive_int(value: Any, label: str, maximum: int = _MAXIMUM_INT64) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise LaunchError(f"{label} must be a positive integer <= {maximum}")
    return value


@dataclass(frozen=True)
class LaunchLayout:
    """MPI rank count and OpenMP threads per rank of one native launch."""

    ranks: int
    threads: int = 1

    def __post_init__(self) -> None:
        _positive_int(self.ranks, "MPI rank count", _MAXIMUM_INT32)
        _positive_int(self.threads, "OpenMP thread count", _MAXIMUM_INT32)

    @property
    def cores(self) -> int:
        """Physical cores claimed by the whole launch."""
        return self.ranks * self.threads

    @classmethod
    def parse(cls, text: str) -> "LaunchLayout":
        """Parse ``"RxT"`` (for example ``"4x8"``) into a layout."""
        parts = text.lower().split("x")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise LaunchError(f"layout must have the form RxT, got {text!r}")
        return cls(int(parts[0]), int(parts[1]))

    def __str__(self) -> str:
        return f"{self.ranks}x{self.threads}"


@dataclass(frozen=True)
class CpuTopology:
    """Physical cores available to this process, grouped by package.

    Each core is identified by the lowest logical CPU number among its
    hardware threads; ``hardware_threads`` maps that representative to all
    logical CPUs of the core.
    """

    packages: tuple[tuple[int, ...], ...]
    hardware_threads: Mapping[int, tuple[int, ...]]

    @property
    def package_count(self) -> int:
        """Number of packages (sockets) with at least one usable core."""
        return len(self.packages)

    @property
    def core_count(self) -> int:
        """Number of usable physical cores."""
        return sum(len(cores) for cores in self.packages)

    @property
    def cores_per_package(self) -> int:
        """Usable physical cores of the smallest package."""
        return min(len(cores) for cores in self.packages)

    def core_of(self, cpu: int) -> int:
        """Return the representative physical core of logical CPU ``cpu``."""
        for core, siblings in self.hardware_threads.items():
            if cpu in siblings:
                return core
        raise LaunchError(f"logical CPU {cpu} is not part of the topology")

    @classmethod
    def detect(
        cls,
        allowed: Sequence[int] | None = None,
        *,
        sysfs: Path = _SYSFS_CPU,
    ) -> "CpuTopology":
        """Read the Linux sysfs topology restricted to the allowed CPUs."""
        cpus = sorted(os.sched_getaffinity(0) if allowed is None else set(allowed))
        if not cpus:
            raise LaunchError("no usable CPUs")
        grouped: dict[tuple[int, int], list[int]] = {}
        for cpu in cpus:
            topology = sysfs / f"cpu{cpu}" / "topology"
            try:
                package = int((topology / "physical_package_id").read_text())
                core = int((topology / "core_id").read_text())
            except (OSError, ValueError):
                package, core = 0, cpu
            grouped.setdefault((package, core), []).append(cpu)
        packages: dict[int, list[int]] = {}
        threads: dict[int, tuple[int, ...]] = {}
        for (package, _core), members in sorted(grouped.items()):
            representative = min(members)
            packages.setdefault(package, []).append(representative)
            threads[representative] = tuple(sorted(members))
        return cls(
            packages=tuple(
                tuple(sorted(cores)) for _, cores in sorted(packages.items())
            ),
            hardware_threads=threads,
        )


@dataclass(frozen=True)
class BindingPlan:
    """Validated per-rank binding of one layout on one topology."""

    layout: LaunchLayout
    policy: BindingPolicy
    ranks_per_package: int
    arguments: tuple[str, ...]


def plan_binding(
    layout: LaunchLayout,
    topology: CpuTopology | None = None,
    *,
    policy: BindingPolicy = "core",
) -> BindingPlan:
    """Validate ``layout`` and derive Open MPI 5 mapping/binding arguments.

    With ``policy="core"`` ranks are spread evenly over packages and each rank
    is bound to ``layout.threads`` whole cores of its package, so rank masks
    are disjoint.  The layout must fit: ``ceil(ranks / packages) * threads``
    cores per package.  ``policy="none"`` performs no binding and only checks
    that the layout does not oversubscribe the usable cores.
    """
    if policy not in ("core", "none"):
        raise LaunchError(f"unsupported binding policy: {policy!r}")
    topology = CpuTopology.detect() if topology is None else topology
    if layout.cores > topology.core_count:
        raise LaunchError(
            f"layout {layout} needs {layout.cores} cores but only "
            f"{topology.core_count} physical cores are usable"
        )
    if policy == "none":
        return BindingPlan(layout, policy, 0, ("--bind-to", "none"))
    per_package = -(-layout.ranks // topology.package_count)
    if per_package * layout.threads > topology.cores_per_package:
        raise LaunchError(
            f"layout {layout} needs {per_package} ranks x {layout.threads} cores "
            f"per package but packages have {topology.cores_per_package} cores"
        )
    return BindingPlan(
        layout,
        policy,
        per_package,
        (
            "--map-by",
            f"ppr:{per_package}:package:pe={layout.threads}",
            "--bind-to",
            "core",
        ),
    )


def openmp_environment(threads: int) -> dict[str, str]:
    """Return the OpenMP and threading variables of a CPU-only launch."""
    _positive_int(threads, "OpenMP thread count", _MAXIMUM_INT32)
    return {
        "OMP_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "GOMP_SPINCOUNT": "20000",
        "BLIS_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMPI_MCA_opal_cuda_support": "false",
    }


def launch_environment(
    base: Mapping[str, str],
    *,
    threads: int,
    runtime_library_path: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Build the environment for a CPU-only native launch.

    ``runtime_library_path`` is prepended to ``LD_LIBRARY_PATH`` (the installed
    engine carries no RPATH).  Open MPI's CUDA-awareness layer is disabled
    because the supported execution contract is CPU-only and would otherwise
    probe every host buffer on collective operations.

    Inherited ``OMP_WAIT_POLICY`` and ``GOMP_SPINCOUNT`` override waiting
    defaults. A policy without a spin count lets the runtime choose its own
    spin count.
    """
    environment = dict(base)
    prior = environment.get("LD_LIBRARY_PATH", "")
    requested = "" if runtime_library_path is None else os.fspath(runtime_library_path)
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (requested, prior) if value
    )
    thread_environment = openmp_environment(threads)
    if "OMP_WAIT_POLICY" in base:
        del thread_environment["OMP_WAIT_POLICY"]
    if "OMP_WAIT_POLICY" in base or "GOMP_SPINCOUNT" in base:
        del thread_environment["GOMP_SPINCOUNT"]
    environment.update(thread_environment)
    return environment


def mpiexec_command(
    command: Sequence[str | os.PathLike[str]],
    layout: LaunchLayout,
    *,
    mpiexec: str | os.PathLike[str] = "mpiexec",
    numproc_flag: str = "-n",
    binding: BindingPolicy = "core",
    topology: CpuTopology | None = None,
) -> list[str]:
    """Wrap ``command`` in an ``mpiexec`` launch of ``layout``.

    A one-rank launch without binding executes ``command`` directly.
    """
    rendered = [os.fspath(item) for item in command]
    if not rendered:
        raise LaunchError("launch command is empty")
    plan = plan_binding(layout, topology, policy=binding)
    if layout.ranks == 1 and binding == "none":
        return rendered
    return [
        os.fspath(mpiexec),
        numproc_flag,
        str(layout.ranks),
        *plan.arguments,
        *rendered,
    ]


def engine_run_options(
    *,
    run_directory: str | os.PathLike[str] | None = None,
    journal: str | os.PathLike[str] | None = None,
    write_journal: bool = True,
    progress: bool = True,
    checkpoint_every: int = 0,
    checkpoint_seconds: int = 0,
    checkpoint_keep: int = 1,
) -> list[str]:
    """Render the operational ``slope-engine run`` options.

    These options only control the live journal, stderr progress and in-run
    accepted-boundary checkpoints; they never change numerical results or the
    run identity.  Defaults match the engine defaults and render no option.
    """
    if checkpoint_every < 0 or checkpoint_seconds < 0 or not 1 <= checkpoint_keep <= 32:
        raise LaunchError("engine checkpoint options are invalid")
    options: list[str] = []
    if run_directory is not None:
        options += ["--run-directory", os.fspath(run_directory)]
    if not write_journal:
        options.append("--no-journal")
    elif journal is not None:
        options += ["--journal", os.fspath(journal)]
    if not progress:
        options.append("--no-progress")
    if checkpoint_every:
        options += ["--checkpoint-every", str(checkpoint_every)]
    if checkpoint_seconds:
        options += ["--checkpoint-seconds", str(checkpoint_seconds)]
    if checkpoint_keep != 1:
        options += ["--checkpoint-keep", str(checkpoint_keep)]
    return options


_BINDING_PROBE = (
    "import json, os\n"
    "rank = os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('PMIX_RANK'))\n"
    "print(json.dumps({'rank': rank, 'cpus': sorted(os.sched_getaffinity(0))}))\n"
)


@dataclass(frozen=True)
class BindingReport:
    """Observed per-rank CPU masks of one bound launch."""

    layout: LaunchLayout
    rank_cores: Mapping[int, tuple[int, ...]]

    @property
    def disjoint(self) -> bool:
        """Whether no physical core is shared between two ranks."""
        cores = [core for owned in self.rank_cores.values() for core in owned]
        return len(cores) == len(set(cores))


def verify_binding(
    layout: LaunchLayout,
    *,
    mpiexec: str | os.PathLike[str] = "mpiexec",
    numproc_flag: str = "-n",
    topology: CpuTopology | None = None,
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float = 60.0,
) -> BindingReport:
    """Launch a probe under the ``core`` binding and validate its masks.

    Raises:
        LaunchError: The probe failed, a rank is missing, a rank owns other
            than ``layout.threads`` physical cores, or two ranks share a core.
    """
    topology = CpuTopology.detect() if topology is None else topology
    command = mpiexec_command(
        (sys.executable, "-I", "-c", _BINDING_PROBE),
        layout,
        mpiexec=mpiexec,
        numproc_flag=numproc_flag,
        binding="core",
        topology=topology,
    )
    completed = run_bounded_process(
        command,
        environment=dict(os.environ if environment is None else environment),
        timeout_seconds=timeout_seconds,
    )
    if completed.exit_code != 0 or completed.timed_out:
        raise LaunchError(f"binding probe failed: {completed.stderr.strip()}")
    rank_cores: dict[int, tuple[int, ...]] = {}
    for line in completed.stdout.splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        rank = int(record["rank"])
        cores = sorted({topology.core_of(int(cpu)) for cpu in record["cpus"]})
        rank_cores[rank] = tuple(cores)
    report = BindingReport(layout, rank_cores)
    if sorted(rank_cores) != list(range(layout.ranks)):
        raise LaunchError(f"binding probe reported ranks {sorted(rank_cores)}")
    for rank, cores in rank_cores.items():
        if len(cores) != layout.threads:
            raise LaunchError(
                f"rank {rank} is bound to {len(cores)} cores, expected {layout.threads}"
            )
    if not report.disjoint:
        raise LaunchError("rank CPU bindings overlap")
    return report


def _local_memory_per_node_mib() -> int | None:
    """Read the current Linux physical and cgroup v2 memory headroom."""
    mib = 1024**2
    available: list[int] = []
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                available.append(int(line.split()[1]) // 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        memberships = Path("/proc/self/cgroup").read_text(encoding="ascii")
        group = next(
            line.removeprefix("0::").lstrip("/")
            for line in memberships.splitlines()
            if line.startswith("0::")
        )
    except (OSError, StopIteration):
        group = ""
    root = Path("/sys/fs/cgroup")
    directory = root / group
    for node in (directory, *directory.parents):
        if node != root and root not in node.parents:
            break
        try:
            limit = (node / "memory.max").read_text(encoding="ascii").strip()
            if limit != "max":
                used = int((node / "memory.current").read_text(encoding="ascii"))
                available.append(max(1, (int(limit) - used) // mib))
        except (OSError, ValueError):
            continue
    return min(available) if available else None


def local_resource_row(
    layout: LaunchLayout,
    *,
    coarse_active_ranks: int | None = None,
    petsc_threads: int = 1,
    binding: BindingPolicy = "core",
    workspace_mib: int | None = None,
    address_space_bytes: int | None = None,
) -> ResourceRow:
    """Describe a single-node launch of ``layout`` as an authoring resource row.

    The row records the binding and OpenMP contract applied by
    :func:`mpiexec_command` and :func:`launch_environment`; it becomes part of
    the resolved run identity.  The first ``coarse_active_ranks`` ranks
    (default: all) own the coarsest PMG level.  ``petsc_threads`` records
    PETSc's internal thread count, which cannot exceed the OpenMP team.
    """
    if type(petsc_threads) is not int or petsc_threads < 1:
        raise LaunchError("PETSc thread count must be a positive integer")
    if petsc_threads > layout.threads:
        raise LaunchError("PETSc threads cannot exceed the rank OpenMP team")
    if binding not in ("core", "none"):
        raise LaunchError(f"unsupported binding policy: {binding!r}")
    if coarse_active_ranks is None:
        coarse_active_ranks = layout.ranks
    if type(coarse_active_ranks) is not int or not (
        1 <= coarse_active_ranks <= layout.ranks
    ):
        raise LaunchError("coarse active ranks must be within the rank count")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if address_space_bytes is not None:
        _positive_int(address_space_bytes, "address-space limit")
        if hard != resource.RLIM_INFINITY and address_space_bytes > hard:
            raise LaunchError(
                "requested address-space limit exceeds the inherited hard limit"
            )
        soft = address_space_bytes
    return ResourceRow(
        memory_per_node_mib=_local_memory_per_node_mib(),
        operator_workspace_mib=workspace_mib,
        address_space_per_rank_mib=(
            None if soft == resource.RLIM_INFINITY else max(1, soft // 1024**2)
        ),
        name=f"local-{layout.ranks}x{layout.threads}",
        nodes=1,
        world_size=layout.ranks,
        ranks_per_node=layout.ranks,
        openmp_threads=layout.threads,
        cores_per_rank=layout.threads,
        smt="off",
        cpu_binding=binding,
        memory_binding="local",
        numa_policy="local",
        mpi_thread_level="funneled",
        openmp={
            "dynamic": False,
            "max_active_levels": 1,
            "proc_bind": "close",
            "places": "cores",
        },
        third_party_threads={"blas": 1, "hypre": 1, "petsc": petsc_threads},
        active_ranks={
            "fine": tuple(range(layout.ranks)),
            "coarse": tuple(range(coarse_active_ranks)),
        },
    )


@dataclass(frozen=True)
class BoundedProcessResult:
    """Captured outcome of one process-group-bounded command."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    cancelled: bool

    @property
    def termination_signal(self) -> int | None:
        """Signal that terminated the process, or ``None`` after a normal exit."""
        return -self.exit_code if self.exit_code < 0 else None


def _descendants(root: int, *, leaves_only: bool = False) -> list[int]:
    """PIDs of every live descendant of ``root`` (empty without ``/proc``)."""
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as stream:
                fields = stream.read().rsplit(b")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(entry))
        except (OSError, IndexError, ValueError):
            continue
    found: list[int] = []
    pending = list(children.get(root, ()))
    while pending:
        pid = pending.pop()
        if not leaves_only or pid not in children:
            found.append(pid)
        pending.extend(children.get(pid, ()))
    return found


def _signal(pids: Sequence[int], number: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, number)
        except ProcessLookupError:
            pass


def _terminate_group(
    process: subprocess.Popen[str], grace_seconds: float
) -> tuple[str | None, str | None]:
    """Stop a launched run, giving its application processes the grace period.

    The engine stops at its next collective safe point on SIGTERM and publishes
    a cancelled result.  MPI launchers (Open MPI's ``prte``) abort the whole job
    when they receive SIGTERM themselves and place the ranks in their own
    process groups, so the launcher's descendants are signalled first and the
    launcher's process group only when they do not finish in time.
    """
    descendants = _descendants(process.pid, leaves_only=True)
    if descendants:
        _signal(descendants, signal.SIGTERM)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        remaining = _descendants(process.pid)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _signal(remaining, signal.SIGKILL)
        return process.communicate()


def run_bounded_process(
    command: Sequence[str | os.PathLike[str]],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
    termination_grace_seconds: float = 10.0,
    cancellation_event: threading.Event | None = None,
    capture_output: bool = True,
) -> BoundedProcessResult:
    """Run ``command`` in a new process group with bounded cleanup.

    On timeout or cancellation the launched processes' descendants (the MPI
    ranks) receive ``SIGTERM`` first so that the engine can stop at a safe point
    and publish a cancelled result; after the grace period the process group
    and any remaining descendants receive ``SIGKILL``. Use
    ``capture_output=False`` to retain live, inherited output streams.
    """
    rendered = tuple(os.fspath(item) for item in command)
    if (
        not rendered
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0.0
        or not math.isfinite(termination_grace_seconds)
        or termination_grace_seconds <= 0.0
    ):
        raise LaunchError("bounded process policy is invalid")
    process = subprocess.Popen(
        rendered,
        env=dict(environment),
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    cancelled = False
    stdout = stderr = ""
    deadline = monotonic() + timeout_seconds
    try:
        while True:
            if cancellation_event is not None and cancellation_event.is_set():
                cancelled = True
                break
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                timed_out = True
                break
            try:
                stdout, stderr = process.communicate(timeout=min(remaining, 0.25))
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        _terminate_group(process, termination_grace_seconds)
        raise
    if timed_out or cancelled:
        grace = (
            min(termination_grace_seconds, 1.0)
            if cancelled
            else termination_grace_seconds
        )
        stdout, stderr = _terminate_group(process, grace)
    return BoundedProcessResult(
        command=rendered,
        exit_code=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        timed_out=timed_out,
        cancelled=cancelled,
    )


def validate_resources(record: Any) -> dict[str, Any]:
    """Validate an execution-resource record produced by
    :func:`configure_resources` and return it unchanged."""
    if not isinstance(record, dict) or set(record) != {
        "operator_workspace_mib",
        "address_space_limit_bytes",
    }:
        raise LaunchError("invalid execution resource record")
    workspace = record["operator_workspace_mib"]
    if workspace is not None:
        _positive_int(workspace, "operator workspace MiB", _MAXIMUM_INT32)
    limits = record["address_space_limit_bytes"]
    if not isinstance(limits, dict) or set(limits) != {"soft", "hard"}:
        raise LaunchError("invalid address-space limits")
    for name, value in limits.items():
        if value is not None:
            _positive_int(value, f"address-space {name} limit")
    if limits["hard"] is not None and (
        limits["soft"] is None or limits["soft"] > limits["hard"]
    ):
        raise LaunchError("address-space soft limit exceeds hard limit")
    return record


def configure_resources(
    environment: Mapping[str, str],
    *,
    workspace_mib: int | None = None,
    address_space_bytes: int | None = None,
) -> dict[str, Any]:
    """Apply and record the explicit resource limits of this launcher.

    Call it before creating threads: the address-space limit is set on the
    current process (``RLIMIT_AS``) and inherited by every child.  A recorded
    ``None`` limit means unlimited.  ``workspace_mib=None`` keeps the engine's
    default operator-workspace cap.  An inherited non-empty ``PETSC_OPTIONS``
    is rejected so that hidden options cannot change a recorded run policy.

    Returns:
        A record with ``operator_workspace_mib`` and the effective
        ``address_space_limit_bytes``.
    """
    if environment.get("PETSC_OPTIONS", "").strip():
        raise LaunchError(
            "inherited PETSC_OPTIONS are rejected; use the explicit workspace option"
        )
    if workspace_mib is not None:
        _positive_int(workspace_mib, "operator workspace MiB", _MAXIMUM_INT32)
    if address_space_bytes is not None:
        _positive_int(address_space_bytes, "address-space limit")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if address_space_bytes is not None:
        if hard != resource.RLIM_INFINITY and address_space_bytes > hard:
            raise LaunchError(
                "requested address-space limit exceeds the inherited hard limit"
            )
        resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, hard))
        soft = address_space_bytes
    return validate_resources(
        {
            "operator_workspace_mib": workspace_mib,
            "address_space_limit_bytes": {
                "soft": None if soft == resource.RLIM_INFINITY else soft,
                "hard": None if hard == resource.RLIM_INFINITY else hard,
            },
        }
    )


__all__ = [
    "BindingPlan",
    "BindingPolicy",
    "BindingReport",
    "BoundedProcessResult",
    "CpuTopology",
    "LaunchError",
    "LaunchLayout",
    "configure_resources",
    "engine_run_options",
    "launch_environment",
    "local_resource_row",
    "mpiexec_command",
    "openmp_environment",
    "plan_binding",
    "run_bounded_process",
    "validate_resources",
    "verify_binding",
]
