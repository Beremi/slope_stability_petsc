# SPDX-License-Identifier: BSD-2-Clause
"""Launch layout, CPU binding, environment, bounded-process and resource tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from slope_stability.launch import (
    CpuTopology,
    LaunchError,
    LaunchLayout,
    configure_resources,
    launch_environment,
    mpiexec_command,
    openmp_environment,
    plan_binding,
    run_bounded_process,
    validate_resources,
    verify_binding,
)

ROOT = Path(__file__).resolve().parents[2]


def _topology(packages: int, cores: int, smt: bool = True) -> CpuTopology:
    """Synthetic topology with ``cores`` cores per package and optional SMT."""
    package_cores = []
    threads = {}
    total = packages * cores
    for package in range(packages):
        members = tuple(range(package * cores, (package + 1) * cores))
        package_cores.append(members)
        for core in members:
            threads[core] = (core, core + total) if smt else (core,)
    return CpuTopology(packages=tuple(package_cores), hardware_threads=threads)


class LaunchLayoutContract(unittest.TestCase):
    def test_layout_parsing_and_validation(self) -> None:
        self.assertEqual(LaunchLayout.parse("4x8"), LaunchLayout(4, 8))
        self.assertEqual(str(LaunchLayout(2, 3)), "2x3")
        self.assertEqual(LaunchLayout(4, 8).cores, 32)
        for text in ("4", "4x", "x8", "4x8x1", "-1x2", "ax2"):
            with self.subTest(text=text), self.assertRaises(LaunchError):
                LaunchLayout.parse(text)
        for ranks, threads in ((0, 1), (1, 0), (True, 1), (1.5, 1)):
            with self.subTest(ranks=ranks, threads=threads):
                with self.assertRaises(LaunchError):
                    LaunchLayout(ranks, threads)  # type: ignore[arg-type]

    def test_binding_plan_is_disjoint_and_validated(self) -> None:
        single = _topology(1, 32)
        plan = plan_binding(LaunchLayout(4, 8), single)
        self.assertEqual(
            plan.arguments,
            ("--map-by", "ppr:4:package:pe=8", "--bind-to", "core"),
        )
        dual = _topology(2, 16)
        plan = plan_binding(LaunchLayout(4, 8), dual)
        self.assertEqual(plan.ranks_per_package, 2)
        self.assertIn("ppr:2:package:pe=8", plan.arguments)
        with self.assertRaisesRegex(LaunchError, "physical cores"):
            plan_binding(LaunchLayout(8, 8), single)
        # Three ranks of 10 threads fit 32 cores in total, but two of them
        # would have to share one 16-core package.
        with self.assertRaisesRegex(LaunchError, "per package"):
            plan_binding(LaunchLayout(3, 10), dual)
        unbound = plan_binding(LaunchLayout(3, 10), dual, policy="none")
        self.assertEqual(unbound.arguments, ("--bind-to", "none"))
        with self.assertRaises(LaunchError):
            plan_binding(LaunchLayout(1, 1), single, policy="socket")  # type: ignore

    def test_mpiexec_command(self) -> None:
        topology = _topology(1, 32)
        command = mpiexec_command(
            ("slope-engine", "run", "p", "r", "o"),
            LaunchLayout(2, 4),
            mpiexec="/opt/mpi/bin/mpiexec",
            topology=topology,
        )
        self.assertEqual(
            command,
            [
                "/opt/mpi/bin/mpiexec",
                "-n",
                "2",
                "--map-by",
                "ppr:2:package:pe=4",
                "--bind-to",
                "core",
                "slope-engine",
                "run",
                "p",
                "r",
                "o",
            ],
        )
        direct = mpiexec_command(
            ("slope-engine",), LaunchLayout(1, 4), binding="none", topology=topology
        )
        self.assertEqual(direct, ["slope-engine"])
        with self.assertRaises(LaunchError):
            mpiexec_command((), LaunchLayout(1, 1), topology=topology)

    def test_topology_detection_matches_affinity(self) -> None:
        topology = CpuTopology.detect()
        allowed = os.sched_getaffinity(0)
        self.assertEqual(
            {cpu for cpus in topology.hardware_threads.values() for cpu in cpus},
            allowed,
        )
        self.assertGreaterEqual(topology.core_count, 1)
        for package in topology.packages:
            for core in package:
                self.assertEqual(topology.core_of(core), core)


class LaunchEnvironmentContract(unittest.TestCase):
    def test_cpu_only_environment(self) -> None:
        base = {
            "LD_LIBRARY_PATH": "/existing/lib",
            "OMP_NUM_THREADS": "99",
            "OMPI_MCA_opal_cuda_support": "true",
            "UNRELATED": "preserved",
        }
        environment = launch_environment(
            base, threads=8, runtime_library_path=Path("/runtime/lib")
        )
        self.assertEqual(base["OMPI_MCA_opal_cuda_support"], "true")
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/runtime/lib:/existing/lib")
        expected = {
            "OMP_NUM_THREADS": "8",
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
        self.assertEqual(openmp_environment(8), expected)
        for name, value in expected.items():
            self.assertEqual(environment[name], value)
        self.assertEqual(environment["UNRELATED"], "preserved")
        self.assertEqual(launch_environment({}, threads=1)["LD_LIBRARY_PATH"], "")
        with self.assertRaises(LaunchError):
            openmp_environment(0)
        with self.assertRaises(LaunchError):
            launch_environment(base, threads=0)

    def test_thread_wait_overrides(self) -> None:
        active = launch_environment({"OMP_WAIT_POLICY": "ACTIVE"}, threads=8)
        self.assertEqual(active["OMP_WAIT_POLICY"], "ACTIVE")
        self.assertNotIn("GOMP_SPINCOUNT", active)

        spin = launch_environment({"GOMP_SPINCOUNT": "100"}, threads=8)
        self.assertEqual(spin["GOMP_SPINCOUNT"], "100")
        self.assertEqual(
            spin["OMP_WAIT_POLICY"], openmp_environment(8)["OMP_WAIT_POLICY"]
        )

        overrides = {"OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0"}
        passive = launch_environment(overrides, threads=8)
        self.assertEqual({name: passive[name] for name in overrides}, overrides)


class BoundedProcessContract(unittest.TestCase):
    def test_timeout_and_cancellation_kill_the_process_group(self) -> None:
        sleeper = (sys.executable, "-c", "import time; time.sleep(60)")
        bounded = run_bounded_process(
            sleeper,
            environment=os.environ,
            timeout_seconds=0.05,
            termination_grace_seconds=0.05,
        )
        self.assertTrue(bounded.timed_out)
        self.assertFalse(bounded.cancelled)
        self.assertIn(bounded.termination_signal, {signal.SIGTERM, signal.SIGKILL})
        cancellation = threading.Event()
        timer = threading.Timer(0.05, cancellation.set)
        timer.start()
        try:
            cancelled = run_bounded_process(
                sleeper,
                environment=os.environ,
                timeout_seconds=60.0,
                termination_grace_seconds=0.05,
                cancellation_event=cancellation,
            )
        finally:
            timer.cancel()
            timer.join()
        self.assertTrue(cancelled.cancelled)
        self.assertFalse(cancelled.timed_out)
        self.assertIn(cancelled.termination_signal, {signal.SIGTERM, signal.SIGKILL})
        finished = run_bounded_process(
            (sys.executable, "-c", "print('ok')"),
            environment=os.environ,
            timeout_seconds=30.0,
        )
        self.assertEqual((finished.exit_code, finished.stdout), (0, "ok\n"))
        self.assertIsNone(finished.termination_signal)
        for timeout in (0.0, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(LaunchError):
                run_bounded_process(
                    sleeper, environment=os.environ, timeout_seconds=timeout
                )

    def test_descendants_are_signalled_before_the_launcher(self) -> None:
        """A launcher that aborts on SIGTERM must not preempt its ranks."""
        with tempfile.TemporaryDirectory(prefix="lch-graceful-") as raw:
            marker = Path(raw) / "rank-finished"
            launcher = (
                "import os, signal, sys, time\n"
                "child = os.fork()\n"
                "if child == 0:\n"
                "    os.setpgid(0, 0)\n"
                "    rank = os.fork()\n"
                "    if rank != 0:\n"
                "        sys.exit(os.waitstatus_to_exitcode(os.waitpid(rank, 0)[1]))\n"
                "    def stop(*_):\n"
                f"        open({str(marker)!r}, 'w').write('published')\n"
                "        os._exit(0)\n"
                "    signal.signal(signal.SIGTERM, stop)\n"
                "    time.sleep(60)\n"
                "    os._exit(1)\n"
                "def abort(*_):\n"
                "    os.kill(child, signal.SIGKILL)\n"
                "    os._exit(3)\n"
                "signal.signal(signal.SIGTERM, abort)\n"
                "sys.exit(os.waitstatus_to_exitcode(os.waitpid(child, 0)[1]))\n"
            )
            bounded = run_bounded_process(
                (sys.executable, "-c", launcher),
                environment=os.environ,
                timeout_seconds=1.0,
                termination_grace_seconds=10.0,
            )
            self.assertTrue(bounded.timed_out)
            self.assertEqual(bounded.exit_code, 0)
            self.assertEqual(marker.read_text(), "published")


class ExecutionResourceContract(unittest.TestCase):
    """QAL-CONTRACT-RESOURCES-001: explicit, recorded launcher limits."""

    def test_capture_and_explicit_limits(self) -> None:
        with (
            patch("resource.getrlimit", return_value=(-1, -1)),
            patch("resource.setrlimit") as setter,
        ):
            record = configure_resources(
                {}, workspace_mib=4096, address_space_bytes=32 * 1024**3
            )
            setter.assert_called_once()
            self.assertEqual(
                record["address_space_limit_bytes"],
                {"soft": 32 * 1024**3, "hard": None},
            )
            self.assertEqual(record["operator_workspace_mib"], 4096)
        with (
            patch("resource.getrlimit", return_value=(16 * 1024**3, -1)),
            patch("resource.setrlimit") as setter,
        ):
            record = configure_resources({})
            setter.assert_not_called()
            self.assertEqual(record["address_space_limit_bytes"]["soft"], 16 * 1024**3)
            self.assertIsNone(record["operator_workspace_mib"])

    def test_reject_hidden_options_and_invalid_limits(self) -> None:
        for options in [
            "-ksp_type preonly",
            "-native_mech_operator_memory_cap_mib 4096",
        ]:
            with self.subTest(options=options), self.assertRaises(LaunchError):
                configure_resources({"PETSC_OPTIONS": options})
        for value in [0, -1, True, 1.5]:
            with self.subTest(value=value), self.assertRaises(LaunchError):
                configure_resources({}, workspace_mib=value)
        with (
            patch("resource.getrlimit", return_value=(1024, 2048)),
            patch("resource.setrlimit") as setter,
        ):
            with self.assertRaises(LaunchError):
                configure_resources({}, address_space_bytes=4096)
            setter.assert_not_called()
        with patch("resource.getrlimit", return_value=(-1, -1)):
            good = configure_resources({})
        for bad in [
            dict(good, petsc_options="-ksp_type cg"),
            dict(good, operator_workspace_mib=True),
            dict(good, address_space_limit_bytes={"soft": 4096, "hard": 2048}),
            dict(good, extra=1),
        ]:
            with self.subTest(bad=bad), self.assertRaises(LaunchError):
                validate_resources(bad)

    def test_child_inherits_address_space_limit(self) -> None:
        code = """import json, os, subprocess, sys
from slope_stability.launch import configure_resources
record = configure_resources(
    {}, workspace_mib=4096, address_space_bytes=8 * 1024**3
)
child = (
    'import resource, os, json; '
    'print(json.dumps([resource.getrlimit(resource.RLIMIT_AS)[0], '
    'os.environ.get("PETSC_OPTIONS")]))'
)
output = subprocess.check_output(
    [sys.executable, '-c', child],
    env=os.environ,
    text=True,
)
print(output.strip())
"""
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "python/src"))
        environment.pop("PETSC_OPTIONS", None)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), [8 * 1024**3, None])


class BindingProbeContract(unittest.TestCase):
    """Observed per-rank binding under mpiexec (needs MPI and 4 free cores)."""

    def test_bound_ranks_own_disjoint_cores(self) -> None:
        mpiexec = os.environ.get("SLOPE_MPIEXEC") or shutil.which("mpiexec")
        if mpiexec is None:
            self.skipTest("mpiexec is unavailable")
        topology = CpuTopology.detect()
        layouts = [LaunchLayout(1, 1), LaunchLayout(2, 2)]
        if topology.cores_per_package >= 8:
            layouts.append(LaunchLayout(4, 2))
        with tempfile.TemporaryDirectory() as raw:
            environment = dict(os.environ, TMPDIR=raw)
            for layout in layouts:
                with self.subTest(layout=str(layout)):
                    report = verify_binding(
                        layout,
                        mpiexec=mpiexec,
                        numproc_flag=os.environ.get("SLOPE_MPI_NUMPROC_FLAG", "-n"),
                        topology=topology,
                        environment=environment,
                    )
                    self.assertTrue(report.disjoint)
                    self.assertEqual(len(report.rank_cores), layout.ranks)


if __name__ == "__main__":
    unittest.main()
