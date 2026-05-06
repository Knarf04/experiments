"""
cp_over_world_bug.py — demonstrate that ``--cp_over_world`` has no effect on
the *model's* CP communication group when ``--sharding_strategy=hsdp`` is
set, even though it DOES change the dataloader's sequence-slicing factor.

Hypothesis under test
---------------------
In ``main_training_mamba.py`` (from the installed ``fms_fsdp`` package),
lines 94-111 do this:

    requires_2d_mesh = (cfg.sharding_strategy == "hsdp") or (
        cfg.cp and not cfg.cp_over_world
    )
    if requires_2d_mesh:
        ...
        cp_mesh = mesh["intra_node"] if cfg.cp else None     # size = gpus/node
    else:
        ...
        cp_mesh = mesh if cfg.cp else None                   # size = world_size

    if cfg.cp:
        cp_degree = world_size if cfg.cp_over_world else torch.cuda.device_count()
    else:
        cp_degree = 1
    dp_degree = world_size // cp_degree

When ``sharding_strategy=="hsdp"``, the first branch is taken regardless of
``cp_over_world``, so ``cp_mesh`` is forced to ``intra_node`` (size =
gpus_per_node). But ``cp_degree`` still flips with ``cp_over_world`` — and
``dp_degree`` is what the dataloader uses to slice each sequence
(``dataloader_utils.py:79-85``). Result: with ``hsdp + cp_over_world``,
the model's CP group covers gpus_per_node ranks while the dataloader
slices each sequence into ``world_size`` pieces. Mismatch.

Method
------
* Spawn WORLD_SIZE gloo workers (CPU only — no GPUs required).
* Replicate the inline mesh logic and the dataloader slicing math, verbatim,
  with line-number references to the repo. The only deliberate diff: pass
  ``"cpu"`` to ``init_device_mesh`` and stub ``torch.cuda.device_count()``
  to a chosen GPUS_PER_NODE — neither affects the size arithmetic the
  bug lives in.
* Run four ``(sharding, cp_over_world)`` combinations and assert that
  exactly the ``("hsdp", True)`` case is inconsistent.

This script does NOT modify the repo. It imports the real ``train_config``
dataclass from the installed ``fms_fsdp`` package and replicates the inline
mesh code; you can diff this file's two ``replicate_*`` functions against
the repo line-by-line.

Assumes ``fms_fsdp`` is importable on the host (i.e. ``pip install -e .``
has been run against the repo, or the repo is otherwise on PYTHONPATH).

Run
---
    python ~/Code/experiments/cp_over_world_bug.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

# Real package import — same dataclass main_training_mamba.py uses.
from fms_fsdp.config.training import train_config

# Simulated cluster shape. Any case where gpus_per_node != world_size suffices.
NODES = 4
GPUS_PER_NODE = 8
WORLD_SIZE = NODES * GPUS_PER_NODE


def replicate_main_mesh_logic(cfg, world_size: int):
    """Verbatim port of main_training_mamba.py:94-111.

    Diffs from the original (none change the size arithmetic):
      * ``init_device_mesh("cuda", ...)`` -> ``init_device_mesh("cpu", ...)``
      * ``torch.cuda.device_count()`` is stubbed by the caller to GPUS_PER_NODE.
    """
    # main_training_mamba.py:94-96
    requires_2d_mesh = (cfg.sharding_strategy == "hsdp") or (
        cfg.cp and not cfg.cp_over_world
    )
    # main_training_mamba.py:97-104
    if requires_2d_mesh:
        num_gpu_per_node = torch.cuda.device_count()
        assert world_size % num_gpu_per_node == 0
        mesh = init_device_mesh(
            "cpu",
            (world_size // num_gpu_per_node, num_gpu_per_node),
            mesh_dim_names=("inter_node", "intra_node"),
        )
        fsdp_mesh = mesh
        cp_mesh = mesh["intra_node"] if cfg.cp else None
    else:
        mesh = init_device_mesh("cpu", (world_size,))
        fsdp_mesh = mesh
        cp_mesh = mesh if cfg.cp else None

    # main_training_mamba.py:106-111
    if cfg.cp:
        cp_degree = world_size if cfg.cp_over_world else torch.cuda.device_count()
    else:
        cp_degree = 1
    dp_degree = world_size // cp_degree
    return fsdp_mesh, cp_mesh, cp_degree, dp_degree


def replicate_loader_slicing(rank: int, world_size: int, dp_degree: int):
    """Verbatim port of dataloader_utils.py:79-85.

    The original then reassigns ``world_size = dp_degree`` and
    ``rank = rank // cp_worldsize``; we don't need those for this demo.
    """
    do_cp = False
    cp_worldsize = 1
    cp_rank = 0
    if dp_degree != world_size:
        do_cp = True
        cp_worldsize = world_size // dp_degree
        cp_rank = rank % cp_worldsize
    return do_cp, cp_worldsize, cp_rank


def worker(rank: int, world_size: int, sharding: str, cow: bool, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Stub torch.cuda.device_count() to our simulated per-node GPU count.
    # On a CPU-only Mac the real value is 0; on GPU hosts it'd be the actual
    # local count. The bug is independent of the hardware — it's pure logic.
    torch.cuda.device_count = lambda: GPUS_PER_NODE  # type: ignore[assignment]

    # Real repo dataclass; only override the three knobs we care about.
    cfg = train_config(sharding_strategy=sharding, cp=True, cp_over_world=cow)

    _, cp_mesh, cp_degree, dp_degree = replicate_main_mesh_logic(cfg, world_size)
    _, cp_worldsize, _ = replicate_loader_slicing(rank, world_size, dp_degree)

    cp_mesh_size = cp_mesh.size() if cp_mesh is not None else 1
    consistent = cp_mesh_size == cp_worldsize

    if rank == 0:
        flag = "OK      " if consistent else "MISMATCH"
        print(
            f"[{flag}] sharding={sharding:<4} cp_over_world={str(cow):<5} "
            f"| cp_mesh.size()={cp_mesh_size}  cp_degree={cp_degree}  "
            f"dp_degree={dp_degree}  loader_cp_worldsize={cp_worldsize}"
        )
        if not consistent:
            print(
                f"          ^^ model's CP all-gather group covers {cp_mesh_size} "
                f"ranks; dataloader hands each rank a 1/{cp_worldsize} slice "
                f"of the sequence."
            )

        # The whole point of this script: the bug fires for exactly one combo.
        if (sharding, cow) == ("hsdp", True):
            assert not consistent, (
                "Expected mismatch for hsdp + cp_over_world, but everything "
                "lined up. Has the repo been patched since this script was "
                "written?"
            )
        else:
            assert consistent, (
                f"Unexpected mismatch for ({sharding}, cp_over_world={cow}). "
                f"cp_mesh.size()={cp_mesh_size} vs loader cp_worldsize={cp_worldsize}."
            )

    dist.destroy_process_group()


def main():
    print(
        f"Simulated cluster: {NODES} nodes x {GPUS_PER_NODE} gpus/node "
        f"= {WORLD_SIZE} ranks (gloo, CPU)\n"
    )
    cases = [
        ("hsdp", True),   # the buggy combination — your invocation
        ("hsdp", False),
        ("fsdp", True),
        ("fsdp", False),
    ]
    base_port = 29500
    for i, (sharding, cow) in enumerate(cases):
        mp.spawn(
            worker,
            args=(WORLD_SIZE, sharding, cow, base_port + i),
            nprocs=WORLD_SIZE,
            join=True,
        )
    print(
        "\nAll asserts passed: the (hsdp, cp_over_world=True) combination "
        "is the only inconsistent one."
    )


if __name__ == "__main__":
    main()
