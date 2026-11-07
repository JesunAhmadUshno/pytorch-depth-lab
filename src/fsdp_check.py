"""A9: FSDP2 shape and initialisation check (NOT a training arm).

Initialises a single-rank process group, applies fully_shard to
the ResNet-18 layer blocks and root, records the sharded parameter
shapes (DTensor local vs global), and runs one forward and
backward step at batch 32. The claim stops at: FSDP2 initialises
and the shapes check out on this machine. No scaling claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DTensor

from .common import env_info, seed_all, write_json
from .data import make_datasets, make_loader
from .model import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    payload: dict = {"env": env_info(device), "claim": "shape and initialisation only"}
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        backend = "gloo"  # single rank; gloo works for CUDA and CPU tensors here
        dist.init_process_group(backend, rank=0, world_size=1)
        mesh = init_device_mesh(device.type, (1,), mesh_dim_names=("dp",))
        seed_all(42)
        model = build_model(False, 42)
        shapes = {}
        for name in ("layer1", "layer2", "layer3", "layer4"):
            block = getattr(model, name)
            fully_shard(block, mesh=mesh)
        fully_shard(model, mesh=mesh)
        model = model.to(device)
        for name, p in model.named_parameters():
            if isinstance(p.data, DTensor):
                shapes[name] = {
                    "global": list(p.data.shape),
                    "local": list(p.data.to_local().shape),
                    "dtensor": True,
                }
            else:
                shapes[name] = {"global": list(p.shape), "dtensor": False}
        payload["n_params"] = len(shapes)
        payload["n_dtensor"] = sum(1 for v in shapes.values() if v["dtensor"])
        payload["shapes_sample"] = dict(list(shapes.items())[:6])
        train_ds, _, _ = make_datasets()
        loader = make_loader(train_ds, 32, shuffle=False, workers=0)
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        payload["step"] = {"loss": float(loss.item()), "output_shape": list(out.shape)}
        payload["status"] = "ok"
        dist.destroy_process_group()
    except Exception as exc:  # a failure here is a recorded result
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
    write_json(Path("results/benchmarks/fsdp_check.json"), payload)
    print(json.dumps(payload, indent=2)[:1200])


if __name__ == "__main__":
    main()
