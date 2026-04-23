"""
scripts/train_parallel.py
--------------------------
Trains multiple backbone models simultaneously across all available CPU cores.
Uses pure subprocess.Popen — no ProcessPoolExecutor, no multiprocessing module,
no shared memory. Fully compatible with Windows 10/11.

Usage:
    python scripts/train_parallel.py --data data/processed/v1.0/dataset.yaml
    python scripts/train_parallel.py --backbones resnet18 efficientnet_b0 mobilenet_v3_small
    python scripts/train_parallel.py --max-workers 4
    python scripts/train_parallel.py --epochs 5 --backbones resnet18 efficientnet_b0
    python scripts/train_parallel.py --epochs 50 --track-energy
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


ALL_BACKBONES = [
    ("ResNet",        "resnet18"),
    ("ResNet",        "resnet34"),
    ("ResNet",        "resnet50"),
    ("ResNet",        "resnet101"),
    ("EfficientNet",  "efficientnet_b0"),
    ("EfficientNet",  "efficientnet_b2"),
    ("MobileNet",     "mobilenet_v3_small"),
    ("MobileNet",     "mobilenet_v3_large"),
    ("SqueezeNet",    "squeezenet1_1"),
    ("ShuffleNet",    "shufflenet_v2"),
    ("RegNet",        "regnet_y_400mf"),
    ("DenseNet",      "densenet121"),
    ("ConvNeXt",      "convnext_tiny"),
    ("ViT",           "vit_b_16"),
]

BACKBONE_NAMES = [name for _, name in ALL_BACKBONES]
FAMILY_MAP     = {name: family for family, name in ALL_BACKBONES}


@dataclass
class Job:
    backbone:     str
    family:       str
    status:       str = "queued"
    proc:         object = None
    start_time:   Optional[float] = None
    end_time:     Optional[float] = None
    best_iou:     Optional[float] = None
    total_co2_g:  Optional[float] = None
    total_kwh:    Optional[float] = None
    returncode:   int = -1
    resume_epoch: int = 0       # 0 = fresh start, N = resuming from epoch N
    is_complete:  bool = False  # True if already finished from a prior run

    @property
    def elapsed_str(self):
        if self.start_time is None:
            return "-"
        end = self.end_time or time.time()
        return str(timedelta(seconds=int(end - self.start_time)))


def read_run_log(backbone):
    path = Path("models") / "checkpoints" / backbone / "run_log.json"
    if not path.exists():
        return None, None, None
    try:
        with open(path, encoding="utf-8") as f:
            run_log = json.load(f)
        if not run_log:
            return None, None, None
        last     = run_log[-1]
        best_iou = round(max(e["val_iou"] for e in run_log), 4)
        co2_g    = round(last.get("cumulative_co2_kg", 0) * 1000, 2)
        kwh      = round(last.get("cumulative_energy_kwh", 0), 4)
        return best_iou, co2_g, kwh
    except Exception:
        return None, None, None


def check_resume_state(backbone, total_epochs):
    """
    Returns (resume_epoch, is_complete) by inspecting resume.pt and run_log.json.
    resume.pt present  = partial run, crashed mid-training
    resume.pt absent + run_log present with full epochs = complete
    neither present    = never started
    """
    ckpt_dir    = Path("models") / "checkpoints" / backbone
    resume_path = ckpt_dir / "resume.pt"
    log_path    = ckpt_dir / "run_log.json"

    if not resume_path.exists() and log_path.exists():
        # run_log exists but resume.pt was deleted = completed cleanly
        try:
            with open(log_path, encoding="utf-8") as f:
                run_log = json.load(f)
            if run_log and run_log[-1]["epoch"] >= total_epochs:
                return total_epochs, True
        except Exception:
            pass

    if resume_path.exists():
        try:
            import torch as _torch
            ckpt = _torch.load(resume_path, map_location="cpu")
            return ckpt.get("epoch", 0), False
        except Exception:
            pass

    return 0, False


def render_table(jobs, max_workers, start_time, done=False):
    os.system("cls" if os.name == "nt" else "clear")
    elapsed   = timedelta(seconds=int(time.time() - start_time))
    n_done    = sum(1 for j in jobs if j.status == "done")
    n_failed  = sum(1 for j in jobs if j.status == "failed")
    n_running = sum(1 for j in jobs if j.status == "running")
    n_queued  = sum(1 for j in jobs if j.status == "queued")

    print("=" * 78)
    print(f"  OpenLPR Parallel Training  |  {max_workers} workers  |  elapsed {elapsed}")
    print(f"  done:{n_done}  running:{n_running}  queued:{n_queued}  failed:{n_failed}")
    print("=" * 78)
    print(f"  {'Backbone':<22} {'Family':<12} {'Status':<10} {'Time':>8}  {'IoU':>7}  {'CO2':>7}  {'kWh':>7}")
    print("-" * 78)

    order = {"running": 0, "done": 1, "queued": 2, "failed": 3, "skipped": 4}
    for j in sorted(jobs, key=lambda x: (order[x.status], x.backbone)):
        resume_tag = (f" (ep{j.resume_epoch}+)" if j.resume_epoch > 0
                      and j.status in ("queued", "running") else "")
        label = {"running": ">> running", "done": "OK done",
                 "queued":  ".. queued",  "failed": "XX failed",
                 "skipped": "-- skipped"}.get(j.status, j.status) + resume_tag
        iou = f"{j.best_iou:.4f}"     if j.best_iou    is not None else "-"
        co2 = f"{j.total_co2_g:.1f}g" if j.total_co2_g is not None else "-"
        kwh = f"{j.total_kwh:.4f}"    if j.total_kwh   is not None else "-"
        print(f"  {j.backbone:<22} {j.family:<12} {label:<10} {j.elapsed_str:>8}  {iou:>7}  {co2:>7}  {kwh:>7}")

    print("=" * 78)
    if not done:
        print("  Refreshing every 5s  |  monitor a job:")
        print("  Get-Content models\\checkpoints\\<backbone>\\train.log -Wait")
    else:
        completed = [j for j in jobs if j.status == "done"]
        if completed:
            best    = max(completed, key=lambda j: j.best_iou or 0)
            fastest = min(completed, key=lambda j: (j.end_time or 0) - (j.start_time or 0))
            print(f"\n  Best IoU    : {best.backbone} ({best.best_iou:.4f})")
            print(f"  Fastest     : {fastest.backbone} ({fastest.elapsed_str})")
            print(f"  Total CO2   : {sum(j.total_co2_g or 0 for j in completed):.1f} g")
            print(f"  Total energy: {sum(j.total_kwh or 0 for j in completed):.4f} kWh")
        print("\n  Next: python scripts/visualise_results.py --show")
    print()


def run_all(jobs, max_workers, args_dict):
    queue   = list(jobs)
    running = []

    python   = sys.executable
    base_cmd = [
        python, "scripts/train.py",
        "--data",    args_dict["data"],
        "--epochs",  str(args_dict["epochs"]),
        "--batch",   str(args_dict["batch"]),
        "--lr",      str(args_dict["lr"]),
        "--version", str(args_dict["version"]),
        "--seed",    str(args_dict["seed"]),
        "--resume",  "auto",   # always pass auto — train.py decides
    ]
    if args_dict.get("track_energy"):
        base_cmd.append("--track-energy")

    # Pre-check all jobs for existing partial or complete runs
    for job in queue:
        ep, complete = check_resume_state(job.backbone, args_dict["epochs"])
        job.resume_epoch = ep
        job.is_complete  = complete
        if complete:
            job.status = "skipped"
            job.best_iou, job.total_co2_g, job.total_kwh =                 read_run_log(job.backbone)

    # Remove already-complete jobs from the queue
    skipped = [j for j in queue if j.is_complete]
    queue   = [j for j in queue if not j.is_complete]
    if skipped:
        print(f"  [resume] Skipping {len(skipped)} already-complete backbone(s): "
              + ", ".join(j.backbone for j in skipped))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"

    while queue or running:
        # Launch new jobs up to max_workers
        while queue and len(running) < max_workers:
            job      = queue.pop(0)
            log_dir  = Path("models") / "checkpoints" / job.backbone
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "train.log"
            cmd      = base_cmd + ["--backbone", job.backbone]

            try:
                log_file       = open(log_path, "w", encoding="utf-8")
                proc           = subprocess.Popen(cmd, stdout=log_file,
                                                  stderr=subprocess.STDOUT,
                                                  cwd=str(Path.cwd()), env=env)
                job.proc       = proc
                job._log_file  = log_file
                job.status     = "running"
                job.start_time = time.time()
                running.append(job)
            except Exception as e:
                job.status   = "failed"
                job.end_time = time.time()

        # Poll for completions
        still_running = []
        for job in running:
            rc = job.proc.poll()
            if rc is None:
                still_running.append(job)
            else:
                job._log_file.close()
                job.returncode = rc
                job.end_time   = time.time()
                if rc == 0:
                    job.status = "done"
                    job.best_iou, job.total_co2_g, job.total_kwh = \
                        read_run_log(job.backbone)
                else:
                    job.status = "failed"
        running = still_running
        time.sleep(1)


def main():
    args        = parse_args()
    backbones   = args.backbones if args.backbones else BACKBONE_NAMES
    cpu_count   = os.cpu_count() or 2
    max_workers = min(args.max_workers, len(backbones)) if args.max_workers \
                  else max(1, cpu_count // 2)

    jobs = [Job(backbone=bb, family=FAMILY_MAP.get(bb, "?")) for bb in backbones]

    args_dict = {
        "data": args.data, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "version": args.version, "seed": args.seed,
        "track_energy": args.track_energy,
    }

    print(f"\nOpenLPR Parallel Training")
    print(f"  Backbones : {len(backbones)}  |  Workers : {max_workers}  |  Epochs : {args.epochs}")
    print(f"  Data      : {args.data}")
    print(f"  Logs      : models/checkpoints/<backbone>/train.log")
    print(f"\nStarting in 3 seconds...")
    time.sleep(3)

    start_time   = time.time()
    stop_display = threading.Event()

    def display_loop():
        while not stop_display.is_set():
            render_table(jobs, max_workers, start_time)
            time.sleep(5)

    threading.Thread(target=display_loop, daemon=True).start()

    try:
        run_all(jobs, max_workers, args_dict)
    except KeyboardInterrupt:
        print("\nStopping...")
        for j in jobs:
            if j.status == "running" and j.proc:
                j.proc.terminate()
        stop_display.set()
        return

    stop_display.set()
    time.sleep(0.3)
    render_table(jobs, max_workers, start_time, done=True)

    summary_path = Path("evaluation") / "parallel_run_summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "backbones": backbones, "max_workers": max_workers,
            "total_elapsed_s": round(time.time() - start_time, 1),
            "results": {
                j.backbone: {
                    "status": j.status,
                    "elapsed_s": round((j.end_time or 0) - (j.start_time or 0), 1),
                    "best_iou": j.best_iou,
                    "total_co2_g": j.total_co2_g,
                    "total_kwh": j.total_kwh,
                } for j in jobs
            },
        }, f, indent=2)
    print(f"  Summary: {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train multiple LPR backbones in parallel")
    p.add_argument("--backbones", nargs="+", choices=BACKBONE_NAMES, default=None)
    p.add_argument("--data",         default="data/processed/v1.0/dataset.yaml")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch",        type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--version",      default="v1.0")
    p.add_argument("--max-workers",  type=int,   default=None)
    p.add_argument("--track-energy", action="store_true")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
