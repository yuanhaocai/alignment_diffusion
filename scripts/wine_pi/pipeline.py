"""Sequential, resumable stages; no calibration/test-driven selection."""
import json
import os
import subprocess
import time
from .common import ROOT, REPO, PYTHON, frozen_protocol, sha, write_json, check_test, log


def verify_sources():
    path = ROOT / "upstream_sources.json"
    if path.exists():
        for source, meta in json.loads(path.read_text()).items():
            if sha(source) != meta["sha256"]:
                raise ValueError(f"Source changed since preparation: {source}; use a fresh run directory")


def stage(name, module, *args, gpu=None, variant=None):
    verify_sources()
    status = ROOT / "status" / f"{name}.json"
    if status.exists() and json.loads(status.read_text())["state"] == "completed":
        log(f"Skipping completed stage {name}")
        return
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "scripts")]),
               CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
    if variant:
        env["WINE_VARIANT"] = variant
    cmd = [PYTHON, "-u", "-m", f"wine_pi.{module}", *args]
    meta = dict(state="running", command=cmd, gpu=gpu, started=time.strftime("%Y-%m-%d %H:%M:%S %z"))
    write_json(status, meta)
    log(f"Starting {name}; log: {ROOT / 'logs' / (name + '.log')}")
    with (ROOT / "logs" / f"{name}.log").open("a") as output:
        completed = subprocess.run(cmd, env=env, cwd=REPO, stdout=output, stderr=subprocess.STDOUT)
    meta.update(state="completed" if completed.returncode == 0 else "failed",
                returncode=completed.returncode, finished=time.strftime("%Y-%m-%d %H:%M:%S %z"))
    write_json(status, meta)
    if completed.returncode:
        raise RuntimeError(f"{name} failed; see {ROOT / 'logs' / (name + '.log')}")


def freeze(variant):
    selection = json.loads((ROOT / variant / "selection.json").read_text())
    digest = sha(ROOT / variant / "diffusion/selected_model.pt")
    if selection["selected_model_sha256"] != digest:
        raise ValueError("selected checkpoint changed")
    path = ROOT / "frozen_models.json"
    frozen = json.loads(path.read_text()) if path.exists() else {}
    if variant in frozen and frozen[variant]["model_sha256"] != digest:
        raise ValueError("a previously frozen model changed")
    frozen[variant] = dict(selection=selection, model_sha256=digest)
    write_json(path, frozen)


def run(phase, variants, gpu):
    for directory in ["status", "logs", "shared", "full", "tabular_only"]:
        (ROOT / directory).mkdir(exist_ok=True)
    frozen_protocol()
    verify_sources()
    write_json(ROOT / "status/master.json", dict(state="running", phase=phase, variants=variants))
    try:
        if phase in ["all", "prepare"]:
            stage("prepare_data", "prepare", "data")
            stage("preprocessing", "prepare", "preprocess")
        if phase in ["all", "train"]:
            check_test()
            stage("selfcheck", "train", "selfcheck", gpu=gpu)
            stage("vae", "train", "vae", gpu=gpu)
            stage("encode_fit", "train", "encode_fit", gpu=gpu)
            for variant in variants:
                if variant == "full":
                    stage("text_fit", "prepare", "text_fit")
                    stage("alignment", "train", "align", gpu=gpu)
                    stage("project_fit", "train", "project_fit", gpu=gpu)
                stage(f"diffusion_{variant}", "train", "diffusion", "--variant", variant, gpu=gpu)
                stage(f"select_{variant}", "train", "select", "--variant", variant, gpu=gpu)
                freeze(variant)
        if phase in ["all", "sample"]:
            for variant in variants:
                freeze(variant)
                stage(f"heldout_preprocess_{variant}", "prepare", "heldout", variant=variant)
                stage("encode_heldout", "train", "encode_heldout", gpu=gpu)
                if variant == "full":
                    stage("text_heldout", "prepare", "text_heldout")
                    stage("project_heldout", "train", "project_heldout", gpu=gpu)
                for split in ["cal", "test"]:
                    stage(f"sample_{variant}_{split}", "train", "sample", "--variant", variant,
                          "--split", split, gpu=gpu)
        if phase in ["all", "evaluate"]:
            for variant in variants:
                freeze(variant)
                stage(f"evaluate_{variant}", "evaluate", "--variant", variant)
        check_test()
        write_json(ROOT / "status/master.json", dict(state="completed", phase=phase, variants=variants))
    except BaseException as error:
        write_json(ROOT / "status/master.json", dict(state="failed", phase=phase, error=str(error)))
        raise
