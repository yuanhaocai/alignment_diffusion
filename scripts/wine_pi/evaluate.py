"""Evaluate a frozen variant without requiring private historical sample banks."""
import argparse
import json
import numpy as np
from observed_residual_pi import evaluate, load_bank
from .common import ROOT, PROTOCOL, read_y, sha, write_json, check_test, frozen_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["full", "tabular_only"], required=True)
    variant = parser.parse_args().variant
    frozen_protocol()
    frozen = json.loads((ROOT / "frozen_models.json").read_text())[variant]
    assert sha(ROOT / variant / "diffusion/selected_model.pt") == frozen["model_sha256"]
    banks = {}
    for split in ["cal", "test"]:
        directory = ROOT / variant / f"sampling_{split}_m{PROTOCOL['M']}"
        meta = json.loads((directory / "semantic_split.json").read_text())
        assert meta["actual_split"] == split and meta["M"] == PROTOCOL["M"]
        assert meta["model_sha256"] == frozen["model_sha256"]
        assert not meta["outcomes_loaded_for_sampling"]
        banks[split] = load_bank(directory / "y_test_all.npy")
    ycal, ytest = read_y("cal"), read_y("test")
    kwargs = dict(m=PROTOCOL["M"], alpha=PROTOCOL["alpha"], metric_scale=PROTOCOL["paper_response_sd"])
    revised, observations = evaluate(ycal, banks["cal"], ytest, banks["test"], **kwargs)
    linear, _ = evaluate(ycal, banks["cal"], ytest, banks["test"], endpoint_rule="linear", **kwargs)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    summary_path = out / "summary.json"
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {"protocol": PROTOCOL, "models": {}}
    result["test_immutability"] = check_test()
    result["models"][variant] = dict(revised=revised, new_model_linear_endpoints=linear, selected=frozen,
        calibration_bank_sha256=sha(ROOT / variant / f"sampling_cal_m{PROTOCOL['M']}/y_test_all.npy"),
        test_bank_sha256=sha(ROOT / variant / f"sampling_test_m{PROTOCOL['M']}/y_test_all.npy"))
    np.savez_compressed(out / f"{variant}_per_observation.npz", **observations)
    write_json(summary_path, result)
    write_json(out / "test_immutability_final.json", result["test_immutability"])
    rows = []
    for key, label in [("tabular_only", "tabular only"), ("full", "tabular + text")]:
        if key in result["models"]:
            v = result["models"][key]["revised"]
            rows.append(f"Ours ({label}) & {v['coverage']:.3f} & {v['width']:.3f} & {v['crps']:.3f} " + r"\\")
    (out / "table3_new_rows.tex").write_text("\n".join(rows) + "\n")
    print(json.dumps(revised, indent=2))


if __name__ == "__main__":
    main()
