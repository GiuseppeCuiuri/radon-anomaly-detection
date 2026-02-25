import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import csv
import io
import contextlib
from RADON_AE_SAD import RADON, fit

# --- regex per parsing log ---
_NUM = r"(?:[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?|nan|NaN)"

_epoch_re = re.compile(
    rf"^Ep\s+(?P<epoch>\d+)\s+\[(?P<phase>[^\]]+)\]\s+\|\s+"
    rf"Loss=(?P<loss_mean>{_NUM})\s+\|\s+"
    rf"Lin=(?P<loss_in_mean>{_NUM})\s+\|\s+"
    rf"Ls\(real\)=(?P<loss_s_real>{_NUM})\s+\|\s+"
    rf"λ=(?P<lambda_s>{_NUM})\s+\|\s+"
    rf"Ratio\(λ\*Ls/Lin\)=(?P<ratio>{_NUM})x\s+\|\s+"
    rf"S_real_sum=(?P<s_real_sum>\d+)\s+\|\s+"
    rf"cos\(Ŝ,S\)=(?P<cos_ss>{_NUM})\s+\|\s+"
    rf"cos\(Ŝ,(?:F\(S\)|1[-−]S)\)=(?P<cos_sneg>{_NUM})\s+\|\s+"
    rf"grad_norm=(?P<grad_norm>{_NUM})\s+\|\s+"
    rf"param_norm=(?P<param_norm>{_NUM})\s+\|\s+"
    rf"time=(?P<time_s>{_NUM})s$"
)


_proj_header_re = re.compile(r"^Projection\s+@\s+epoch\s+(?P<epoch>\d+)\s*$")
_bins_re = re.compile(r"^bins=(?P<bins>\d+)\s*$")
_thresholds_re = re.compile(
    rf"^threshold_L=(?P<threshold_L>{_NUM})\s+\|\s+threshold_S=(?P<threshold_S>{_NUM})\s*$"
)
_sizes_re = re.compile(
    rf"^\|L\|=(?P<L>\d+)\s+\|\s+\|S_raw\|=(?P<S_raw>\d+)\s+\|\s+lambda_eff=(?P<lambda_eff>{_NUM})\s*$"
)
_cos_scores_stats_re = re.compile(
    r"^cos_scores stats:\s+min=(?P<min>[-+eE0-9\.]+)\s+mean=(?P<mean>[-+eE0-9\.]+)\s+max=(?P<max>[-+eE0-9\.]+)\s+\|\s+"
    r"p1=(?P<p1>[-+eE0-9\.]+)\s+p5=(?P<p5>[-+eE0-9\.]+)\s+p10=(?P<p10>[-+eE0-9\.]+)\s+"
    r"p50=(?P<p50>[-+eE0-9\.]+)\s+p90=(?P<p90>[-+eE0-9\.]+)\s+p95=(?P<p95>[-+eE0-9\.]+)\s+p99=(?P<p99>[-+eE0-9\.]+)\s*$"
)
_U_re = re.compile(r"^\|U\|=(?P<U>\d+)\s*$"
)
_cos_L_re = re.compile(
    r"^mean\s+cos\(L_hat,\s*L\)\s*=\s*(?P<mean>[-+eE0-9\.]+)\s*±\s*(?P<std>[-+eE0-9\.]+)\s*$"
)
_cos_S_re = re.compile(
    r"^mean\s+cos\(S_hat,\s*S\)\s*=\s*(?P<mean>[-+eE0-9\.]+)\s*±\s*(?P<std>[-+eE0-9\.]+)\s*$"
)
_cos_Sneg_re = re.compile(
    r"^mean\s+cos\(S_hat,\s*(?:1[-−]S|F\(S\))\)\s*=\s*(?P<mean>[-+eE0-9\.]+|nan|NaN)\s*±\s*(?P<std>[-+eE0-9\.]+|nan|NaN)\s*$"
)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def parse_fit_stdout(log_text: str):
    epochs_rows = []
    proj_rows = []
    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    cur_proj = None

    for ln in lines:
        m = _epoch_re.match(ln)
        if m:
            d = m.groupdict()
            epochs_rows.append({k: (_safe_float(v) if k not in ['epoch', 'phase', 's_real_sum'] else (int(v) if k != 'phase' else v)) for k, v in d.items()})
            continue

        m = _proj_header_re.match(ln)
        if m:
            if cur_proj is not None:
                proj_rows.append(cur_proj)
            cur_proj = {"epoch": int(m.group("epoch"))}
            continue

        if cur_proj is not None:
            if m := _bins_re.match(ln):
                cur_proj["bins"] = int(m.group("bins"))
                continue
            if m := _thresholds_re.match(ln):
                cur_proj.update({k: _safe_float(v) for k, v in m.groupdict().items()})
                continue
            if m := _sizes_re.match(ln):
                d = m.groupdict()
                cur_proj["L"] = int(d["L"])
                cur_proj["S_raw"] = int(d["S_raw"])
                cur_proj["lambda_eff"] = _safe_float(d["lambda_eff"])
                continue

            if m := _U_re.match(ln):
                cur_proj["U"] = int(m.group("U"))
                continue

            if m := _cos_scores_stats_re.match(ln):
                cur_proj.update({f"cos_{k}": _safe_float(v) for k, v in m.groupdict().items()})
                continue
            if m := _cos_L_re.match(ln):
                cur_proj["cos_Lhat_L_mean"] = _safe_float(m.group("mean"))
                cur_proj["cos_Lhat_L_std"] = _safe_float(m.group("std"))
                continue
            if m := _cos_S_re.match(ln):
                cur_proj["cos_Shat_S_mean"] = _safe_float(m.group("mean"))
                cur_proj["cos_Shat_S_std"] = _safe_float(m.group("std"))
                continue
            if m := _cos_Sneg_re.match(ln):
                cur_proj["cos_Shat_1mS_mean"] = _safe_float(m.group("mean"))
                cur_proj["cos_Shat_1mS_std"] = _safe_float(m.group("std"))
                continue

    if cur_proj is not None:
        proj_rows.append(cur_proj)
    return epochs_rows, proj_rows

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            clean_r = {k: r.get(k, None) for k in fieldnames}
            w.writerow(clean_r)

def _normalize_fit_kwargs(fit_kwargs: dict) -> dict:
    """Normalizza kwargs per retrocompatibilita."""
    kw = dict(fit_kwargs)

    if "projection_after_epochs" in kw:
        pae = kw.pop("projection_after_epochs")
        kw.setdefault("projection_after_epochs_wu", pae)
        kw.setdefault("projection_after_epochs_nl", pae)

    if ("projection_after_epochs_wu" in kw) and ("projection_after_epochs_nl" not in kw):
        kw["projection_after_epochs_nl"] = kw["projection_after_epochs_wu"]
    if ("projection_after_epochs_nl" in kw) and ("projection_after_epochs_wu" not in kw):
        kw["projection_after_epochs_wu"] = kw["projection_after_epochs_nl"]

    kw.setdefault("verbose", False)

    return kw

def train_radon_debug(
    model,
    X_train,
    dataset_name: str,
    repetition_index: int,
    debug_dir: str,
    device="cpu",
    input_dim=None,
    dim1=None,
    dim2=None,
    **fit_kwargs
):
    """Fit con cattura stdout per debug e salvataggio CSV."""
    fit_kwargs = _normalize_fit_kwargs(fit_kwargs)

    print(f"\n--- [DEBUG TRAIN] Repetition : {repetition_index} ({dataset_name}) ---")
    start_time = time.time()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        loss, th, th_L, th_S = fit(model, X_train, **fit_kwargs)

    fit_log = buf.getvalue()
    print(fit_log, end="")

    os.makedirs(debug_dir, exist_ok=True)

    # 1) raw log
    rawlog_path = os.path.join(debug_dir, f"radon_{dataset_name}_rep{repetition_index}_rawlog.txt")
    with open(rawlog_path, "w", encoding="utf-8") as f:
        f.write(fit_log)

    # 2) CSV parsing
    epochs_rows, proj_rows = parse_fit_stdout(fit_log)

    # Metadata
    meta = {
        "dataset": dataset_name,
        "repetition": repetition_index,
        "bin_estimator": str(fit_kwargs.get("bin_estimator", "")),
        "epochs": int(fit_kwargs.get("epochs", -1)),
        "projection_after_epochs": int(fit_kwargs.get("projection_after_epochs_nl", -1)),
        "batch_size": int(fit_kwargs.get("batch_size", -1)),
        "lr": float(fit_kwargs.get("lr", float("nan"))),
        "input_dim": input_dim,
        "dim1": dim1,
        "dim2": dim2,
        "device": str(device),
    }
    for r in epochs_rows: r.update(meta)
    for r in proj_rows: r.update(meta)

    # fieldnames
    epoch_fields = [
        "dataset","repetition","bin_estimator","device",
        "input_dim","dim1","dim2",
        "epochs","projection_after_epochs","batch_size","lr",
        "epoch","phase",
        "loss_mean","loss_in_mean","loss_s_real",
        "lambda_s","ratio","s_real_sum",
        "cos_ss","cos_sneg",
        "grad_norm","param_norm",
        "time_s",
    ]
    proj_fields = [
        "dataset","repetition","bin_estimator","device",
        "input_dim","dim1","dim2",
        "epochs","projection_after_epochs","batch_size","lr",
        "epoch",
        "bins","threshold_L","threshold_S","lambda_eff",
        "L","S_raw","U",
        "cos_min","cos_mean","cos_max",
        "p1","p5","p10","p50","p90","p95","p99",
        "cos_Lhat_L_mean","cos_Lhat_L_std",
        "cos_Shat_S_mean","cos_Shat_S_std",
        "cos_Shat_1mS_mean","cos_Shat_1mS_std",
    ]

    epochs_csv = os.path.join(debug_dir, f"radon_{dataset_name}_rep{repetition_index}_epochs.csv")
    proj_csv   = os.path.join(debug_dir, f"radon_{dataset_name}_rep{repetition_index}_projections.csv")

    write_csv(epochs_csv, epochs_rows, epoch_fields)
    write_csv(proj_csv, proj_rows, proj_fields)

    print(f"Debug dump salvato:\n  - {epochs_csv}\n  - {proj_csv}")

    train_time = time.time() - start_time
    return loss, th, th_L, th_S, train_time


def train_radon_simple(
    model,
    X_train,
    repetition_index: int,
    **fit_kwargs
):
    """Fit standard senza cattura log."""
    fit_kwargs = _normalize_fit_kwargs(fit_kwargs)

    print(f"\n--- [SIMPLE TRAIN] Repetition : {repetition_index} ---")
    start_time = time.time()

    loss, th, th_L, th_S = fit(model, X_train, **fit_kwargs)

    train_time = time.time() - start_time
    print(f"Training finito in {train_time:.2f} secondi.")
    return loss, th, th_L, th_S, train_time

def plot_training_losses(loss_data, warmup_epochs=0, dataset_name="", bin_estimator=""):
    """Plotta la loss di training per ogni ripetizione."""
    
    if len(loss_data) > 0 and not isinstance(loss_data[0], (list, np.ndarray, torch.Tensor)):
        loss_data = [loss_data]
    elif len(loss_data) > 0 and isinstance(loss_data[0], torch.Tensor) and loss_data[0].ndim == 0:
        loss_data = [loss_data]

    repetitions = len(loss_data)

    for rep, current_loss in enumerate(loss_data):
        
        # --- normalizzazione formato ---
        if isinstance(current_loss, torch.Tensor):
            current_loss = current_loss.detach().cpu().numpy().tolist()
        elif isinstance(current_loss, np.ndarray):
            current_loss = current_loss.tolist()
        elif isinstance(current_loss, list) and len(current_loss) > 0:
            if isinstance(current_loss[0], torch.Tensor):
                current_loss = [l.detach().cpu().item() for l in current_loss]
            elif isinstance(current_loss[0], (np.floating, float)):
                current_loss = [float(l) for l in current_loss]
        
        # Asse X (Epoche)
        epochs_range = np.arange(1, len(current_loss) + 1)
        
        # plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, current_loss, label=f'Total Loss', linewidth=2, color='#1f77b4')
        
        if warmup_epochs > 0 and warmup_epochs < len(current_loss):
            plt.axvline(x=warmup_epochs, color='red', linestyle='--', alpha=0.8, 
                        label='Start Negative Learning', linewidth=1.5)

        # Titoli
        title_suffix = f"({dataset_name})" if dataset_name else ""
        bin_suffix = f"[{bin_estimator}]" if bin_estimator else ""
        plt.title(f'Training Loss - Repetition {rep+1}/{repetitions} {title_suffix} {bin_suffix}', fontsize=12)
        
        plt.ylabel('Loss (Log Scale)', fontsize=11)
        plt.xlabel('Epoch', fontsize=11)
        
        # Scala log
        plt.yscale('log')
        
        # griglia e legenda
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()