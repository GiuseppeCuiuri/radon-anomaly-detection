# RADON AE-SAD: autoencoder robusti per anomaly detection
# versione estesa con negative learning e analisi feature tabulari

import math
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class RADON(nn.Module):

    def __init__(self, input_dim, dim1, dim2):
        super(RADON, self).__init__()

        self.input_dim = input_dim
        self.dim1 = dim1
        self.dim2 = dim2

        # encoder
        self.fc1 = nn.Linear(input_dim, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        # decoder
        self.fc3 = nn.Linear(dim2, dim1)
        self.fc4 = nn.Linear(dim1, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


# --- utilita ---

def _grad_norm_l2(m: nn.Module) -> float:
    """Norma L2 dei gradienti."""
    total = 0.0
    for p in m.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            return float("nan")
        total += g.pow(2).sum().item()
    return math.sqrt(total)


def _param_norm_l2(m: nn.Module) -> float:
    """Norma L2 dei parametri."""
    total = 0.0
    for p in m.parameters():
        v = p.detach()
        if not torch.isfinite(v).all():
            return float("nan")
        total += v.pow(2).sum().item()
    return math.sqrt(total)


def _describe_scores(scores_np: np.ndarray) -> OrderedDict:
    """Statistiche descrittive array di score."""
    if scores_np.size == 0:
        return {"empty": True}
    q = np.quantile(scores_np, [0.0, 0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 1.0])
    return OrderedDict([
        ("min", float(scores_np.min())),
        ("mean", float(scores_np.mean())),
        ("max", float(scores_np.max())),
        ("p1", float(q[1])),
        ("p5", float(q[2])),
        ("p10", float(q[3])),
        ("p50", float(q[4])),
        ("p90", float(q[5])),
        ("p95", float(q[6])),
        ("p99", float(q[7])),
    ])


def _fmt_mean_std(x: torch.Tensor) -> Tuple[float, float]:
    """Media e std di un tensore."""
    if x.numel() == 0:
        return float("nan"), float("nan")
    return x.mean().item(), x.std(unbiased=False).item()


# --- statistiche temporali ---

def _compute_timing_stats(
    epoch_times: List[float],
    epoch_phases: List[str],
    epoch_has_projection: List[bool]
) -> Dict[str, Any]:
    """Statistiche tempi per epoca."""
    if not epoch_times:
        return {}
    
    times_arr = np.array(epoch_times)
    phases_arr = np.array(epoch_phases)
    proj_arr = np.array(epoch_has_projection)
    
    # statistiche globali
    stats = {
        "total_epochs": len(epoch_times),
        "total_time_s": float(np.sum(times_arr)),
        "mean_time_s": float(np.mean(times_arr)),
        "std_time_s": float(np.std(times_arr)),
        "p50_time_s": float(np.percentile(times_arr, 50)),
        "p90_time_s": float(np.percentile(times_arr, 90)),
    }
    
    # statistiche per fase
    warmup_mask = (phases_arr == "WARMUP")
    nl_mask = (phases_arr == "NEG_LRN")
    
    if warmup_mask.any():
        wu_times = times_arr[warmup_mask]
        stats["warmup_mean_s"] = float(np.mean(wu_times))
        stats["warmup_count"] = int(warmup_mask.sum())
    else:
        stats["warmup_mean_s"] = 0.0
        stats["warmup_count"] = 0
    
    if nl_mask.any():
        nl_times = times_arr[nl_mask]
        stats["nl_mean_s"] = float(np.mean(nl_times))
        stats["nl_count"] = int(nl_mask.sum())
    else:
        stats["nl_mean_s"] = 0.0
        stats["nl_count"] = 0
    
    # statistiche per proiezione
    with_proj_mask = proj_arr
    without_proj_mask = ~proj_arr
    
    if with_proj_mask.any():
        stats["with_proj_mean_s"] = float(np.mean(times_arr[with_proj_mask]))
        stats["with_proj_count"] = int(with_proj_mask.sum())
    else:
        stats["with_proj_mean_s"] = 0.0
        stats["with_proj_count"] = 0
    
    if without_proj_mask.any():
        stats["without_proj_mean_s"] = float(np.mean(times_arr[without_proj_mask]))
        stats["without_proj_count"] = int(without_proj_mask.sum())
    else:
        stats["without_proj_mean_s"] = 0.0
        stats["without_proj_count"] = 0
    
    # overhead stimato proiezione
    if stats["with_proj_count"] > 0 and stats["without_proj_count"] > 0:
        stats["proj_overhead_est_s"] = stats["with_proj_mean_s"] - stats["without_proj_mean_s"]
    else:
        stats["proj_overhead_est_s"] = float("nan")
    
    return stats


def _print_timing_summary(stats: Dict[str, Any], verbose: bool = True) -> None:
    """Stampa summary tempi."""
    if not verbose or not stats:
        return
    
    print("\n" + "=" * 72)
    print("RIEPILOGO TIMING")
    print("=" * 72)
    print(f"Epoche totali: {stats.get('total_epochs', 0)}")
    print(f"Tempo totale: {stats.get('total_time_s', 0):.2f}s")
    print(f"Tempo medio per epoca: {stats.get('mean_time_s', 0):.4f}s ± {stats.get('std_time_s', 0):.4f}s")
    print(f"Mediana (p50): {stats.get('p50_time_s', 0):.4f}s | p90: {stats.get('p90_time_s', 0):.4f}s")
    print("-" * 72)
    print(f"WARMUP: {stats.get('warmup_count', 0)} epochs, mean={stats.get('warmup_mean_s', 0):.4f}s")
    print(f"NEG_LRN: {stats.get('nl_count', 0)} epochs, mean={stats.get('nl_mean_s', 0):.4f}s")
    print("-" * 72)
    print(f"Con proiezione: {stats.get('with_proj_count', 0)} epoche, media={stats.get('with_proj_mean_s', 0):.4f}s")
    print(f"Senza proiezione: {stats.get('without_proj_count', 0)} epoche, media={stats.get('without_proj_mean_s', 0):.4f}s")
    proj_overhead = stats.get('proj_overhead_est_s', float('nan'))
    if np.isfinite(proj_overhead):
        print(f"Overhead stimato proiezione: {proj_overhead:.4f}s per epoca")
    print("=" * 72 + "\n")


# --- analizzatore feature tabulari ---

class TabularFeatureAnalyzer:
    """Identifica tipo di ogni colonna: binaria, numerica, one-hot."""

    def __init__(self, device: torch.device):
        self.device = device
        self.binary_mask: Optional[torch.Tensor] = None
        self.num_mask: Optional[torch.Tensor] = None
        self.onehot_groups: List[List[int]] = []
        self.onehot_mask: Optional[torch.Tensor] = None

    def analyze(
        self,
        X: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        max_samples: int = 5000,
        verbose: bool = False
    ) -> None:
        """Analizza le feature del dataset X."""
        with torch.no_grad():
            N_mask = min(X.shape[0], max_samples)
            Xsub = X[:N_mask].detach()

            # colonne binarie (solo 0 o 1)
            is0 = (Xsub - 0.0).abs().le(1e-6)
            is1 = (Xsub - 1.0).abs().le(1e-6)
            self.binary_mask = (is0 | is1).all(dim=0)
            self.num_mask = ~self.binary_mask

            if verbose:
                print(f"[Mask] binary cols: {int(self.binary_mask.sum().item())}/{X.shape[1]}")
                print(f"[Mask] numeric cols: {int(self.num_mask.sum().item())}/{X.shape[1]}")

            # gruppi one-hot
            if feature_names is not None:
                self.onehot_groups = self._onehot_groups_from_feature_names(feature_names)
                self.onehot_mask = torch.zeros((X.shape[1],), dtype=torch.bool, device=self.device)
                for g in self.onehot_groups:
                    self.onehot_mask[g] = True
            else:
                self.onehot_groups, self.onehot_mask = self._infer_onehot_groups_from_binary(
                    Xsub, self.binary_mask
                )

            if verbose:
                via = 'names' if feature_names is not None else 'data'
                print(f"[OneHot] via={via} | groups={len(self.onehot_groups)} | "
                      f"onehot_cols={int(self.onehot_mask.sum().item())}")

    def _onehot_groups_from_feature_names(self, names: List[str]) -> List[List[int]]:
        """Gruppi one-hot da prefissi nomi feature (protocol_type_, service_, flag_)."""
        groups = defaultdict(list)
        for i, n in enumerate(names):
            if n.startswith("protocol_type_"):
                groups["protocol_type"].append(i)
            elif n.startswith("service_"):
                groups["service"].append(i)
            elif n.startswith("flag_"):
                groups["flag"].append(i)
        out = [idxs for idxs in groups.values() if len(idxs) >= 2]
        return [sorted(idxs) for idxs in out]

    def _infer_onehot_groups_from_binary(
        self,
        Xsub: torch.Tensor,
        binary_mask: torch.Tensor,
        cooc_eps: float = 1e-4,
        min_pos_rate: float = 1e-4,
        min_group_size: int = 3,
        min_onehot_frac: float = 0.90,
        sum_tol: float = 1e-3
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Inferisce gruppi one-hot dalla co-occorrenza delle colonne binarie."""
        D = Xsub.shape[1]
        bin_idx = torch.where(binary_mask)[0]
        if bin_idx.numel() == 0:
            return [], torch.zeros((D,), dtype=torch.bool, device=Xsub.device)

        Xb = Xsub[:, bin_idx]
        N, B = Xb.shape

        # filtra colonne con tasso positivi troppo basso
        pos_rate = Xb.mean(dim=0)
        keep = pos_rate >= min_pos_rate
        bin_idx = bin_idx[keep]
        Xb = Xb[:, keep]
        N, B = Xb.shape
        if B == 0:
            return [], torch.zeros((D,), dtype=torch.bool, device=Xsub.device)

        # matrice co-occorrenza
        Xbf = Xb.float()
        cooc = (Xbf.T @ Xbf) / max(float(N), 1.0)
        cooc.fill_diagonal_(0.0)

        # adiacenza: colonne con bassa co-occorrenza
        adj = (cooc <= cooc_eps)

        # componenti connesse (gruppi one-hot candidati)
        visited = torch.zeros((B,), dtype=torch.bool, device=Xsub.device)
        groups_local = []

        for i in range(B):
            if visited[i]:
                continue
            stack = [int(i)]
            comp = []
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                neigh = torch.where(adj[u])[0]
                for v in neigh.tolist():
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)

            if len(comp) < min_group_size:
                continue

            # verifica one-hot
            cols = torch.tensor(comp, device=Xsub.device, dtype=torch.long)
            Xg = Xb[:, cols]
            sums = Xg.sum(dim=1)

            ok0 = (sums - 0.0).abs() <= sum_tol
            ok1 = (sums - 1.0).abs() <= sum_tol
            frac_ok = (ok0 | ok1).float().mean().item()
            frac_ones = ok1.float().mean().item()

            if (frac_ok >= min_onehot_frac) and (frac_ones >= 0.05):
                groups_local.append(cols)

        # indici locali -> globali
        groups = []
        in_group_mask = torch.zeros((D,), dtype=torch.bool, device=Xsub.device)
        for cols in groups_local:
            g = bin_idx[cols].tolist()
            g = sorted(set(int(x) for x in g))
            for j in g:
                in_group_mask[j] = True
            groups.append(g)

        return groups, in_group_mask


# --- stato proiezione ---

@dataclass
class ProjectionState:
    """Stato proiezione: soglie, EMA, pesatura dinamica e early stopping."""
    # soglie
    threshold_L: float = 0.0
    threshold_S: float = 0.0
    threshold_S_t: Optional[torch.Tensor] = None

    # EMA stabilizzazione
    thS_ema: Optional[float] = None
    ema_S: Optional[float] = None

    # pesatura dinamica
    lambda_eff: float = 0.0
    lambda_card_ema: Optional[float] = None
    lambda_eff_ema: Optional[float] = None

    # stato dati
    have_S: bool = False
    L: Optional[torch.Tensor] = None
    S: Optional[torch.Tensor] = None
    S_mask: Optional[torch.Tensor] = None

    # S persistente
    S_raw: Optional[torch.Tensor] = None
    S_count: int = 0

    # early stopping
    best_metric: float = float("inf")
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
    bad_projections: int = 0
    nl_baseline_set: bool = False
    best_proj_snapshot: Optional[Dict[str, Any]] = None


# --- helper training ---

def _pad_S_to_L(
    S_raw: Optional[torch.Tensor],
    num_L: int,
    D: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Padding S a dimensione |L| con maschera."""
    if (S_raw is None) or (S_raw.shape[0] == 0):
        S = torch.zeros((num_L, D), device=device)
        S_mask = torch.zeros((num_L,), device=device)
    else:
        num_S = S_raw.shape[0]
        if num_S >= num_L:
            S = S_raw[:num_L]
            S_mask = torch.ones((num_L,), device=device)
        else:
            pad_n = num_L - num_S
            S = torch.cat([S_raw, torch.zeros((pad_n, D), device=device)], dim=0)
            S_mask = torch.cat([
                torch.ones((num_S,), device=device),
                torch.zeros((pad_n,), device=device)
            ], dim=0)
    return S, S_mask


def _create_feature_inverter(num_mask: torch.Tensor):
    """Crea inversione feature per negative learning: inverte numeriche, preserva binarie."""
    def F_feature_aware(t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        out[:, num_mask] = 1.0 - t[:, num_mask]
        return out
    return F_feature_aware


def _train_warmup_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CosineSimilarity,
    device: torch.device
) -> Dict[str, Any]:
    """Epoca warmup: ricostruzione standard su L e S."""
    model.train()
    n_seen = 0
    sum_loss_tot = 0.0
    sum_loss_in = 0.0

    sum_s_num = 0.0
    sum_s_den = 0.0

    sum_cos_Shat_S = 0.0
    sum_cos_Shat_1mS = 0.0
    n_s_real_total = 0.0

    last_grad = float("nan")
    nonfinite_flag = False


    for l, s in train_loader:
        optimizer.zero_grad()
        bs = l.size(0)

        l_hat = model(l)
        s_hat = model(s)

        loss_in_vec = ((1 - criterion(l_hat, l)) ** 2)         
        loss_s_vec  = (criterion(s_hat, s) ** 2)               
        loss_in = loss_in_vec.mean()
        loss_s  = loss_s_vec.mean()
        loss = loss_in + loss_s

        if not torch.isfinite(loss).item():
            nonfinite_flag = True
            continue

        loss.backward()
        last_grad = _grad_norm_l2(model)
        optimizer.step()

        n_seen += bs
        sum_loss_tot += loss.item() * bs
        sum_loss_in += loss_in.item() * bs

        sum_s_num += loss_s_vec.sum().item()
        sum_s_den += float(bs)

        sum_cos_Shat_S += criterion(s_hat, s).sum().item()
        sum_cos_Shat_1mS += 0.0
        n_s_real_total += float(bs)

    return {
        "n_seen": n_seen,
        "sum_loss_tot": sum_loss_tot,
        "sum_loss_in": sum_loss_in,
        "sum_s_num": sum_s_num,
        "sum_s_den": sum_s_den,
        "sum_cos_Shat_S": sum_cos_Shat_S,
        "sum_cos_Shat_1mS": sum_cos_Shat_1mS,
        "n_s_real_total": n_s_real_total,
        "last_grad": last_grad,
        "nonfinite_flag": nonfinite_flag,
        "lambda_used": 0.0,
    }


def _train_negative_learning_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CosineSimilarity,
    F_feature_aware,
    proj_state: ProjectionState,
    ramp_factor: float,
    num_mask: torch.Tensor,
    consistency_weight: float,
    consistency_sigma: float,
    device: torch.device
) -> Dict[str, Any]:
    """Epoca negative learning AE-SAD con target invertiti per outliers."""
    model.train()
    n_seen = 0
    sum_loss_tot = 0.0
    sum_loss_in = 0.0
    sum_s_num = 0.0
    sum_s_den = 0.0
    sum_cos_Shat_S = 0.0
    sum_cos_Shat_1mS = 0.0
    n_s_real_total = 0.0
    last_grad = float("nan")
    nonfinite_flag = False

    threshold_S_t = proj_state.threshold_S_t
    lambda_eff = proj_state.lambda_eff

    for l, s, m in train_loader:
        optimizer.zero_grad()
        bs = l.size(0)

        # ----- loss L (inliers) -----
        l_hat = model(l)
        loss_in_vec = ((1 - criterion(l_hat, l)) ** 2)
        loss_in_mean = loss_in_vec.mean()

        # ----- loss S (outliers con negative learning) -----
        m = m.float()
        num_s_real = float(m.sum().item())

        s_hat = model(s)
        loss_cons_mean = torch.tensor(0.0, device=device)

        if num_s_real > 0:
            target = F_feature_aware(s)
            loss_s_vec = (1 - criterion(s_hat, target))

            cos_s = criterion(s_hat, s)
            cos_neg = criterion(s_hat, target)

            w = m
            wden = w.sum().clamp(min=1e-12)

            # confidence weighting
            if threshold_S_t is not None:
                conf = (threshold_S_t - cos_s).clamp(min=0.0)
                conf = conf * m
                conf_max = conf.max()
                # fallback pesi uniformi se tutti sopra soglia
                if conf_max.item() > 1e-12:
                    conf = (conf / conf_max).clamp(min=0.1, max=1.0)
                    w = conf
                    wden = w.sum().clamp(min=1e-12)

            loss_s_mean = (loss_s_vec * w).sum() / wden

            # consistency loss con rumore gaussiano
            if consistency_weight > 0.0:
                s_aug = s.clone()
                if num_mask.any():
                    noise = consistency_sigma * torch.randn_like(s_aug[:, num_mask])
                    s_aug[:, num_mask] = (s_aug[:, num_mask] + noise).clamp(0.0, 1.0)
                s_hat_aug = model(s_aug)
                cons_vec = torch.nn.functional.mse_loss(s_hat, s_hat_aug, reduction="none").mean(dim=1)
                loss_cons_mean = (cons_vec * w).sum() / wden

            sum_s_num += (loss_s_vec * w).sum().item()
            sum_s_den += wden.item()
            sum_cos_Shat_S += (cos_s * m).sum().item()
            sum_cos_Shat_1mS += (cos_neg * m).sum().item()
            n_s_real_total += num_s_real
        else:
            loss_s_mean = torch.tensor(0.0, device=device)

        # total loss
        lam = ramp_factor * lambda_eff
        loss = loss_in_mean + lam * loss_s_mean
        if consistency_weight > 0.0:
            loss = loss + consistency_weight * loss_cons_mean

        if not torch.isfinite(loss).item():
            nonfinite_flag = True
            continue

        loss.backward()
        last_grad = _grad_norm_l2(model)
        optimizer.step()

        n_seen += bs
        sum_loss_tot += loss.item() * bs
        sum_loss_in += loss_in_mean.item() * bs

    return {
        "n_seen": n_seen,
        "sum_loss_tot": sum_loss_tot,
        "sum_loss_in": sum_loss_in,
        "sum_s_num": sum_s_num,
        "sum_s_den": sum_s_den,
        "sum_cos_Shat_S": sum_cos_Shat_S,
        "sum_cos_Shat_1mS": sum_cos_Shat_1mS,
        "n_s_real_total": n_s_real_total,
        "last_grad": last_grad,
        "nonfinite_flag": nonfinite_flag,
        "lambda_used": ramp_factor * lambda_eff,
    }


def _perform_projection(
    model: nn.Module,
    X: torch.Tensor,
    criterion: nn.CosineSimilarity,
    bin_estimator: str,
    epoch: int,
    warmup_epochs: int,
    proj_state: ProjectionState,
    lambda_s: float,
    lambda_card_beta: float,
    lambda_card_mode: str,
    curriculum_nl: bool,
    curriculum_start: float,
    curriculum_end: float,
    curriculum_min_S: int,
    epochs: int,
    device: torch.device,
    verbose: bool,
    plot_hist: bool = False
) -> Tuple[float, float, torch.Tensor, Optional[torch.Tensor]]:
    """Proiezione: calcola score, soglie unimodali, separa X in L e S."""
    model.eval()
    cos_scores = criterion(model(X), X).detach().cpu().numpy()

    if bin_estimator == "knuth":
        bins = knuth_bin_width(cos_scores, 10, 250, 1)
    else:
        bins = np.histogram_bin_edges(cos_scores, bins=bin_estimator)

    histogram_X = np.histogram(cos_scores, bins=bins)
    hist_X = histogram_X[0][::-1].astype(int)

    bin_centers_X = ((histogram_X[1][1:] + histogram_X[1][:-1]) / 2)[::-1]
    _, threshold = unimodal(hist_X, bin_centers_X)
    threshold_L = float(threshold)
    threshold_S = float(threshold_L)  # default safe

    # Calcola Y_L e Y_S
    if epoch <= warmup_epochs:
        first_time_no_S = (getattr(proj_state, "S_count", 0) == 0)

        bins_len = len(hist_X)
        centers_all = ((histogram_X[1][1:] + histogram_X[1][:-1]) / 2)   # crescente
        centers_all_rev = centers_all[::-1]                               # decrescente

        if first_time_no_S:
            #th_L == th_S
            _, th = unimodal(hist_X, bin_centers_X)  # bin_centers_X è decrescente
            threshold_L = float(th)
            threshold_S = float(th)
        else:
            hist_L = histogram_X[0][::-1][:bins_len // 2].astype(int)
            centers_L = centers_all_rev[:bins_len // 2]  # decrescente

            hist_S = histogram_X[0][:bins_len // 2].astype(int)
            centers_S = centers_all[:bins_len // 2]      # crescente

            _, thL = unimodal(hist_L, centers_L)
            _, thS = unimodal(hist_S, centers_S)
            threshold_L = float(thL)
            threshold_S = float(thS)

        Y_L = np.where(cos_scores > threshold_L)[0]
        Y_S = np.where(cos_scores < threshold_S)[0]

        # Salva versione tensor della soglia S 
        if np.isfinite(threshold_S):
            proj_state.threshold_S_t = torch.tensor(threshold_S, device=device)
        else:
            proj_state.threshold_S_t = None


    else:
        # NEGATIVE LEARNING
        bins_len = len(hist_X)
        centers_all = ((histogram_X[1][1:] + histogram_X[1][:-1]) / 2)  
        centers_all_rev = centers_all[::-1]                               

        # 1) soglia L (parte destra: cos alti)
        hist_L = histogram_X[0][::-1][:bins_len // 2].astype(int)
        centers_L = centers_all_rev[:bins_len // 2]
        _, thL = unimodal(hist_L, centers_L)
        threshold_L = float(thL)

        # 2) soglia S + EMA
        hist_S = histogram_X[0][:bins_len // 2].astype(int)
        centers_S = centers_all[:bins_len // 2]
        _, thS_raw = unimodal(hist_S, centers_S)

        if proj_state.thS_ema is None:
            proj_state.thS_ema = float(thS_raw)
        else:
            beta = 0.90
            proj_state.thS_ema = beta * proj_state.thS_ema + (1 - beta) * float(thS_raw)

        threshold_S = float(proj_state.thS_ema)

        margin = 1e-4
        if threshold_S >= threshold_L - margin:
            threshold_S = threshold_L - margin

        if np.isfinite(threshold_S):
            proj_state.threshold_S_t = torch.tensor(threshold_S, device=device)
        else:
            proj_state.threshold_S_t = None

        Y_L = np.where(cos_scores > threshold_L)[0]
        Y_S = np.where(cos_scores < threshold_S)[0]

        # Curriculum NL
        if curriculum_nl and (len(Y_S) >= curriculum_min_S):
            denom = max(epochs - warmup_epochs, 1)
            prog = float(np.clip((epoch - warmup_epochs) / denom, 0.0, 1.0))
            frac = float(np.clip(curriculum_start + (curriculum_end - curriculum_start) * prog, 0.0, 1.0))
            cos_S = cos_scores[Y_S]
            cutoff = np.quantile(cos_S, frac)
            Y_S = Y_S[cos_S <= (cutoff + 1e-12)]

    # plot istogramma
    if plot_hist:
        plt.figure(figsize=(10, 7))
        plt.hist(cos_scores, bins=bins)
        plt.axvline(threshold_L, color="red", linewidth=2, label="th_L")
        if np.isfinite(threshold_S):
            plt.axvline(threshold_S, color="green", linewidth=2, label="th_S")
        plt.legend()
        if verbose:
            plt.show()
        else:
            plt.close()

    L_new = X[Y_L]
    S_raw = X[Y_S] if (len(Y_S) > 0) else None

    have_S = (S_raw is not None) and (S_raw.shape[0] > 0)

    proj_state.S_raw = S_raw
    proj_state.S_count = 0 if (S_raw is None) else int(S_raw.shape[0])

    # aggiorna lambda_eff con EMA cardinalita
    if have_S:
        S_size = max(len(Y_S), 1)
        if proj_state.ema_S is None:
            proj_state.ema_S = float(S_size)
        else:
            proj_state.ema_S = 0.9 * proj_state.ema_S + 0.1 * float(S_size)

        lambda_card = len(Y_L) / max(proj_state.ema_S, 1.0)

        if proj_state.lambda_card_ema is None:
            proj_state.lambda_card_ema = float(lambda_card)
        else:
            proj_state.lambda_card_ema = (lambda_card_beta * proj_state.lambda_card_ema +
                                          (1.0 - lambda_card_beta) * float(lambda_card))

        if lambda_card_mode == "linear":
            proj_state.lambda_eff = lambda_s * proj_state.lambda_card_ema
        elif lambda_card_mode == "sqrt":
            proj_state.lambda_eff = lambda_s * math.sqrt(proj_state.lambda_card_ema)
        elif lambda_card_mode == "log":
            proj_state.lambda_eff = lambda_s * np.log1p(proj_state.lambda_card_ema)
        else:
            raise ValueError(f"Unknown lambda_card_mode: {lambda_card_mode}. "
                             f"Valid values: 'linear', 'sqrt', 'log'")

        # smoothing EMA lambda_eff
        beta_lam = 0.90
        if proj_state.lambda_eff_ema is None:
            proj_state.lambda_eff_ema = float(proj_state.lambda_eff)
        else:
            proj_state.lambda_eff_ema = beta_lam * proj_state.lambda_eff_ema + (1.0 - beta_lam) * float(proj_state.lambda_eff)

        proj_state.lambda_eff = float(proj_state.lambda_eff_ema)

    if verbose:
        stats = _describe_scores(cos_scores)
        print("\n" + "-" * 74)
        print(f"Projection @ epoch {epoch}")
        bins_dbg = (len(bins) - 1) if isinstance(bins, (list, tuple, np.ndarray)) else int(bins)
        print(f"bins={bins_dbg}")
        print(f"threshold_L={threshold_L:.6f} | threshold_S="
              f"{threshold_S if np.isfinite(threshold_S) else float('nan')}")
        print(f"|L|={L_new.shape[0]} | |S_raw|={(S_raw.shape[0] if have_S else 0)} | "
              f"lambda_eff={proj_state.lambda_eff:.3f}")
        print("cos_scores stats: "
              f"min={stats['min']:.4f} mean={stats['mean']:.4f} max={stats['max']:.4f} | "
              f"p1={stats['p1']:.4f} p5={stats['p5']:.4f} p10={stats['p10']:.4f} "
              f"p50={stats['p50']:.4f} p90={stats['p90']:.4f} p95={stats['p95']:.4f} p99={stats['p99']:.4f}")
        U_size = X.shape[0] - int(Y_L.shape[0]) - int(Y_S.shape[0])
        print(f"|U|={U_size}")

        if L_new.numel() > 0:
            L_hat = model(L_new)
            muL, sdL = _fmt_mean_std(criterion(L_hat, L_new))
            print(f"mean cos(L_hat,L) = {muL:.6f} ± {sdL:.6f}")

        if have_S:
            S_hat_dbg = model(S_raw)
            muS, sdS = _fmt_mean_std(criterion(S_hat_dbg, S_raw))
            print(f"mean cos(S_hat,S) = {muS:.6f} ± {sdS:.6f}")
        print("-" * 74 + "\n")

    proj_state.threshold_L = threshold_L
    proj_state.threshold_S = threshold_S
    proj_state.have_S = have_S
    proj_state.L = L_new

    return threshold_L, threshold_S, L_new, S_raw


def _check_early_stopping(
    metric: float,
    epoch: int,
    warmup_epochs: int,
    proj_state: ProjectionState,
    model: nn.Module,
    patience: int,
    min_delta: float,
    device: torch.device,
    verbose: bool
) -> bool:
    """Early stopping basato su proiezioni. Ritorna True se deve fermarsi."""
    if not np.isfinite(metric):
        if verbose:
            print("   >>> EarlyStopping saltato (metrica non finita).")
        return False

    if epoch <= warmup_epochs:
        if (proj_state.best_model_state is None) or (metric < (proj_state.best_metric - min_delta)):
            proj_state.best_metric = metric
            proj_state.best_model_state = {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()}
            proj_state.best_epoch = epoch
            if verbose:
                print(f"   >>> Miglior warmup aggiornato: Lin={proj_state.best_metric:.6f} @ epoca {proj_state.best_epoch}")
        proj_state.bad_projections = 0
        proj_state.nl_baseline_set = False
        return False

    else:
        if not proj_state.nl_baseline_set:
            proj_state.best_metric = metric
            proj_state.best_model_state = {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()}
            proj_state.best_epoch = epoch
            proj_state.bad_projections = 0
            proj_state.nl_baseline_set = True
            proj_state.best_proj_snapshot = dict(
                threshold_L=proj_state.threshold_L,
                threshold_S=proj_state.threshold_S,
                threshold_S_t=(None if proj_state.threshold_S_t is None else float(proj_state.threshold_S_t.item())),
                thS_ema=proj_state.thS_ema,
                lambda_eff=proj_state.lambda_eff,
                lambda_card_ema=proj_state.lambda_card_ema,
                ema_S=proj_state.ema_S,
            )
            if verbose:
                print("   >>> Baseline NL proiezione impostata per EarlyStopping.")
            return False
        else:
            if metric < (proj_state.best_metric - min_delta):
                proj_state.best_metric = metric
                proj_state.best_model_state = {k: v.detach().cpu().clone()
                                               for k, v in model.state_dict().items()}
                proj_state.best_epoch = epoch
                proj_state.bad_projections = 0
                return False
            else:
                proj_state.bad_projections += 1
                if verbose:
                    print(f"   >>> EarlyStopping: {proj_state.bad_projections}/{patience} "
                          f"(metric={metric:.6f}, best={proj_state.best_metric:.6f})")
                if proj_state.bad_projections >= patience:
                    print(f"\nEARLY STOPPING: epoca {epoch}. "
                          f"Ripristino epoca migliore {proj_state.best_epoch} (best Lin={proj_state.best_metric:.6f})")
                    return True
                return False


# --- funzione fit principale ---

def fit(model, X, bin_estimator="knuth", epochs=20,
        projection_after_epochs_wu=50, projection_after_epochs_nl=50,
        batch_size=256, lr=0.001, verbose=False,
        warmup_epochs=0,
        lambda_s=0.0,
        early_stopping=True,
        patience=3,
        min_delta=1e-6,
        curriculum_nl=True,
        curriculum_start=0.10,
        curriculum_end=1.00,
        curriculum_min_S=200,
        consistency_weight=0.0,
        consistency_sigma=0.02,
        ramp_epochs=50,
        ramp_enabled=True,
        return_stats=False):
    """Training RADON: warmup + negative learning + early stopping."""
    device = next(model.parameters()).device
    criterion = nn.CosineSimilarity(dim=1, eps=1e-5)
    model.to(device)

    # nomi feature se X e' DataFrame
    feature_names = None
    if hasattr(X, "columns"):
        try:
            feature_names = list(X.columns)
        except Exception:
            feature_names = None

    # converti X in tensor
    if isinstance(X, torch.Tensor):
        X = X.float().to(device)
    elif hasattr(X, "values"):
        X = torch.FloatTensor(X.values).to(device)
    else:
        X = torch.FloatTensor(X).to(device)

    losses = []
    th, th_L, th_S = [], [], []
    
    # liste timing
    epoch_times: List[float] = []
    epoch_phases: List[str] = []
    epoch_has_projection: List[bool] = []

    # analisi feature tabulari
    analyzer = TabularFeatureAnalyzer(device)
    analyzer.analyze(X, feature_names=feature_names, verbose=verbose)
    num_mask = analyzer.num_mask

    F_feature_aware = _create_feature_inverter(num_mask)

    # inizializza stato proiezione
    proj_state = ProjectionState()
    proj_state.lambda_eff = float(lambda_s)
    proj_state.L = X
    proj_state.lambda_eff_ema = float(lambda_s)

    proj_state.S = None
    proj_state.S_raw = None
    proj_state.S_count = 0
    proj_state.have_S = False

    lambda_card_beta = 0.90
    lambda_card_mode = "sqrt"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def _make_loader_warmup(L_tensor, S_tensor):
        """Loader warmup."""
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(L_tensor,S_tensor),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

    def _make_loader_nl(L_tensor, S_tensor, S_mask):
        """Loader negative learning."""
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(L_tensor, S_tensor, S_mask),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

    # -------------------------------------------------------------------------
    # inizializzazione: L = X, S = 0
    # -------------------------------------------------------------------------
    proj_state.L = X
    proj_state.S = torch.zeros_like(X)
    proj_state.have_S = False
    proj_state.S_raw = None
    proj_state.S_count = 0
    proj_state.threshold_S_t = None
    train_loader = _make_loader_warmup(proj_state.L, proj_state.S)

    # -------------------------------------------------------------------------
    # fattore rampa
    # -------------------------------------------------------------------------
    def _ramp_factor(epoch: int) -> float:
        """Fattore rampa per curriculum NL."""
        if not ramp_enabled:
            return 1.0
        if epoch <= warmup_epochs:
            return 0.0
        progress = (epoch - warmup_epochs) / max(ramp_epochs, 1)
        return min(1.0, progress)

    # -------------------------------------------------------------------------
    # header debug
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 72)
        print("Metadati esperimento (RADON warmup + NEG_LRN)")
        print(f"device: {device}")
        print(f"|X| = {X.shape[0]} | dim = {X.shape[1]}")
        print(f"epochs = {epochs} | proj_wu={projection_after_epochs_wu} | proj_nl={projection_after_epochs_nl}")
        print(f"batch_size = {batch_size} | lr = {lr} | bin_estimator = {bin_estimator}")
        print(f"warmup_epochs = {warmup_epochs}  (WARMUP)")
        print(f"lambda_s = {lambda_s}")
        print(f"param_norm_l2(init) = {_param_norm_l2(model):.6e}")
        print("=" * 72 + "\n")

    # =========================================================================
    # loop training
    # =========================================================================
    stop_training = False

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Determina fase
        use_negative_learning = (epoch > warmup_epochs)

        if epoch == warmup_epochs + 1:
            proj_state.lambda_card_ema = None
            proj_state.nl_baseline_set = False
            
            num_L = proj_state.L.shape[0]
            D = proj_state.L.shape[1]
            S, S_mask = _pad_S_to_L(proj_state.S_raw, num_L, D, device)

            proj_state.S = S
            proj_state.S_mask = S_mask
            train_loader = _make_loader_nl(proj_state.L, S, S_mask)

        phase_str = "NEG_LRN" if use_negative_learning else "WARMUP(RADON)"

        # ---------------------------------------------------------------------
        # epoca training
        # ---------------------------------------------------------------------
        if not use_negative_learning:
            epoch_stats = _train_warmup_epoch(
                model, train_loader, optimizer, criterion, device
            )
        else:
            ramp = _ramp_factor(epoch)
            epoch_stats = _train_negative_learning_epoch(
                model, train_loader, optimizer, criterion,
                F_feature_aware, proj_state, ramp, num_mask,
                consistency_weight, consistency_sigma, device
            )

        # ---------------------------------------------------------------------
        # medie
        # ---------------------------------------------------------------------
        n_seen = epoch_stats["n_seen"]
        nonfinite_flag = epoch_stats["nonfinite_flag"]

        if (n_seen == 0) or nonfinite_flag:
            avg_loss = float("inf")
            avg_in = float("inf")
        else:
            avg_loss = epoch_stats["sum_loss_tot"] / n_seen
            avg_in = epoch_stats["sum_loss_in"] / n_seen

        # statistiche S
        s_den = epoch_stats.get("sum_s_den", 0.0)
        if s_den and (s_den > 0):
            avg_s_real = epoch_stats["sum_s_num"] / max(s_den, 1e-12)
            s_denom = max(epoch_stats.get("n_s_real_total", 0.0), 1.0)
            cos_Shat_S = epoch_stats.get("sum_cos_Shat_S", 0.0) / s_denom
            cos_Shat_1mS = epoch_stats.get("sum_cos_Shat_1mS", 0.0) / s_denom
            n_s_real_total = epoch_stats.get("n_s_real_total", 0.0)
        else:
            avg_s_real = 0.0
            cos_Shat_S = 0.0
            cos_Shat_1mS = float("nan")
            n_s_real_total = 0.0

        lam = epoch_stats["lambda_used"]
        ratio_s_in = (lam * avg_s_real) / (avg_in + 1e-9) if np.isfinite(avg_in) else float("nan")

        # step proiezione: forza al confine warmup_epochs
        do_proj = (
            (epoch <= warmup_epochs and epoch % projection_after_epochs_wu == 0) or
            (epoch == warmup_epochs) or
            (epoch > warmup_epochs and ((epoch - warmup_epochs) % projection_after_epochs_nl == 0))
        )

        if do_proj:
            with torch.no_grad():
                threshold_L, threshold_S, L_new, S_raw = _perform_projection(
                    model, X, criterion, bin_estimator, epoch, warmup_epochs,
                    proj_state, lambda_s, lambda_card_beta, lambda_card_mode,
                    curriculum_nl, curriculum_start, curriculum_end, curriculum_min_S,
                    epochs, device, verbose
                )

                # Aggiorna data loaders
                if epoch <= warmup_epochs:
                    num_L = L_new.shape[0]
                    D = L_new.shape[1]

                    if (S_raw is None) or (S_raw.shape[0] == 0):
                        S_new = torch.zeros_like(L_new)
                    else:
                        num_S = S_raw.shape[0]
                        if num_S >= num_L:
                            S_new = S_raw[:num_L]
                        else:
                            pad_n = num_L - num_S
                            S_new = torch.cat([S_raw, torch.zeros((pad_n, D), device=device)], dim=0)


                    proj_state.L = L_new
                    proj_state.S = S_new
                    train_loader = _make_loader_warmup(L_new, S_new)

                else:
                    num_L = L_new.shape[0]
                    D = L_new.shape[1]
                    S, S_mask = _pad_S_to_L(S_raw if proj_state.have_S else None, num_L, D, device)

                    proj_state.S = S
                    proj_state.L = L_new
                    proj_state.S_mask = S_mask
                    train_loader = _make_loader_nl(L_new, S, S_mask)

                th.append(float(threshold_L)) 
                th_L.append(float(threshold_L))
                th_S.append(float(threshold_S) if np.isfinite(threshold_S) else float("nan"))

            # Early stopping
            if early_stopping:
                should_stop = _check_early_stopping(
                    avg_in, epoch, warmup_epochs, proj_state,
                    model, patience, min_delta, device, verbose
                )
                if should_stop:
                    if proj_state.best_model_state is not None:
                        model.load_state_dict(proj_state.best_model_state, strict=True)
                        model.to(device)
                    stop_training = True

        # timing
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        epoch_phases.append("WARMUP" if not use_negative_learning else "NEG_LRN")
        epoch_has_projection.append(do_proj)

        # ---------------------------------------------------------------------
        # log output
        # ---------------------------------------------------------------------
        if verbose:
            pnorm = _param_norm_l2(model)
            print(
                f"Ep {epoch:03d} [{phase_str}] | "
                f"Loss={avg_loss:.6f} | Lin={avg_in:.6f} | "
                f"Ls(real)={avg_s_real:.6f} | λ={lam:.3g} | "
                f"Ratio(λ*Ls/Lin)={ratio_s_in:.2f}x | "
                f"S_real_sum={int(n_s_real_total)} | "
                f"cos(Ŝ,S)={cos_Shat_S:.3f} | cos(Ŝ,F(S))={cos_Shat_1mS:.3f} | "
                f"grad_norm={epoch_stats['last_grad']:.3e} | param_norm={pnorm:.3e} | "
                f"time={elapsed:.2f}s"
            )

        losses.append(avg_loss)

        if stop_training:
            break

    # -------------------------------------------------------------------------
    # ripristina modello migliore
    # -------------------------------------------------------------------------
    if early_stopping and proj_state.best_model_state is not None:
        if verbose:
            print(f"Fine training. Ripristino miglior stato modello da epoca {proj_state.best_epoch}")
        model.load_state_dict(proj_state.best_model_state, strict=True)
        model.to(device)
        if proj_state.best_proj_snapshot is not None:
            snap = proj_state.best_proj_snapshot
            proj_state.threshold_L = snap["threshold_L"]
            proj_state.threshold_S = snap["threshold_S"]
            proj_state.thS_ema = snap["thS_ema"]
            proj_state.lambda_eff = snap["lambda_eff"]
            proj_state.lambda_card_ema = snap["lambda_card_ema"]
            proj_state.ema_S = snap["ema_S"]

            tS = snap["threshold_S_t"]
            proj_state.threshold_S_t = None if tS is None else torch.tensor(tS, device=device)

            # soglie ripristinate
            th_L.append(proj_state.threshold_L)
            th_S.append(proj_state.threshold_S)
            th.append(proj_state.threshold_L)

    # -------------------------------------------------------------------------
    # summary timing
    # -------------------------------------------------------------------------
    timing_stats = _compute_timing_stats(epoch_times, epoch_phases, epoch_has_projection)
    if verbose:
        _print_timing_summary(timing_stats, verbose=True)
    
    # stats completo
    stats = {
        "epoch_times": epoch_times,
        "epoch_phases": epoch_phases,
        "epoch_has_projection": epoch_has_projection,
        "timing_summary": timing_stats,
    }

    # retrocompatibile
    if return_stats:
        return losses, th, th_L, th_S, stats
    else:
        return losses, th, th_L, th_S


# --- thresholding unimodale ---

def unimodal(hist, bin_centers):
    """Thresholding unimodale: distanza massima dalla retta picco-ultimo bin."""
    pos_peak = np.argmax(hist)
    peak_max = hist[pos_peak]
    p1 = np.array([pos_peak, peak_max])

    nz = np.where(hist != 0)[0]
    if len(nz) > 0:
        last_zeroBin = nz[-1] + 1
    else:
        last_zeroBin = len(hist) - 1

    p2 = np.array([last_zeroBin, 0])
    best_idx = -1
    max_dist = -1
    for x0 in range(pos_peak, last_zeroBin):
        y0 = hist[x0]
        a = p1 - p2
        b = [x0, y0] - p2
        cross_ab = a[0] * b[1] - b[0] * a[1]
        d = np.linalg.norm(cross_ab) / np.linalg.norm(a)
        if d > max_dist:
            best_idx = x0
            max_dist = d
    return best_idx, bin_centers[best_idx]


# --- stimatore bin Knuth ---

def knuth_bin_width(data, N_MIN, N_MAX, step, return_bins=False, quiet=True):
    """Stimatore bin di Knuth."""
    knuthF = _KnuthF(data)
    cost = []
    B = range(N_MIN, N_MAX + 1, step)
    for i in range(N_MIN, N_MAX + 1, step):
        cost.append(knuthF.eval(i))

    bins = B[np.argmin(cost)]
    return bins


class _KnuthF:
    def __init__(self, data):
        self.data = np.array(data, copy=True)
        if self.data.ndim != 1:
            raise ValueError("data should be 1-dimensional")
        self.data.sort()
        self.n = self.data.size
        from scipy import special
        self.gammaln = special.gammaln

    def bins(self, M):
        """Edge dei bin."""
        return np.linspace(self.data[0], self.data[-1], int(M) + 1)

    def __call__(self, M):
        return self.eval(M)

    def eval(self, M):
        M = int(M)

        if M <= 0:
            return np.inf

        bins = self.bins(M)
        nk, bins = np.histogram(self.data, bins)

        return -(self.n * np.log(M) +
                 self.gammaln(0.5 * M) -
                 M * self.gammaln(0.5) -
                 self.gammaln(self.n + 0.5 * M) +
                 np.sum(self.gammaln(nk + 0.5)))
