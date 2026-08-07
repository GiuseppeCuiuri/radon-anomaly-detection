import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def evaluate_and_plot(models, test_loader, input_dim, device, dataset_name, 
                     normal_digit=None, threshold_S_cos=None, verbose=True, plot=True):
    """Valuta modelli sul test set con boxplot anomaly score. Score = 1 - cos(xhat, x)."""
    dataset_name = dataset_name.upper().strip()
    repetitions = len(models)
    criterion = nn.CosineSimilarity(dim=1, eps=1e-5)
    all_preds = None
    if threshold_S_cos is not None:
        all_preds = [[] for _ in range(repetitions)]
    
    all_scores = [[] for _ in range(repetitions)]
    all_labels = [[] for _ in range(repetitions)]

    if verbose:
        print(f"Valutazione sul test set: {dataset_name}")

    for rep in range(repetitions):
        model = models[rep]
        model.eval()
        model.to(device)

        scores_temp = []
        labels_temp = []
        preds_temp = [] if threshold_S_cos is not None else None

        with torch.no_grad():
            for data, label in test_loader:
                data = data.float().reshape(-1, input_dim).to(device)
                recon_batch = model(data)
                cos_batch = criterion(recon_batch, data)
                batch_score = 1.0 - cos_batch

                scores_temp.extend(batch_score.detach().cpu().numpy())
                labels_temp.extend(label.detach().cpu().numpy())

                if threshold_S_cos is not None:
                    th_score = 1.0 - float(threshold_S_cos)
                    batch_pred = (batch_score > th_score).to(torch.int64)
                    preds_temp.extend(batch_pred.detach().cpu().numpy())

        if threshold_S_cos is not None:
            all_preds[rep] = np.array(preds_temp).reshape(-1).astype(int)
        all_scores[rep] = np.array(scores_temp).reshape(-1)
        all_labels[rep] = np.array(labels_temp).reshape(-1).astype(int)

        if verbose:
            unique, counts = np.unique(all_labels[rep], return_counts=True)
            dist = ", ".join([f"{k}:{v}" for k, v in zip(unique, counts)])
            print(f"Rep {rep+1}/{repetitions}: score [{all_scores[rep].min():.4f}, {all_scores[rep].max():.4f}] "
                  f"media={all_scores[rep].mean():.4f} | label={dist}")

    if "MNIST" in dataset_name.upper():
        if normal_digit is None:
            raise ValueError("MNIST richiede normal_digit (es. normal_digit=4)")
        for rep in range(repetitions):
            y = all_labels[rep]
            if not np.all(np.isin(np.unique(y), [0, 1])):
                all_labels[rep] = (y != int(normal_digit)).astype(int)

    if verbose:
        print("Generazione boxplot")
    
    if plot:
        x_tick_labels = ["Inliers (cifra normale)", "Outliers (altre)"] if "MNIST" in dataset_name else ["Inliers", "Outliers"]
        
        for rep in range(repetitions):
            scores = all_scores[rep]
            labels = all_labels[rep]
            inlier_scores = scores[labels == 0]
            outlier_scores = scores[labels == 1]

            plt.figure(figsize=(10, 8))
            bplot = plt.boxplot([inlier_scores, outlier_scores], positions=[0, 1], 
                                patch_artist=True, widths=0.6,
                                medianprops=dict(color="black", linewidth=1.5))
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            plt.ylabel('Score anomalia (1 - cos)', fontsize=12)
            plt.title(f'{dataset_name} - Rep {rep+1}', fontsize=14)
            plt.xticks([0, 1], x_tick_labels, fontsize=11)
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            sep_check = "ok" if (inlier_scores.mean() < outlier_scores.mean()) else "overlap"
            plt.xlabel(f"Separazione: {sep_check} | In={inlier_scores.mean():.3f} Out={outlier_scores.mean():.3f}")
            plt.show()

    if threshold_S_cos is not None:
        return all_scores, all_labels, all_preds
    return all_scores, all_labels


def plot_aucroc(all_scores, all_labels, dataset_name="", verbose=True, plot=True):
    """Calcola e plotta curve ROC, stampa media e std dell'AUC."""
    repetitions = len(all_scores)
    aucroc_list = []
    
    if verbose:
        print(f"Generazione curve ROC per {dataset_name}")

    for rep in range(repetitions):
        y_true = np.array(all_labels[rep]).reshape(-1).astype(int)
        y_scores = np.array(all_scores[rep]).reshape(-1).astype(float)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        aucroc_list.append(roc_auc)
        
        if plot:
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            plt.figure(figsize=(10, 7))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.5f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Casuale')
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='black', s=100, 
                        label=f'Soglia ottimale (J) = {optimal_threshold:.4f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f'ROC - {dataset_name} (Rep {rep+1})')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.show()

    mean_auc = np.mean(aucroc_list)
    std_auc = np.std(aucroc_list)
    if verbose:
        print(f"{dataset_name}: media={mean_auc:.5f} ± {std_auc:.5f} | valori={aucroc_list}")
    
    return aucroc_list


def plot_auprc(all_scores, all_labels, dataset_name="", verbose=True, plot=True):
    """Calcola e plotta curve Precision-Recall, stampa media e std dell'AUC."""
    repetitions = len(all_scores)
    aucpr_list = []
    
    if verbose:
        print(f"Generazione curve Precision-Recall per {dataset_name}")

    for rep in range(repetitions):
        y_true = np.array(all_labels[rep]).reshape(-1).astype(int)
        y_scores = np.array(all_scores[rep]).reshape(-1).astype(float)
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label=1)
        pr_auc = auc(recall, precision)
        aucpr_list.append(pr_auc)
        
        if plot:
            f1_proxy = (2 * precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_proxy)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]

            plt.figure(figsize=(10, 7))
            plt.plot(recall, precision, color='blue', lw=2, label=f'AUPRC = {pr_auc:.5f}')
            baseline = y_true.sum() / len(y_true)
            plt.plot([0, 1], [baseline, baseline], linestyle='--', color='gray', label=f'Baseline = {baseline:.4f}')
            plt.scatter(recall[optimal_idx], precision[optimal_idx], marker='o', color='red', s=100, 
                        label=f'Soglia ottimale (F1) = {optimal_threshold:.4f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR - {dataset_name} (Rep {rep+1})')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.show()

    mean_auc = np.mean(aucpr_list)
    std_auc = np.std(aucpr_list)
    if verbose:
        print(f"{dataset_name}: media={mean_auc:.5f} ± {std_auc:.5f} | valori={aucpr_list}")

    return aucpr_list


def visualize_mnist_reconstruction(model, test_loader, device, num_samples=5, verbose=True, plot=True):
    """Visualizza originale, ricostruzione e mappa errore per campioni MNIST."""
    if verbose:
        print("Generazione visualizzazioni MNIST")
    
    model.eval()
    model.to(device)
    criterion = nn.CosineSimilarity(dim=1, eps=1e-5)
    
    normal_samples = []
    anomaly_samples = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.float().reshape(-1, 784).to(device)
            recon = model(data)
            scores = 1.0 - criterion(recon, data)
            
            for i in range(len(data)):
                item = {
                    'orig': data[i].cpu().reshape(28, 28),
                    'recon': recon[i].cpu().reshape(28, 28),
                    'score': scores[i].item(),
                    'label': label[i].item()
                }
                if item['label'] == 0 and len(normal_samples) < num_samples:
                    normal_samples.append(item)
                elif item['label'] == 1 and len(anomaly_samples) < num_samples:
                    anomaly_samples.append(item)
            
            if len(normal_samples) >= num_samples and len(anomaly_samples) >= num_samples:
                break
    
    if plot:
        def plot_row(samples, title_prefix):
            fig, axes = plt.subplots(nrows=len(samples), ncols=3, figsize=(10, 2*len(samples)))
            plt.suptitle(f"{title_prefix}", fontsize=16)
            
            for idx, item in enumerate(samples):
                ax = axes[idx, 0]
                ax.imshow(item['orig'], cmap='gray')
                ax.set_title(f"Originale (etichetta: {item['label']})")
                ax.axis('off')
                
                ax = axes[idx, 1]
                ax.imshow(item['recon'], cmap='gray')
                ax.set_title("Ricostruzione")
                ax.axis('off')
                
                diff = torch.abs(item['orig'] - item['recon'])
                ax = axes[idx, 2]
                im = ax.imshow(diff, cmap='hot')
                ax.set_title(f"Errore (score: {item['score']:.4f})")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
            plt.tight_layout()
            plt.show()

        if normal_samples:
            plot_row(normal_samples, "Inliers")
        if anomaly_samples:
            plot_row(anomaly_samples, "Outliers")


@dataclass
class FeatureRow:
    """Feature numerica con metriche per analisi."""
    name: str
    x: float
    xhat: float
    ql: float
    qh: float
    outside: bool
    dscore: float
    err_pos: float
    err_neg: float
    pref: str


def build_onehot_groups(feature_names):
    """Raggruppa feature categoriche one-hot (protocol_type_*, service_*, flag_*)."""
    groups = defaultdict(list)
    for i, name in enumerate(feature_names):
        if name.startswith("protocol_type_"):
            groups["protocol_type"].append(i)
        elif name.startswith("service_"):
            groups["service"].append(i)
        elif name.startswith("flag_"):
            groups["flag"].append(i)
    return dict(groups)


def decode_onehot_observed(x_np, idxs, feature_names, top_k=2):
    """Decodifica valori in un gruppo one-hot, top-K."""
    vals = x_np[idxs]
    order = np.argsort(vals)[::-1][:top_k]
    return [(feature_names[idxs[i]], float(vals[i])) for i in order]


def get_numeric_indices(feature_names, group_indices_flat):
    """Indici delle feature numeriche (non categoriche)."""
    group_set = set(group_indices_flat)
    return [i for i in range(len(feature_names)) if i not in group_set]


def _is_constant_range(ql, qh, eps=1e-3):
    """Verifica se range quasi costante."""
    return (qh - ql) < eps


def onehot_decode(x_row, idxs, feature_names, top=2):
    """Decodifica gruppi one-hot con nome breve."""
    vals = x_row[idxs]
    order = np.argsort(vals)[::-1][:top]
    return [(feature_names[idxs[i]].split("_", 1)[-1] if "_" in feature_names[idxs[i]] else feature_names[idxs[i]], 
             float(vals[i])) for i in order]


def build_numeric_rows(indices, x_np, xhat_np, ql_np, qh_np, dscore_map, err_pos, err_neg, feature_names):
    """Costruisce lista FeatureRow per feature numeriche."""
    rows = []
    for j in indices:
        outside = (x_np[j] < ql_np[j]) or (x_np[j] > qh_np[j])
        pref = "NEG" if err_neg[j] < err_pos[j] else "POS"
        rows.append(FeatureRow(
            name=feature_names[j],
            x=float(x_np[j]),
            xhat=float(xhat_np[j]),
            ql=float(ql_np[j]),
            qh=float(qh_np[j]),
            outside=bool(outside),
            dscore=float(dscore_map.get(j, 0.0)),
            err_pos=float(err_pos[j]),
            err_neg=float(err_neg[j]),
            pref=pref
        ))
    return rows


def filter_numeric_rows_causal(rows, min_abs_dscore=1e-3, min_clip_delta=0.02):
    """Filtra feature numeriche per attribution causale."""
    out = []
    for r in rows:
        if _is_constant_range(r.ql, r.qh):
            continue
        clip_val = min(max(r.x, r.ql), r.qh)
        if (not r.outside) and (abs(r.x - clip_val) < min_clip_delta):
            continue
        if abs(r.dscore) < min_abs_dscore:
            continue
        out.append(r)
    return out


def filter_group_rank(group_rank_raw, min_abs_dscore=1e-3):
    """Filtra gruppi categorici con impatto trascurabile."""
    return [(g, dv) for (g, dv) in group_rank_raw if abs(dv) >= min_abs_dscore]


def print_executive_summary(tag, score, group_rank, groups, group_modes, x_np, feature_names, 
                            num_rows_causal, num_rows_viol, top_causal=3, top_viol=3, verbose=False):
    """Stampa cause di anomalia per un campione."""
    if not verbose:
        return
    
    print(f"\nSummary: {tag} | score={score:.4f}")
    
    if group_rank:
        print("Ablazione categorica (Dscore):")
        for g, dv in group_rank[:3]:
            if g in groups:
                observed = onehot_decode(x_np, groups[g], feature_names, top=2)
                baseline_name = feature_names[group_modes[g]].split("_", 1)[-1] if "_" in feature_names[group_modes[g]] else feature_names[group_modes[g]]
                obs_str = ", ".join([f"{n}={v:.2f}" for n, v in observed if v > 0.01])
                if not obs_str:
                    obs_str = "nessuna attiva"
                print(f"  - {g:14s}: [{obs_str}] -> {baseline_name}  Dscore={dv:+.4f}")
    
    if num_rows_causal:
        print("\nAblazione numerica (Dscore):")
        for r in num_rows_causal[:top_causal]:
            status = "out" if r.outside else "in"
            print(f"  - {r.name:28s}: x={r.x:.3f} clip=[{r.ql:.3f},{r.qh:.3f}] ({status}) Dscore={r.dscore:+.4f}")
    
    if num_rows_viol:
        neg_count = sum(1 for r in num_rows_viol[:10] if r.pref == "NEG")
        print(f"\nViolazioni ricostruzione (top {top_viol}):")
        for r in num_rows_viol[:top_viol]:
            nl_marker = "NL" if r.pref == "NEG" else ""
            print(f"  - {r.name:28s}: x={r.x:.3f} xhat={r.xhat:.3f} |x-xhat|={r.err_pos:.3f} |(1-x)-xhat|={r.err_neg:.3f} {nl_marker}")
        print(f"Check NL: {neg_count}/10 feature preferiscono target negativo")
    print()


def print_audit_table(rows, title, show_dscore=True, top_k=8, verbose=False):
    """Tabella audit per feature numeriche."""
    if not verbose or not rows:
        return
    
    print(f"\n{title}")
    if show_dscore:
        print(f"{'feature':<30} | {'x':>6} | {'xhat':>6} | {'[ql,qh]':>15} | {'out':>3} | {'Dscore':>8} | {'|x-xhat|':>8} | {'|(1-x)-xhat|':>12} | {'pref':>4}")
        for r in rows[:top_k]:
            out_str = "si" if r.outside else "no"
            print(f"{r.name:<30} | {r.x:>6.3f} | {r.xhat:>6.3f} | [{r.ql:.3f},{r.qh:.3f}] | {out_str:>3} | {r.dscore:>+8.4f} | {r.err_pos:>8.3f} | {r.err_neg:>12.3f} | {r.pref:>4}")
    else:
        print(f"{'feature':<30} | {'x':>6} | {'xhat':>6} | {'|x-xhat|':>8} | {'|(1-x)-xhat|':>12} | {'pref':>4}")
        for r in rows[:top_k]:
            print(f"{r.name:<30} | {r.x:>6.3f} | {r.xhat:>6.3f} | {r.err_pos:>8.3f} | {r.err_neg:>12.3f} | {r.pref:>4}")


def build_thresholds_and_group_modes(X_train, y_train, feature_names, q_low=0.05, q_high=0.95):
    """Calcola quantili dai normali e modalita dominanti per gruppi one-hot."""
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train)
    y_train = y_train.view(-1)

    Xn = X_train[y_train == 0].float()
    ql = torch.quantile(Xn, q_low, dim=0)
    qh = torch.quantile(Xn, q_high, dim=0)

    groups = build_onehot_groups(feature_names)
    group_flat = [i for idxs in groups.values() for i in idxs]
    num_idx = get_numeric_indices(feature_names, group_flat)

    group_modes = {}
    for g, idxs in groups.items():
        freqs = Xn[:, idxs].mean(dim=0)
        k_star = torch.argmax(freqs).item()
        group_modes[g] = idxs[k_star]

    return ql, qh, group_modes, groups, num_idx


@torch.no_grad()
def cosine_score(model, x, cos=None):
    """Score anomalia: 1 - cos(xhat, x)."""
    if cos is None:
        cos = nn.CosineSimilarity(dim=1, eps=1e-5)
    xhat = model(x)
    s = 1.0 - cos(xhat, x)
    return xhat, s


@torch.no_grad()
def explain_nslkdd(
    model,
    test_loader,
    input_dim,
    device,
    X_train, y_train,
    feature_names,
    q_low=0.05, q_high=0.95,
    top_k_num=8,
    top_k_groups=3,
    n_anom=1,
    n_norm=2,
    add_borderline_norm=True,
    pool_target_anom=80,
    pool_target_norm=300,
    min_abs_dscore=1e-3,
    min_clip_delta=0.02,
    min_group_dscore=1e-3,
    verbose=True,
    plot=True
):
    """Analisi XAI per NSL-KDD: ablation categoriale/numerica e diagnostica Negative Learning."""
    if len(feature_names) != input_dim:
        raise ValueError(f"feature_names length ({len(feature_names)}) != input_dim ({input_dim})")

    model.eval().to(device)
    cos = nn.CosineSimilarity(dim=1, eps=1e-5)

    X_train = X_train.float().reshape(-1, input_dim)
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train)
    y_train = y_train.view(-1)

    ql, qh, group_modes, groups, num_idx = build_thresholds_and_group_modes(
        X_train, y_train, feature_names, q_low=q_low, q_high=q_high
    )
    M = len(num_idx)

    ql, qh = ql.to(device), qh.to(device)
    ql_full = ql.detach().cpu().numpy()
    qh_full = qh.detach().cpu().numpy()

    anom_pool = []
    norm_pool = []

    for data, label in test_loader:
        x = data.float().reshape(-1, input_dim).to(device)
        lab = label.to(device).view(-1).long()
        xhat, score = cosine_score(model, x, cos=cos)

        for i in range(x.shape[0]):
            item = (score[i].item(), x[i:i+1].detach(), xhat[i:i+1].detach())
            if lab[i].item() == 1:
                anom_pool.append(item)
            else:
                norm_pool.append(item)

        if len(anom_pool) >= pool_target_anom and len(norm_pool) >= pool_target_norm:
            break

    if len(anom_pool) == 0:
        if verbose:
            print("Nessuna anomalia trovata")
        return
    if len(norm_pool) == 0:
        if verbose:
            print("Nessun campione normale trovato")
        return

    anom_pool.sort(key=lambda t: t[0], reverse=True)
    norm_pool.sort(key=lambda t: t[0])

    selected = []
    for k in range(min(n_anom, len(anom_pool))):
        selected.append(("ANOMALY", anom_pool[k]))
    for k in range(min(n_norm, len(norm_pool))):
        selected.append(("NORMAL", norm_pool[k]))
    if add_borderline_norm and len(norm_pool) > 10:
        selected.append(("NORMAL (borderline)", norm_pool[-1]))

    for tag, (s0, x0, x0hat) in selected:
        x0 = x0.to(device)
        x0hat = x0hat.to(device)

        _, s0_check = cosine_score(model, x0, cos=cos)
        s0_check = s0_check.item()
        if abs(s0_check - s0) > 1e-4 and verbose:
            print(f"Warning [{tag}]: discrepanza score pool={s0:.6f} vs check={s0_check:.6f}")

        x0hat, _ = cosine_score(model, x0, cos=cos)
        
        group_delta = {}
        for g, idxs in groups.items():
            xg = x0.clone()
            xg[0, idxs] = 0.0
            xg[0, group_modes[g]] = 1.0
            _, sg = cosine_score(model, xg, cos=cos)
            group_delta[g] = (sg.item() - s0)

        group_rank_raw = sorted(group_delta.items(), key=lambda kv: kv[1])
        group_rank = filter_group_rank(group_rank_raw, min_abs_dscore=min_group_dscore)[:top_k_groups]

        if M == 0:
            if verbose:
                print(f"[{tag}] Ablazione numerica saltata (M=0)")
            pos_map = {}
        else:
            x_base = torch.clamp(x0, ql, qh)
            Xpert = x0.repeat(M, 1)
            rows_tensor = torch.arange(M, device=device)
            cols_tensor = torch.tensor(num_idx, device=device, dtype=torch.long)
            Xpert[rows_tensor, cols_tensor] = x_base[0, cols_tensor]

            Xpert_hat = model(Xpert)
            s_pert = 1.0 - cos(Xpert_hat, Xpert)
            delta_num = (s_pert - s0).detach().cpu().numpy()
            pos_map = {num_idx[p]: delta_num[p] for p in range(M)}

        x_np = x0[0].detach().cpu().numpy()
        xhat_np = x0hat[0].detach().cpu().numpy()
        err_pos = np.abs(x_np - xhat_np)
        err_neg = np.abs((1.0 - x_np) - xhat_np)
        
        group_explanations = []
        for g, dscore in group_rank:
            idxs = groups[g]
            observed = decode_onehot_observed(x_np=x_np, idxs=idxs, feature_names=feature_names, top_k=2)
            baseline_idx = group_modes[g]
            baseline = feature_names[baseline_idx]
            observed_top1 = observed[0][0] if len(observed) > 0 else "nessuna attiva"
            is_mismatch = (observed_top1 != baseline)
            group_explanations.append({
                "group": g,
                "observed": observed,
                "baseline": baseline,
                "dscore": float(dscore),
                "mismatch": bool(is_mismatch),
            })

        if group_explanations and verbose:
            print("\nAudit categorico:")
            for info in group_explanations:
                obs_str = ", ".join([f"{name}={val:.2f}" for name, val in info["observed"] if val > 1e-3])
                if not obs_str:
                    obs_str = "nessuna attiva"
                flag = "X" if info["mismatch"] else "ok"
                baseline_short = info["baseline"].split("_", 1)[-1] if "_" in info["baseline"] else info["baseline"]
                print(f"  - {info['group']:14s} {flag} osservato=[{obs_str}] -> baseline={baseline_short} Dscore={info['dscore']:+.4f}")

        if group_explanations and verbose:
            n_mismatch = sum(1 for info in group_explanations if info["mismatch"])
            if n_mismatch > 0:
                print(f"Mismatch: {n_mismatch}/{len(group_explanations)} gruppi differiscono dalla baseline")
        
        all_num_rows = build_numeric_rows(
            indices=num_idx,
            x_np=x_np,
            xhat_np=xhat_np,
            ql_np=ql_full,
            qh_np=qh_full,
            dscore_map=pos_map,
            err_pos=err_pos,
            err_neg=err_neg,
            feature_names=feature_names
        ) if M > 0 else []
        
        num_rows_causal = filter_numeric_rows_causal(all_num_rows, min_abs_dscore=min_abs_dscore, min_clip_delta=min_clip_delta)
        num_rows_causal = sorted(num_rows_causal, key=lambda r: r.dscore)[:top_k_num]
        num_rows_viol = sorted(all_num_rows, key=lambda r: r.err_pos, reverse=True)[:top_k_num]
        
        print_executive_summary(
            tag=tag, score=s0,
            group_rank=group_rank,
            groups=groups,
            group_modes=group_modes,
            x_np=x_np,
            feature_names=feature_names,
            num_rows_causal=num_rows_causal,
            num_rows_viol=num_rows_viol,
            top_causal=3,
            top_viol=3,
            verbose=verbose
        )
        
        print_audit_table(num_rows_causal, "Ablazione numerica (Dscore)", show_dscore=True, top_k=top_k_num, verbose=verbose)
        print_audit_table(num_rows_viol, "Violazioni ricostruzione (|x-xhat|)", show_dscore=False, top_k=top_k_num, verbose=verbose)
        
        if plot and group_rank:
            g_labels = [g for g, _ in group_rank]
            g_vals = [v for _, v in group_rank]
            plt.figure(figsize=(8, 4))
            colors = ['red' if v < 0 else 'green' for v in g_vals]
            plt.bar(np.arange(len(g_labels)), g_vals, color=colors, alpha=0.8)
            plt.axhline(0.0, linewidth=1, color='black')
            plt.xticks(np.arange(len(g_labels)), g_labels, rotation=0)
            plt.ylabel("Dscore (negativo = colpevole)")
            plt.title(f"{tag} | score={s0:.4f} | Attribuzione categorica")
            plt.grid(True, linestyle="--", alpha=0.25)
            plt.tight_layout()
            plt.show()

        if plot and num_rows_causal:
            labels_c = [r.name for r in num_rows_causal]
            xvals_c = np.array([r.x for r in num_rows_causal])
            rvals_c = np.array([r.xhat for r in num_rows_causal])
            ql_c = np.array([r.ql for r in num_rows_causal])
            qh_c = np.array([r.qh for r in num_rows_causal])
            bvals_c = np.clip(xvals_c, ql_c, qh_c)
            dvals_c = np.array([r.dscore for r in num_rows_causal])

            fig, ax = plt.subplots(figsize=(12, 6))
            x_axis = np.arange(len(num_rows_causal))
            w = 0.25

            for i in range(len(num_rows_causal)):
                ax.fill_between([x_axis[i] - 0.4, x_axis[i] + 0.4], ql_c[i], qh_c[i], 
                               alpha=0.15, color='green', label='Intervallo normale' if i == 0 else "")

            ax.bar(x_axis - w, xvals_c, width=w, label="Ingresso x", color='steelblue', alpha=0.9)
            ax.bar(x_axis, rvals_c, width=w, label="Ricostruzione xhat", color='coral', alpha=0.9)
            ax.bar(x_axis + w, bvals_c, width=w, label="Baseline (clip)", color='seagreen', alpha=0.9)
            ax.set_ylabel("Valore (0..1)")
            ax.set_xticks(x_axis)
            ax.set_xticklabels(labels_c, rotation=45, ha="right")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.legend(loc="upper left")

            ax2 = ax.twinx()
            ax2.plot(x_axis, dvals_c, marker="o", linestyle="--", linewidth=2, color='purple', label="Dscore")
            ax2.axhline(0.0, linewidth=1, color='gray', linestyle=':')
            ax2.set_ylabel("Dscore (negativo = colpevole)", color='purple')
            ax2.legend(loc="upper right")
            ax.set_title(f"{tag} | score={s0:.4f} | Attribuzione numerica (clip su range normale)")
            plt.tight_layout()
            plt.show()

        if plot and num_rows_viol:
            labels_v = [r.name for r in num_rows_viol]
            xvals_v = np.array([r.x for r in num_rows_viol])
            rvals_v = np.array([r.xhat for r in num_rows_viol])
            epos_v = np.array([r.err_pos for r in num_rows_viol])
            eneg_v = np.array([r.err_neg for r in num_rows_viol])
            prefs_v = [r.pref for r in num_rows_viol]

            fig, ax = plt.subplots(figsize=(12, 6))
            x_axis = np.arange(len(num_rows_viol))
            w = 0.35

            ax.bar(x_axis - w/2, xvals_v, width=w, label="Ingresso x", color='steelblue', alpha=0.9)
            ax.bar(x_axis + w/2, rvals_v, width=w, label="Ricostruzione xhat", color='coral', alpha=0.9)
            ax.set_ylabel("Valore (0..1)")
            ax.set_xticks(x_axis)
            ax.set_xticklabels(labels_v, rotation=45, ha="right")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.legend(loc="upper left")

            ax2 = ax.twinx()
            ax2.plot(x_axis, epos_v, marker="o", linestyle="--", linewidth=2, color='red', label="|x-xhat| (errore ricostruzione)")
            ax2.plot(x_axis, eneg_v, marker="^", linestyle="--", linewidth=2, color='blue', label="|(1-x)-xhat| (target NL)")
            ax2.set_ylabel("Entita errore")
            ax2.legend(loc="upper right")

            for i, pref in enumerate(prefs_v):
                if pref == "NEG":
                    ax.axvspan(i - 0.45, i + 0.45, alpha=0.1, color='blue')

            ax.set_title(f"{tag} | score={s0:.4f} | Violazioni ricostruzione")
            plt.tight_layout()
            plt.show()
