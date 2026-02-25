import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import pandas as pd

def get_data(dataset_name, root_dir, flatten=True):
    """Carica il dataset richiesto (MNIST o NSL_KDD) dalla cartella indicata."""
    name = dataset_name.upper().strip()
    print(f"[Loader] Richiesto: {name} | Root: {root_dir}")

    if name == "MNIST":
        X_train, Y_train, X_test, Y_test = _load_mnist_from_raw(root_dir, flatten)
        return X_train, Y_train, X_test, Y_test, None
    
    elif name == "NSL_KDD":
        return _load_nsl_kdd_preprocessed(root_dir)
    
    else:
        raise ValueError(f"Dataset '{name}' non riconosciuto.")

# --- MNIST ---
def _load_mnist_from_raw(root, flatten):
    """Carica MNIST raw via torchvision e normalizza in [0,1]."""
    # download se non presente
    ds_train = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    ds_test = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())

    # estrazione e normalizzazione
    X_train = ds_train.data.float() / 255.0
    Y_train = ds_train.targets.clone()
    
    X_test = ds_test.data.float() / 255.0
    Y_test = ds_test.targets.clone()

    # canale: (N, 28, 28) -> (N, 1, 28, 28)
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    if flatten:
        # (N, 784)
        X_train = X_train.view(X_train.size(0), -1)
        X_test = X_test.view(X_test.size(0), -1)

    print(f"MNIST caricato. X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    return X_train, Y_train, X_test, Y_test

# --- NSL-KDD ---
def _load_nsl_kdd_preprocessed(root):
    """Carica NSL-KDD da CSV pre-processati."""
    base_path = os.path.join(root, "NSL_KDD")
    
    files = {
        "X_train": os.path.join(base_path, "X_train"),
        "Y_train": os.path.join(base_path, "Y_train"),
        "X_test": os.path.join(base_path, "X_test"),
        "Y_test": os.path.join(base_path, "Y_test")
    }

    # Verifica esistenza file
    for name, fpath in files.items():
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Impossibile trovare il file: {fpath}")

    print("Caricamento tensori NSL-KDD...")
    
    try:
        df_xt = pd.read_csv(files["X_train"])
        df_yt = pd.read_csv(files["Y_train"])
        df_xs = pd.read_csv(files["X_test"])
        df_ys = pd.read_csv(files["Y_test"])
        
        # rimozione colonna indice se presente
        feature_names = df_xt.columns.tolist()
        if len(feature_names) > 0 and feature_names[0].startswith("Unnamed"):
            df_xt = df_xt.drop(columns=[feature_names[0]])
            feature_names = df_xt.columns.tolist()
        
        feature_names_test = df_xs.columns.tolist()
        if len(feature_names_test) > 0 and feature_names_test[0].startswith("Unnamed"):
            df_xs = df_xs.drop(columns=[feature_names_test[0]])

        # conversione in tensori
        X_train = torch.tensor(df_xt.values, dtype=torch.float32)
        X_test = torch.tensor(df_xs.values, dtype=torch.float32)
        
        # label 1D intere
        Y_train = torch.tensor(df_yt.values, dtype=torch.long).squeeze()
        Y_test = torch.tensor(df_ys.values, dtype=torch.long).squeeze()
        
    except Exception as e:
        print(f"Errore durante la lettura CSV: {e}")
        raise e

    print(f"NSL-KDD caricato. X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    return X_train, Y_train, X_test, Y_test, feature_names


# --- visualizzazione ---
def plot_distribution(Y_data, dataset_name, normal_class=None):
    """Plotta distribuzione inliers/outliers e stampa statistiche."""
    dataset_name = dataset_name.upper().strip()
    
    # Conversione tensor -> numpy
    if isinstance(Y_data, torch.Tensor):
        Y_np = Y_data.cpu().numpy()
    else:
        Y_np = Y_data

    if dataset_name == "MNIST":
        if normal_class is None:
            raise ValueError("Specificare 'normal_class' per MNIST")
        X_label = (1 * (Y_np != normal_class)).reshape(-1)
        labels_legend = [f"Inliers (Digit {normal_class})", "Outliers (Altri)"]
    else:
        X_label = Y_np.reshape(-1)
        labels_legend = ["Inliers (Normal)", "Outliers (Attack)"]

    # Plot istogramma
    plt.figure(figsize=(6, 4))
    plt.hist(X_label, bins=[-0.1, 0.1, 0.9, 1.1], rwidth=0.8, align='mid')
    plt.xticks([0, 1], labels_legend)
    plt.title(f"{dataset_name} - Distribuzione Dati Training")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # statistiche
    n_inliers = (X_label == 0).sum()
    n_outliers = (X_label == 1).sum()
    n_total = len(X_label)

    print("-" * 30)
    print(f"Statistiche {dataset_name}")
    print("-" * 30)
    print(f"Inliers      : {n_inliers}")
    print(f"Outliers     : {n_outliers}")
    print(f"Totale       : {n_total}")
    print(f"Contaminazione: {n_outliers / n_total:.4f} ({n_outliers / n_total * 100:.2f}%)")
    print("-" * 30)
    
    return X_label


# --- preprocessing ---
def preprocess_dataset(dataset_name, X_train, Y_train, X_test, Y_test, **kwargs):
    """Preprocessing specifico per dataset (subsampling, contaminazione). Label finali binarie 0/1."""
    name = dataset_name.upper().strip()
    seed = kwargs.get('seed', 42)
    
    # riproducibilita
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # CUDA determinismo
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"\nPreprocessing specifico per: {name}")

    if name == "MNIST":
        X_tr, Y_tr, X_te, Y_te = _preprocess_mnist(X_train, Y_train, X_test, Y_test, **kwargs)
    elif name == "NSL_KDD":
        X_tr, Y_tr, X_te, Y_te = _preprocess_nsl_kdd(X_train, Y_train, X_test, Y_test, **kwargs)
    else:
        raise ValueError(f"Dataset {name} non supportato nel preprocessing.")
    
    # controlli
    assert X_tr.dim() == 2, f"X_train deve essere 2D, trovato {X_tr.dim()}D"
    assert Y_tr.dim() == 1, f"Y_train deve essere 1D, trovato {Y_tr.dim()}D"
    assert X_tr.shape[0] == Y_tr.shape[0], "X_train e Y_train devono avere lo stesso numero di campioni"
    assert X_te.shape[0] == Y_te.shape[0], "X_test e Y_test devono avere lo stesso numero di campioni"
    
    # Verifica label binarie {0, 1}
    unique_train = torch.unique(Y_tr)
    unique_test = torch.unique(Y_te)
    assert torch.all((unique_train == 0) | (unique_train == 1)), "Y_train deve contenere solo {0, 1}"
    assert torch.all((unique_test == 0) | (unique_test == 1)), "Y_test deve contenere solo {0, 1}"
    
    return X_tr, Y_tr, X_te, Y_te

# --- logica MNIST ---
def _preprocess_mnist(X_train, Y_train, X_test, Y_test, **kwargs):
    """Seleziona un digit come classe normale, altri come anomalie, train contaminato e test bilanciato."""
    # parametri default
    normal_digit = kwargs.get('normal_digit', 4)
    n_normal = kwargs.get('n_normal', 4859)
    n_anomaly = kwargs.get('n_anomaly', 265)
    anomaly_digits = kwargs.get('anomaly_digits', [0, 2, 3])
    n_test_normal = kwargs.get('n_test_normal', None)
    n_test_anomaly = kwargs.get('n_test_anomaly', 265)

    print(f"   Config: Digit Normale={normal_digit} | N_Normal={n_normal} | N_Anomaly(Train)={n_anomaly}")

    # --- TRAINING SET ---
    # campioni normali
    idx_normal_all = (Y_train == normal_digit).nonzero(as_tuple=True)[0]
    perm_n = torch.randperm(len(idx_normal_all))
    idx_normal = idx_normal_all[perm_n[:n_normal]]
    X_normal = X_train[idx_normal]
    y_normal = torch.zeros(len(idx_normal), dtype=torch.long)

    # campioni anomali
    if n_anomaly > 0:
        anomaly_mask = torch.isin(Y_train, torch.tensor(anomaly_digits, device=Y_train.device))
        idx_anomaly_all = anomaly_mask.nonzero(as_tuple=True)[0]
        
        perm_a = torch.randperm(len(idx_anomaly_all))
        idx_anomaly = idx_anomaly_all[perm_a[:n_anomaly]]
        X_anomaly = X_train[idx_anomaly]
        y_anomaly = torch.ones(len(idx_anomaly), dtype=torch.long)
        
        X_train_final = torch.cat([X_normal, X_anomaly], dim=0)
        Y_train_final = torch.cat([y_normal, y_anomaly], dim=0)
    else:
        X_train_final = X_normal
        Y_train_final = y_normal

    # shuffle training
    perm_train = torch.randperm(X_train_final.shape[0])
    X_train_final = X_train_final[perm_train]
    Y_train_final = Y_train_final[perm_train]

    contamination = 0 if n_anomaly == 0 else n_anomaly / X_train_final.shape[0]
    print(f"   TRAIN completato: {X_train_final.shape} | Contaminazione: {contamination:.4f}")

    # --- TEST SET ---
    # campioni normali
    idx_test_normal_all = (Y_test == normal_digit).nonzero(as_tuple=True)[0]
    if n_test_normal is None or n_test_normal >= len(idx_test_normal_all):
        idx_test_normal = idx_test_normal_all
    else:
        perm_n_test = torch.randperm(len(idx_test_normal_all))
        idx_test_normal = idx_test_normal_all[perm_n_test[:n_test_normal]]

    X_test_normal = X_test[idx_test_normal]
    y_test_normal = torch.zeros(len(idx_test_normal), dtype=torch.long)

    # campioni anomali (tutti i digit diversi da normal_digit)
    idx_test_anom_all = (Y_test != normal_digit).nonzero(as_tuple=True)[0]
    if n_test_anomaly >= len(idx_test_anom_all):
        idx_test_anom = idx_test_anom_all
    else:
        perm_a_test = torch.randperm(len(idx_test_anom_all))
        idx_test_anom = idx_test_anom_all[perm_a_test[:n_test_anomaly]]

    X_test_anomaly = X_test[idx_test_anom]
    y_test_anomaly = torch.ones(len(idx_test_anom), dtype=torch.long)

    # concatena e mescola test
    X_test_final = torch.cat([X_test_normal, X_test_anomaly], dim=0)
    Y_test_final = torch.cat([y_test_normal, y_test_anomaly], dim=0)

    perm_test = torch.randperm(len(Y_test_final))
    X_test_final = X_test_final[perm_test]
    Y_test_final = Y_test_final[perm_test]

    n_norm_test = int((Y_test_final == 0).sum())
    n_anom_test = int((Y_test_final == 1).sum())
    anom_ratio = Y_test_final.float().mean().item()
    
    print(f"   TEST completato: {X_test_final.shape} | Normali={n_norm_test} | Anomalie={n_anom_test} | Ratio Anomalie={anom_ratio:.4f}")
    
    return X_train_final, Y_train_final, X_test_final, Y_test_final

# --- logica NSL-KDD ---
def _preprocess_nsl_kdd(X_train, Y_train, X_test, Y_test, **kwargs):
    """Sottocampiona outliers per raggiungere la contaminazione target."""
    target_contamination = kwargs.get('target_contamination', 0.01)

    y_np = Y_train.cpu().numpy().reshape(-1).astype(int)
    
    idx_in = np.where(y_np == 0)[0]
    idx_out_all = np.where(y_np == 1)[0]
    
    n_in = len(idx_in)
    
    # contaminazione: n_out = (c * n_in) / (1 - c)
    n_out_needed = int((target_contamination * n_in) / (1.0 - target_contamination))
    
    print(f"   Statistiche originali -> Inliers: {n_in}, Outliers disponibili: {len(idx_out_all)}")
    print(f"   Target contaminazione: {target_contamination*100:.1f}% -> Outliers richiesti: {n_out_needed}")
    
    if n_out_needed > len(idx_out_all):
        print(f"   ATTENZIONE: Richiesti {n_out_needed} outliers ma disponibili solo {len(idx_out_all)}. Uso tutti i disponibili.")
        n_out_needed = len(idx_out_all)

    # selezione casuale outliers
    rng = np.random.default_rng(kwargs.get('seed', 42))
    idx_out_sel = rng.choice(idx_out_all, size=n_out_needed, replace=False)
    
    # combina inliers + outliers selezionati
    idx_final = np.concatenate([idx_in, idx_out_sel])
    rng.shuffle(idx_final)
    
    X_train_final = X_train[idx_final]
    Y_train_final = Y_train[idx_final]
    
    # contaminazione effettiva
    contamination_real = Y_train_final.float().mean().item()
    
    print(f"   TRAIN completato: {X_train_final.shape} | Contaminazione effettiva: {contamination_real:.4f}")
    
    return X_train_final, Y_train_final, X_test, Y_test
# --- dataloaders ---
def create_dataloaders(dataset_name, X_train, Y_train, X_test, Y_test, batch_size=256, device='cpu'):
    """Crea DataLoader PyTorch per training e test."""
    name = dataset_name.upper().strip()
    print(f"\nCreazione DataLoaders per {name} (Batch={batch_size}, Device={device})...")

    # sposta dati sul device
    X_train = X_train.to(device).float()
    X_test = X_test.to(device).float()
    
    # label long per classificazione
    if Y_train is not None:
        Y_train = Y_train.to(device).long()
    if Y_test is not None:
        Y_test = Y_test.to(device).long()

    if name == "MNIST":
        return _dataloaders_mnist(X_train, X_test, Y_test, batch_size)
    
    elif name == "NSL_KDD":
        return _dataloaders_nsl_kdd(X_train, X_test, Y_test, batch_size)
    
    else:
        raise ValueError(f"Dataset {name} non supportato per i dataloaders.")


def _dataloaders_mnist(X_train, X_test, Y_test, batch_size):
    """DataLoader MNIST: train unsupervised (solo X), test con label."""
    train_loader = data.DataLoader(
        data.TensorDataset(X_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    test_loader = data.DataLoader(
        data.TensorDataset(X_test, Y_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    print(f"DataLoaders MNIST pronti. Batch training: {len(train_loader)}, Batch test: {len(test_loader)}")
    return train_loader, test_loader


def _dataloaders_nsl_kdd(X_train, X_test, Y_test, batch_size):
    """DataLoader NSL-KDD: train unsupervised (solo X), test con label."""
    train_loader = data.DataLoader(
        data.TensorDataset(X_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    test_loader = data.DataLoader(
        data.TensorDataset(X_test, Y_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    print(f"DataLoaders NSL-KDD pronti. Batch training: {len(train_loader)}, Batch test: {len(test_loader)}")
    return train_loader, test_loader