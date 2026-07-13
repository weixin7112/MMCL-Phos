import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    accuracy_score,
    roc_auc_score,
)

from feature_extraction import (
    Dic_1_gram,
    ProSentence,
    pad_sequences,
    ZScale,
    compute_cksaap,
)
from Networks import (
    PTMDataset,
    CNN_BiLSTM_CKSAAP_ZScale,
    l2_penalty,
    EarlyStopping,
    anchor_contrastive_loss,
)


SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

path = "../datasets/"
path_model = "../models/dynamic_fusion/"
os.makedirs(path_model, exist_ok=True)

LAMBDA_CONTRASTIVE = 0.1
TEMPERATURE = 0.10
lr = 0.001
num_epochs = 80
batch_size = 32
k = 1
N_FOLDS = 5

def compute_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan')

    return {
        'acc': acc, 'mcc': mcc, 'sn': sn, 'sp': sp, 'auc': auc,
        'cm': cm,
    }


df_train = pd.read_csv(path + "ST-train.csv", delimiter=',')
df_test = pd.read_csv(path + "ST-test.csv", delimiter=',')

word_index1 = Dic_1_gram()
vocab_size = len(word_index1)

texts_train = [ProSentence(i, k) for i in df_train['Sequence']]
train_sequences = [[word_index1[aa] for aa in s.split(' ')] for s in texts_train]
data_token_train = [s.split(' ') for s in texts_train]
MAX_SEQUENCE_LENGTH = len(data_token_train[1])
Xtrain_all = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
ytrain_all = df_train['Label'].values.astype(np.float32)

text_test = [ProSentence(i, k) for i in df_test['Sequence']]
test_sequences = [[word_index1[aa] for aa in s.split(' ')] for s in text_test]
data_token_test = [s.split(' ') for s in text_test]
MAX_SEQUENCE_LENGTH = len(data_token_test[1])
Xtest = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
ytest = df_test['Label'].values.astype(np.float32)

X_zscale_train_all = ZScale(df_train['Sequence'])
X_zscale_test = ZScale(df_test['Sequence'])
X_cksaap_train_all = compute_cksaap(df_train['Sequence'], k_max=0)
X_cksaap_test = compute_cksaap(df_test['Sequence'], k_max=0)

print(f"Data loading complete. | train_seq={Xtrain_all.shape} test_seq={Xtest.shape} "
      f"max_len={MAX_SEQUENCE_LENGTH}")

test_loader = DataLoader(
    PTMDataset(Xtest, X_zscale_test, X_cksaap_test, ytest),
    batch_size=batch_size, shuffle=False,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | "
          f"cuDNN: {torch.backends.cudnn.version()}")

def evaluate_on_loader(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for seq_b, zs_b, ck_b, y_b in loader:
            seq_b, zs_b, ck_b = (
                seq_b.to(device), zs_b.to(device), ck_b.to(device),
            )
            logits, z_b, z_c, z_d = model(seq_b, zs_b, ck_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_b.numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    return y_prob, y_true, y_pred, metrics

def train_one_fold(fold_idx, train_idx, val_idx,
                   X_seq, X_zs, X_ck, y):
    Xtr_seq = X_seq[train_idx]
    Xva_seq = X_seq[val_idx]
    Xtr_zs = X_zs[train_idx]
    Xva_zs = X_zs[val_idx]
    Xtr_ck = X_ck[train_idx]
    Xva_ck = X_ck[val_idx]
    ytr = y[train_idx]
    yva = y[val_idx]

    print(f"\n{'#' * 60}")
    print(f"# Fold {fold_idx + 1}/{N_FOLDS} | "
          f"train={len(train_idx)} val={len(val_idx)}")
    print(f"{'#' * 60}")

    train_loader = DataLoader(
        PTMDataset(Xtr_seq, Xtr_zs, Xtr_ck, ytr),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        PTMDataset(Xva_seq, Xva_zs, Xva_ck, yva),
        batch_size=batch_size, shuffle=False,
    )

    model = CNN_BiLSTM_CKSAAP_ZScale(
        cksaap_dim=X_ck.shape[1],
        max_seq_len=MAX_SEQUENCE_LENGTH,
        use_contrastive=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=20)
    best_model_path = os.path.join(path_model, f"best_model_fold{fold_idx}.pt")

    smooth_lo, smooth_hi = 0.05, 0.95
    best_val_acc = 0.0

    # --- 训练循环 ---
    for epoch in range(num_epochs):
        # === Train ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for seq_b, zs_b, ck_b, y_b in train_loader:
            if seq_b.size(0) < 2:
                continue
            seq_b = seq_b.to(device)
            zs_b = zs_b.to(device)
            ck_b = ck_b.to(device)
            y_b = y_b.to(device)
            y_smooth = y_b * (smooth_hi - smooth_lo) + smooth_lo
            y_long = y_b.long()

            optimizer.zero_grad()
            logits, z_b, z_c, z_d = model(seq_b, zs_b, ck_b)

            loss_bce = criterion(logits, y_smooth)
            loss_ctr = anchor_contrastive_loss(z_b, z_c, z_d, y_long, TEMPERATURE)
            loss = (loss_bce +
                    LAMBDA_CONTRASTIVE * loss_ctr
                    + l2_penalty(model))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y_b).sum().item()
            train_total += y_b.size(0)

        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for seq_b, zs_b, ck_b, y_b in val_loader:
                seq_b, zs_b, ck_b, y_b = (
                    seq_b.to(device), zs_b.to(device),
                    ck_b.to(device), y_b.to(device),
                )
                logits, z_b, z_c, z_d = model(seq_b, zs_b, ck_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += (preds == y_b).sum().item()
                val_total += y_b.size(0)

        val_acc = val_correct / max(val_total, 1)

        print(f"Fold {fold_idx+1} Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss/max(len(train_loader),1):.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/max(len(val_loader),1):.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if early_stopping(val_acc):
            print(f"Fold {fold_idx+1} Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(best_model_path))

    val_prob, val_true, val_pred, val_metrics = evaluate_on_loader(
        model, val_loader, device
    )

    test_prob, test_true, test_pred, test_metrics = evaluate_on_loader(
        model, test_loader, device
    )
    print(f"Fold {fold_idx+1} Test Results | "
          f"ACC: {test_metrics['acc']:.4f}  MCC: {test_metrics['mcc']:.4f}  "
          f"SN: {test_metrics['sn']:.4f}  SP: {test_metrics['sp']:.4f}  "
          f"AUC: {test_metrics['auc']:.4f}")
    print(f"Confusion Matrix:\n{test_metrics['cm']}")

    return {
        'acc': test_metrics['acc'], 'mcc': test_metrics['mcc'],
        'sn': test_metrics['sn'], 'sp': test_metrics['sp'],
        'auc': test_metrics['auc'],
        'cm': test_metrics['cm'],
        'y_prob': test_prob, 'y_true': test_true,
    }


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
all_fold_probs = []

for fold_idx, (train_idx, val_idx) in enumerate(
    skf.split(Xtrain_all, ytrain_all)
):
    result = train_one_fold(
        fold_idx, train_idx, val_idx,
        Xtrain_all, X_zscale_train_all, X_cksaap_train_all, ytrain_all,
    )
    fold_results.append(result)
    all_fold_probs.append(result['y_prob'])

ensemble_prob = np.mean(all_fold_probs, axis=0)
ensemble_pred = (ensemble_prob >= 0.5).astype(int)
y_true = fold_results[0]['y_true']

metrics_ens = compute_metrics(y_true, ensemble_pred, ensemble_prob)

print("\n" + "=" * 60)
print("Results")
print("=" * 60)
print(f"ACC: {metrics_ens['acc']:.4f}")
print(f"MCC: {metrics_ens['mcc']:.4f}")
print(f"SN:  {metrics_ens['sn']:.4f}")
print(f"SP:  {metrics_ens['sp']:.4f}")
print(f"AUC: {metrics_ens['auc']:.4f}")
print("Confusion Matrix:")
print(metrics_ens['cm'])
