import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False


class PTMDataset(TensorDataset):
    def __init__(self, seq, zscale, cksaap, labels):
        super().__init__(
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(zscale, dtype=torch.float32),
            torch.tensor(cksaap, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )


class DynamicWeightedFusion(nn.Module):

    def __init__(self, dim_b, dim_c, dim_d, output_dim, hidden_dim=64):
        super().__init__()
        self.proj_b = nn.Linear(dim_b, output_dim)
        self.proj_c = nn.Linear(dim_c, output_dim)
        self.proj_d = nn.Linear(dim_d, output_dim)

        total_dim = dim_b + dim_c + dim_d
        self.weight_gen = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, feat_b, feat_c, feat_d):
        h_b = self.proj_b(feat_b)
        h_c = self.proj_c(feat_c)
        h_d = self.proj_d(feat_d)

        concat = torch.cat([feat_b, feat_c, feat_d], dim=1)
        weights = F.softmax(self.weight_gen(concat), dim=1)

        fused = (weights[:, 0:1] * h_b +
                 weights[:, 1:2] * h_c +
                 weights[:, 2:3] * h_d)

        return fused


class CNN_BiLSTM_CKSAAP_ZScale(nn.Module):

    def __init__(self, vocab_size=21, max_seq_len=33, embed_dim=300,
                 lstm_units=32,
                 dense_units=128, cksaap_dim=1764, zscale_dim=5,
                 contrastive_dim=64, use_contrastive=True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.use_contrastive = use_contrastive

        # 分支 B: BiLSTM (锚点)
        self.embed_b = nn.Embedding(vocab_size + 1, embed_dim)
        self.drop_b = nn.Dropout(0.7)
        self.lstm_b = nn.LSTM(embed_dim, lstm_units,
                              bidirectional=True, batch_first=True)
        self.dim_b = max_seq_len * lstm_units * 2

        # 分支 A: CKSAAP
        self.bn_c = nn.BatchNorm1d(cksaap_dim, momentum=0.01)
        self.fc_c = nn.Linear(cksaap_dim, 16)
        self.drop_c = nn.Dropout(0.7)
        self.dim_c = 16

        # 分支 C: ZScale
        self.conv_d1 = nn.Conv1d(zscale_dim, 16, 3, padding=1)
        self.pool_d = nn.MaxPool1d(kernel_size=2)
        self.drop_d1 = nn.Dropout(0.7)
        self.conv_d2 = nn.Conv1d(16, 8, 3, padding=1)
        self.fc_d = nn.Linear(8, 8)
        self.drop_d2 = nn.Dropout(0.7)
        self.dim_d = 8

        self.fusion = DynamicWeightedFusion(
            dim_b=self.dim_b,
            dim_c=self.dim_c,
            dim_d=self.dim_d,
            output_dim=dense_units,
            hidden_dim=64,
        )
        self.drop_fusion = nn.Dropout(0.7)
        self.fc_out = nn.Linear(dense_units, 1)

        if use_contrastive:
            self.proj_b = nn.Sequential(
                nn.Linear(self.dim_b, 128),
                nn.GELU(),
                nn.Linear(128, contrastive_dim),
            )
            self.proj_c = nn.Sequential(
                nn.Linear(self.dim_c, 128),
                nn.GELU(),
                nn.Linear(128, contrastive_dim),
            )
            self.proj_d = nn.Sequential(
                nn.Linear(self.dim_d, 128),
                nn.GELU(),
                nn.Linear(128, contrastive_dim),
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embed_b.weight, -0.05, 0.05)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, seq_input, zscale_input, cksaap_input):
        x_b = self.embed_b(seq_input)
        x_b = self.drop_b(x_b)
        x_b, _ = self.lstm_b(x_b)
        feat_b = x_b.reshape(x_b.size(0), -1)   # (B, dim_b)

        x_c = self.bn_c(cksaap_input)
        x_c = F.relu(self.fc_c(x_c))
        feat_c = self.drop_c(x_c)               # (B, dim_c)

        x_d = zscale_input.transpose(1, 2)
        x_d = F.relu(self.conv_d1(x_d))
        x_d = self.pool_d(x_d)
        x_d = self.drop_d1(x_d)
        x_d = F.relu(self.conv_d2(x_d))
        x_d = x_d.mean(dim=-1)
        x_d = F.relu(self.fc_d(x_d))
        feat_d = self.drop_d2(x_d)              # (B, dim_d)

        fused = self.fusion(feat_b, feat_c, feat_d)  # (B, dense_units)
        fused = self.drop_fusion(fused)
        logits = self.fc_out(fused).squeeze(-1)

        if self.use_contrastive:
            z_b = F.normalize(self.proj_b(feat_b), dim=-1)
            z_c = F.normalize(self.proj_c(feat_c), dim=-1)
            z_d = F.normalize(self.proj_d(feat_d), dim=-1)
            return logits, z_b, z_c, z_d

        return logits, None, None, None

    def get_fusion_weights(self, seq_input, zscale_input, cksaap_input):
        with torch.no_grad():
            x_b = self.embed_b(seq_input)
            x_b = self.drop_b(x_b)
            x_b, _ = self.lstm_b(x_b)
            feat_b = x_b.reshape(x_b.size(0), -1)

            x_c = self.bn_c(cksaap_input)
            x_c = F.relu(self.fc_c(x_c))
            feat_c = self.drop_c(x_c)

            x_d = zscale_input.transpose(1, 2)
            x_d = F.relu(self.conv_d1(x_d))
            x_d = self.pool_d(x_d)
            x_d = self.drop_d1(x_d)
            x_d = F.relu(self.conv_d2(x_d))
            x_d = x_d.mean(dim=-1)
            x_d = F.relu(self.fc_d(x_d))
            feat_d = self.drop_d2(x_d)

            concat = torch.cat([feat_b, feat_c, feat_d], dim=1)
            weights = F.softmax(self.fusion.weight_gen(concat), dim=1)

        return weights  # (B, 3)


def l2_penalty(model, l2_lambda=1e-3):
    return l2_lambda * (
        model.fc_c.weight.pow(2).sum() + model.fc_d.weight.pow(2).sum()
    )


def anchor_contrastive_loss(z_b, z_c, z_d, labels=None, temperature=0.07):
    device = z_b.device
    B = z_b.size(0)
    if B < 2:
        return torch.tensor(0.0, device=device)

    sim_bc = torch.matmul(z_b, z_c.t()) / temperature
    sim_bd = torch.matmul(z_b, z_d.t()) / temperature

    if labels is not None:
        labels = labels.view(-1, 1)
        same_class = (labels == labels.t()).float()
        mask_self = torch.eye(B, device=device)
        same_class = same_class * (1 - mask_self)
        sim_bb = torch.matmul(z_b, z_b.t()) / temperature
        sim_bb = sim_bb.masked_fill(mask_self.bool(), -1e9)

    candidates = torch.cat([sim_bc, sim_bd], dim=1)

    if labels is not None:
        candidates = torch.cat([candidates, sim_bb], dim=1)

    log_denom = torch.logsumexp(candidates, dim=1)

    pos_bc = torch.diagonal(sim_bc)
    pos_bd = torch.diagonal(sim_bd)

    loss = 0.0
    n_pos = 0

    loss = loss - (pos_bc - log_denom).mean()
    n_pos += 1

    loss = loss - (pos_bd - log_denom).mean()
    n_pos += 1

    if labels is not None:
        pos_bb = sim_bb.clone()
        pos_bb_masked = pos_bb.masked_fill(~same_class.bool(), 0.0)
        pos_bb_count = same_class.sum(dim=1).clamp(min=1)
        avg_pos_bb = (pos_bb_masked.sum(dim=1) / pos_bb_count)
        has_same = (pos_bb_count > 0).float()
        if has_same.sum() > 0:
            supervised_loss = -((avg_pos_bb - log_denom) * has_same).sum() / has_same.sum()
            loss = loss + supervised_loss
            n_pos += 1

    return loss / n_pos


class EarlyStopping:
    def __init__(self, patience=20, mode='max', verbose=False, delta=0.0):
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        score = val_metric if self.mode == 'max' else -val_metric
        if self.best_score is None:
            self.best_score = score
            return False
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False
