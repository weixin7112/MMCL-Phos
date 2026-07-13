import pandas as pd
import numpy as np
import re


def Dic_1_gram():
    AA_list_sort = ['G','A','V','L','I','M','P','F','W','S','T','N','Q','Y','C','K','R','H','D','E','X']
    AA_dict = {}
    numm = 1
    for i in AA_list_sort:
        AA_dict[i] = numm
        numm += 1
    return AA_dict


def ProSentence(pro, K):
    sentence = ""
    length = len(pro)
    for i in range(length - K + 1):
        sentence += pro[i: i + K] + " "
    sentence = sentence[0 : len(sentence) - 1]
    return sentence


def pad_sequences(sequences, maxlen):
    result = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        result[i, :length] = seq[:length]
    return result


def compute_cksaap(sequences, k_max=3):
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
    n_aa = len(aa_list)
    feats_all = []
    for seq in sequences:
        feats = []
        for k in range(k_max + 1):
            pair_counts = np.zeros((n_aa, n_aa), dtype=np.float32)
            for i in range(len(seq) - k - 1):
                j = i + k + 1
                if seq[i] in aa_to_idx and seq[j] in aa_to_idx:
                    pair_counts[aa_to_idx[seq[i]], aa_to_idx[seq[j]]] += 1
            total = pair_counts.sum()
            if total > 0:
                pair_counts /= total
            feats.append(pair_counts.flatten())
        feats_all.append(np.concatenate(feats))
    return np.array(feats_all, dtype=np.float32)


def ZScale(sequences):
    zscale = {
        'A': [0.24, -2.32, 0.60, -0.14, 1.30], 'C': [0.84, -1.67, 3.71, 0.18, -2.65],
        'D': [3.98, 0.93, 1.93, -2.46, 0.75], 'E': [3.11, 0.26, -0.11, -0.34, -0.25],
        'F': [-4.22, 1.94, 1.06, 0.54, -0.62], 'G': [2.05, -4.06, 0.36, -0.82, -0.38],
        'H': [2.47, 1.95, 0.26, 3.90, 0.09], 'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
        'K': [2.29, 0.89, -2.49, 1.49, 0.31], 'L': [-4.28, -1.30, -1.49, -0.72, 0.84],
        'M': [-2.85, -0.22, 0.47, 1.94, -0.98], 'N': [3.05, 1.62, 1.04, -1.15, 1.61],
        'P': [-1.66, 0.27, 1.84, 0.70, 2.00], 'Q': [1.75, 0.50, -1.44, -1.34, 0.66],
        'R': [3.52, 2.50, -3.50, 1.99, -0.17], 'S': [2.39, -1.07, 1.15, -1.39, 0.67],
        'T': [0.75, -2.18, -1.12, -1.46, -0.40], 'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
        'W': [-4.36, 3.94, 0.59, 3.44, -1.59], 'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],
        'X': [0, 0, 0, 0, 0],
    }
    encodings = []
    for seq in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(seq).upper())
        code = []
        for aa in sequence:
            singlecode = []
            if aa in zscale:
                singlecode = singlecode + zscale[aa]
            else:
                singlecode = singlecode + zscale['X']
            code.append(singlecode)
        encodings.append(code)
    return np.array(encodings).astype(np.float32)

if __name__ == '__main__':
    path = "../datasets/DeepPSP/"
    df_train = pd.read_csv(path + "ST-train.csv", delimiter=',')
    X_cksaap_train_all = compute_cksaap(df_train['Sequence'], k_max=0)
    print(X_cksaap_train_all.shape)
