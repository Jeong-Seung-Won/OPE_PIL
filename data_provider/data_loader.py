import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class KAFASATAnomalyDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='kompsat3a_train.csv',
                 scale=True, nonautoregressive=False, test_flag='T',
                 subset_rand_ratio=1.0, use_full_data=True, target_feature=None, stride = None):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        if stride is None:
            self.stride = self.seq_len 
        else:
            self.stride = stride
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        self.use_full_data = use_full_data

        self.manual_mean = None
        self.manual_std = None

        self.__read_data__()

    def __read_data__(self):
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        print(f"Loading KOMPSAT file anomaly dataset from: {dataset_file_path}")

        df_raw = pd.read_csv(dataset_file_path)
        print(f"Original data shape: {df_raw.shape}")
        print(f"Columns: {list(df_raw.columns)}")

        exclude_columns = ['time', 'phase', 'label', 'Anomaly', 'anomaly_type', 'Anomaly_Type']
        available_columns = [col for col in df_raw.columns if col not in exclude_columns]

        numeric_columns = []
        for col in available_columns:
            try:
                pd.to_numeric(df_raw[col])
                numeric_columns.append(col)
            except:
                print(f"Skipping non-numeric column: {col}")

        print(f"Auto-detected numeric columns: {numeric_columns}")

        import json
        columns_path = os.path.join(self.root_path, 'kompsat3a_feature_columns.json')

        selected_columns = list(numeric_columns)
        if self.flag == 'train':
            try:
                with open(columns_path, 'w') as f:
                    json.dump(selected_columns, f)
                print(f"Saved training feature columns to: {columns_path}")
            except Exception as e:
                print(f"[WARN] Failed to save feature columns: {e}")
        else:
            if os.path.exists(columns_path):
                try:
                    with open(columns_path, 'r') as f:
                        train_columns = json.load(f)
                    print(f"Loaded training feature columns from: {columns_path}")
                    for col in train_columns:
                        if col not in df_raw.columns:
                            df_raw[col] = 0.0
                    selected_columns = train_columns
                except Exception as e:
                    print(f"[WARN] Failed to load feature columns: {e}. Using auto-detected columns.")
            else:
                print(f"[WARN] Training feature columns not found at {columns_path}. Using auto-detected columns.")

        data_numeric = df_raw[selected_columns].values.astype(np.float32)
        self.numeric_columns = selected_columns

        if self.flag == 'test':
            if 'Anomaly' in df_raw.columns:
                self.labels = df_raw['Anomaly'].values
                print(f"Label(Anomaly) distribution in test set: {np.unique(self.labels, return_counts=True)}")
            elif 'label' in df_raw.columns:
                self.labels = df_raw['label'].values
                print(f"Label(label) distribution in test set: {np.unique(self.labels, return_counts=True)}")
            else:
                self.labels = None
                print("No label column ('Anomaly' or 'label') found for test set")
        else:
            self.labels = None
            print(f"No labels loaded for {self.flag} set")

        print(f"Numeric data shape: {data_numeric.shape}")
        print(f"Features: {numeric_columns}")

        data_len = len(data_numeric)
        print(f"Using full data - Total: {data_len} samples for {self.flag}")

        if self.scale:
            stats_filename = "kompsat3a_normalization_stats.npz"
            stats_path = os.path.join(self.root_path, stats_filename)
            
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                self.manual_mean = stats['mean']
                self.manual_std = stats['std']
                print(f"Using existing normalization stats from: {stats_path}")
            else:
                print(f"No existing stats found at {stats_path}")
                if self.flag == 'train':
                    print("Computing new normalization stats from training data...")
                    self.manual_mean = np.mean(data_numeric, axis=0, keepdims=True, dtype=np.float32)
                    self.manual_std = np.std(data_numeric, axis=0, keepdims=True, dtype=np.float32)
                    np.savez(stats_path, mean=self.manual_mean, std=self.manual_std)
                    print(f"Saved new normalization stats to: {stats_path}")
                else:
                    raise FileNotFoundError(f"Normalization stats file not found at {stats_path}. Please run training first.")
            
            print(f"Normalization stats - Mean shape: {self.manual_mean.shape}, Std shape: {self.manual_std.shape}")
            data_numeric = ((data_numeric - self.manual_mean) / (self.manual_std + 1e-8)).astype(np.float32)

        self.data_x = data_numeric
        self.data_y = data_numeric

        self.n_var = self.data_x.shape[-1]
        total_len = len(self.data_x)
        max_start = total_len - self.seq_len - self.output_token_len + 1
        self.n_timepoint = (max_start + self.stride - 1) // self.stride

        print(f"Final data shape: {self.data_x.shape}")
        print(f"Stride: {self.stride}")
        print(f"Number of timepoints (with stride): {self.n_timepoint}")  

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[s_begin:s_end]

        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        if hasattr(self, 'labels') and self.labels is not None:
            self.current_sequence_label = self.labels[s_begin:s_end].max()

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return self.n_timepoint

    def inverse_transform(self, data):
        if hasattr(self, 'manual_mean') and hasattr(self, 'manual_std'):
            return data * self.manual_std + self.manual_mean
        else:
            return data

    def get_labels(self):
        return self.labels

    def get_feature_names(self):
        return self.numeric_columns

    @property
    def feature_names(self):
        return self.get_feature_names()
