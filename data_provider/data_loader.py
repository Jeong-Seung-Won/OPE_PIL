import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class KAFASATAnomalyDataset(Dataset):
    """
    KAFASAT 파일 기반 이상 데이터셋 클래스
    - train/test CSV 파일 경로를 직접 지정하여 사용
    - 정규화 통계 파일은 root_path 하위에 저장/로드
    - label은 test 데이터셋에서만 사용
    """
    def __init__(self, root_path, flag='train', size=None, data_path='kompsat3a_train.csv',
                 scale=True, nonautoregressive=False, test_flag='T',
                 subset_rand_ratio=1.0, use_full_data=True, target_feature=None, ddp=False, stride = None):
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
        self.ddp = ddp

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

        # 학습시 사용한 컬럼 고정 파일
        import json
        columns_path = os.path.join(self.root_path, 'kompsat3a_feature_columns.json')

        selected_columns = list(numeric_columns)
        if self.flag == 'train':
            # 학습 시점의 컬럼을 저장 (테스트/검증에서 동일 순서/구성 사용)
            try:
                with open(columns_path, 'w') as f:
                    json.dump(selected_columns, f)
                print(f"Saved training feature columns to: {columns_path}")
            except Exception as e:
                print(f"[WARN] Failed to save feature columns: {e}")
        else:
            # 테스트/검증에서는 학습 컬럼을 불러와 동일한 순서/구성으로 정렬
            if os.path.exists(columns_path):
                try:
                    with open(columns_path, 'r') as f:
                        train_columns = json.load(f)
                    print(f"Loaded training feature columns from: {columns_path}")
                    # 누락 컬럼은 0으로 채워 추가, 불필요한 컬럼은 제거
                    for col in train_columns:
                        if col not in df_raw.columns:
                            df_raw[col] = 0.0
                    selected_columns = train_columns
                except Exception as e:
                    print(f"[WARN] Failed to load feature columns: {e}. Using auto-detected columns.")
            else:
                print(f"[WARN] Training feature columns not found at {columns_path}. Using auto-detected columns.")

        # 선택된 컬럼 순서대로 데이터 구성
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
            # 기존 정규화 통계 파일을 재사용하여 일관성 보장
            stats_filename = "kompsat3a_normalization_stats.npz"
            stats_path = os.path.join(self.root_path, stats_filename)
            
            # 기존 파일이 있는지 먼저 확인
            if os.path.exists(stats_path):
                # 기존 통계 로드
                stats = np.load(stats_path)
                self.manual_mean = stats['mean']
                self.manual_std = stats['std']
                print(f"Using existing normalization stats from: {stats_path}")
            else:
                # 파일이 없는 경우에만 새로 계산 (첫 실행시)
                print(f"No existing stats found at {stats_path}")
                if self.flag == 'train':
                    print("Computing new normalization stats from training data...")
                    self.manual_mean = np.mean(data_numeric, axis=0, keepdims=True, dtype=np.float32)
                    self.manual_std = np.std(data_numeric, axis=0, keepdims=True, dtype=np.float32)
                    
                    # 새로 계산한 통계 저장
                    if self.ddp:
                        try:
                            import torch.distributed as dist
                            if dist.is_initialized() and dist.get_rank() == 0:
                                np.savez(stats_path, mean=self.manual_mean, std=self.manual_std)
                                print(f"Saved new normalization stats to: {stats_path}")
                            if dist.is_initialized():
                                dist.barrier()
                        except ImportError:
                            print("Warning: torch.distributed not available, saving anyway")
                            np.savez(stats_path, mean=self.manual_mean, std=self.manual_std)
                            print(f"Saved new normalization stats to: {stats_path}")
                    else:
                        np.savez(stats_path, mean=self.manual_mean, std=self.manual_std)
                        print(f"Saved new normalization stats to: {stats_path}")
                else:
                    # test/val인데 파일이 없으면 에러
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
        print(f"Stride: {self.stride}")  # ← 추가!
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
