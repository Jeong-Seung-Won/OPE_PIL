import os
import time
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')


# -------------------------
# Anomaly utils
# -------------------------

class Exp_AD(Exp_Basic):
    def __init__(self, args):
        super(Exp_AD, self).__init__(args)
        
    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            # for methods that do not use ddp (e.g. finetuning-based LLM4TS models)
            self.device = self.args.gpu
        
        model = self.model_dict[self.args.model].Model(self.args)
        
        if self.args.ddp:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
            
        if self.args.adaptation:
            print(f"Loading pretrained model from: {self.args.pretrain_model_path}")
            
            # 파일 확장자에 따라 로딩 방법 선택
            if self.args.pretrain_model_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    pretrained_dict = load_file(self.args.pretrain_model_path)
                    print("Loaded .safetensors file successfully")
                except ImportError:
                    raise ImportError("safetensors library not found. Install with: pip install safetensors")
            else:
                # .pth, .pt 등의 파일
                pretrained_dict = torch.load(self.args.pretrain_model_path, map_location='cpu')
                print("Loaded .pth/.pt file successfully")
            
            model.load_state_dict(pretrained_dict, strict=False)
            print("Pretrained model loaded with strict=False (ignoring mismatched keys)")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_feature_names(self):
        """데이터셋에서 feature 이름들을 가져오기"""
        # data_factory에서 설정된 feature_names 사용
        if hasattr(self.args, 'feature_names'):
            return self.args.feature_names
        else:
            # args에 없다면 직접 train 데이터에서 가져오기
            train_data, _ = self._get_data(flag='train')
            if hasattr(train_data, 'feature_names'):
                return train_data.feature_names
            elif hasattr(train_data, 'get_feature_names'):
                return train_data.get_feature_names()
            else:
                # 기본값으로 feature 수만큼 생성
                n_features = getattr(self.args, 'enc_in', 6)
                return [f'feature_{i}' for i in range(n_features)]

    def _compute_reconstruction_errors(self, data_loader):
        """Compute reconstruction errors for threshold-based anomaly detection"""
        self.model.eval()
        all_errors = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                reconstructed = self.model(batch_x, batch_x_mark, batch_y_mark)
                reconstruction_errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
                all_errors.extend(reconstruction_errors.cpu().numpy())
        
        return np.array(all_errors)

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim
    
    def _validate(self, val_loader, criterion):
        """Validation 수행"""
        self.model.eval()
        total_val_loss = 0
        val_steps = len(val_loader)
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Reconstruction
                reconstructed = self.model(batch_x, batch_x_mark, batch_y_mark)
                val_loss = criterion(reconstructed, batch_x)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / val_steps
        return avg_val_loss

    def _select_criterion(self):
        """다양한 loss function 선택"""
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
            print("📊 Using MSE Loss")
        elif self.args.loss == 'MAE' or self.args.loss == 'L1':
            criterion = nn.L1Loss()
            print("📊 Using MAE/L1 Loss")
        elif self.args.loss == 'Huber' or self.args.loss == 'SmoothL1':
            criterion = nn.SmoothL1Loss(beta=1.0)
            print("📊 Using Huber/SmoothL1 Loss")
        elif self.args.loss == 'LogCosh':
            # Log-Cosh Loss (custom implementation)
            def log_cosh_loss(y_pred, y_true):
                def _log_cosh(x):
                    return x + torch.nn.functional.softplus(-2.0 * x) - torch.log(torch.tensor(2.0))
                return torch.mean(_log_cosh(y_pred - y_true))
            criterion = log_cosh_loss
            print("📊 Using Log-Cosh Loss")
        elif self.args.loss == 'Combined':
            # MSE + MAE 조합
            def combined_loss(y_pred, y_true):
                mse = nn.MSELoss()(y_pred, y_true)
                mae = nn.L1Loss()(y_pred, y_true)
                return 0.7 * mse + 0.3 * mae
            criterion = combined_loss
            print("📊 Using Combined Loss (0.7*MSE + 0.3*MAE)")
        elif self.args.loss == 'Physics':
            # Physics-Informed Loss for Satellite Orbit Anomaly Detection
            from utils.physics_loss import PhysicsAwareCriterion
            criterion = PhysicsAwareCriterion(
                lambda_recon=getattr(self.args, 'physics_lambda_recon', 1.0),
                lambda_smooth=getattr(self.args, 'physics_lambda_smooth', 0.1),
                lambda_angular=getattr(self.args, 'physics_lambda_angular', 0.1),
                lambda_bound=getattr(self.args, 'physics_lambda_bound', 0.01)
            )
            print("📊 Using Physics-Informed Loss")
            print(f"   λ_recon={self.args.physics_lambda_recon}, λ_smooth={self.args.physics_lambda_smooth}")
            print(f"   λ_angular={self.args.physics_lambda_angular}, λ_bound={self.args.physics_lambda_bound}")
        else:
            # 기본값: MSE
            criterion = nn.MSELoss()
            print(f"⚠️ Unknown loss '{self.args.loss}', using MSE as default")
        
        return criterion

    def _save_checkpoint(self, path, epoch):
        """매 epoch마다 checkpoint 저장"""
        if self.args.dp or self.args.ddp:
            model = self.model.module
        else:
            model = self.model
            
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        
        # epoch 정보를 포함한 checkpoint 저장
        # checkpoint_path = os.path.join(path, f'checkpoint_epoch_{epoch+1}.pth')
        # torch.save(state_dict, checkpoint_path)
        
        # 최신 checkpoint도 저장 (기존 코드와 호환성)
        latest_path = os.path.join(path, 'checkpoint.pth')
        torch.save(state_dict, latest_path)


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        
        # 단변량/다변량 모드 확인
        if hasattr(self.args, 'target_feature') and self.args.target_feature is not None:
            feature_names = self._get_feature_names()
            target_feature = self.args.target_feature
            
            if 0 <= target_feature < len(feature_names):
                feature_name = feature_names[target_feature]
            else:
                feature_name = f'feature_{target_feature}'
            
            setting = setting + f'_{feature_name}'
            print(f"Univariate training with feature: {feature_name.upper()}")
        else:
            feature_names = self._get_feature_names()
            print(f"Multivariate training with all {len(feature_names)} features ({', '.join(feature_names)})")
        
        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()
        training_start_time = time.time()  # 전체 학습 시작 시간 기록

        train_steps = len(train_loader)
        
        # Early stopping 선택적 사용
        if self.args.use_early_stop:
            val_data, val_loader = self._get_data(flag='val')
            val_steps = len(val_loader)
            early_stopping = EarlyStopping(self.args, verbose=True)
            print("🔄 Early stopping enabled with patience:", self.args.patience)
        else:
            print("📝 Early stopping disabled - saving checkpoints every epoch")
        
        model_optim = self._select_optimizer()
        
        # 스케줄러 조건부 생성
        scheduler = None
        if self.args.no_scheduler:
            print("📈 No scheduler - using fixed learning rate")
        elif self.args.cosine:
            effective_tmax = min(self.args.tmax, self.args.train_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=effective_tmax, eta_min=1e-6)
            print(f"📈 Using Cosine Annealing LR Scheduler (T_max={effective_tmax}, eta_min=1e-6)")
        else:
            print("📈 Using adjust_learning_rate scheduler")
            
        criterion = self._select_criterion()
        
        # CSV 로깅 설정
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            # CSV 파일명 생성
            samples_info = ""
            if hasattr(self.args, 'samples_per_file') and self.args.samples_per_file:
                samples_info = f"_samples{self.args.samples_per_file}perfile"
            
            test_dataset_name = self.args.data
            if hasattr(self.args, 'test_data_path') and self.args.test_data_path:
                test_dataset_name = os.path.splitext(os.path.basename(self.args.test_data_path))[0]
            
            csv_filename = f"training_log_{test_dataset_name}"
            if hasattr(self.args, 'target_feature') and self.args.target_feature is not None:
                csv_filename += f"_feat{self.args.target_feature}"
            else:
                csv_filename += "_multivariate"
            csv_filename += samples_info
            csv_filename += f"_lr{self.args.learning_rate}_ep{self.args.train_epochs}"
            
            if hasattr(self.args, 'cosine') and self.args.cosine:
                csv_filename += "_cosine"
            elif hasattr(self.args, 'lradj') and self.args.lradj:
                csv_filename += f"_{self.args.lradj}"
            
            csv_filename += ".csv"
            
            # CSV 파일 경로 설정
            model_dir = f"results/{self.args.model}"
            training_logs_dir = os.path.join(model_dir, "training_logs")
            os.makedirs(training_logs_dir, exist_ok=True)
            csv_filepath = os.path.join(training_logs_dir, csv_filename)
            
            # CSV 파일 초기화 (헤더 작성)
            with open(csv_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['epoch', 'avg_loss', 'epoch_time', 'cumulative_time', 'learning_rate'])
            
            print(f"📊 Training log will be saved to: {csv_filepath}")
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            epoch_losses = []  # epoch별 loss 저장용
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Reconstruction mode: 입력 시퀀스를 재구성
                # Denoising autoencoder 옵션: 입력에 약한 노이즈를 추가하여 평균 수렴 방지
                denoise_sigma = float(getattr(self.args, 'denoise_sigma', 0.0))
                if denoise_sigma > 0:
                    noise = torch.randn_like(batch_x) * denoise_sigma
                    batch_x_in = batch_x + noise
                else:
                    batch_x_in = batch_x
                reconstructed = self.model(batch_x_in, batch_x_mark, batch_y_mark)
                if self.args.dp:
                    torch.cuda.synchronize()
                
                # Reconstruction loss: base + temporal/frequency regularizers
                # batch_x: [batch_size, seq_len, n_features]
                # reconstructed: [batch_size, seq_len, n_features]
                loss = criterion(reconstructed, batch_x)

                # 추가 정규화 항목들 (시간 기울기, 주파수 스펙트럼, 분산 유지)
                lambda_time = float(getattr(self.args, 'lambda_time_grad', 0.0))
                if lambda_time > 0:
                    dx_true = batch_x[:, 1:, :] - batch_x[:, :-1, :]
                    dx_pred = reconstructed[:, 1:, :] - reconstructed[:, :-1, :]
                    loss_time = torch.nn.functional.l1_loss(dx_pred, dx_true)
                    loss = loss + lambda_time * loss_time

                lambda_freq = float(getattr(self.args, 'lambda_freq', 0.0))
                if lambda_freq > 0:
                    # rfft 기반 스펙트럼 크기 비교 (시간축 dim=1)
                    pred_fft = torch.fft.rfft(reconstructed, dim=1)
                    true_fft = torch.fft.rfft(batch_x, dim=1)
                    spec_loss = torch.nn.functional.l1_loss(torch.abs(pred_fft), torch.abs(true_fft))
                    loss = loss + lambda_freq * spec_loss

                lambda_var = float(getattr(self.args, 'lambda_var', 0.0))
                if lambda_var > 0:
                    # 시점 축 분산 유사성 유지
                    var_true = torch.var(batch_x, dim=1, unbiased=False)
                    var_pred = torch.var(reconstructed, dim=1, unbiased=False)
                    var_loss = torch.nn.functional.l1_loss(var_pred, var_true)
                    loss = loss + lambda_var * var_loss
                
                # loss 저장
                epoch_losses.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                loss.backward()
                # Gradient clipping (폭주 방지 및 안정화)
                max_grad_norm = float(getattr(self.args, 'max_grad_norm', 0.0))
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                model_optim.step()

            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                epoch_elapsed = time.time() - epoch_time
                total_elapsed = time.time() - training_start_time
                
                # epoch 평균 loss 계산
                avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                
                # 현재 learning rate 가져오기
                current_lr = model_optim.param_groups[0]['lr']
                
                # Early stopping 사용 시 validation 수행
                if self.args.use_early_stop:
                    val_loss = self._validate(val_loader, criterion)
                    print("Epoch: {} | Train Loss: {:.7f} | Val Loss: {:.7f} | Time: {:.2f}s | Total: {:.2f}s | LR: {:.2e}".format(
                        epoch + 1, avg_epoch_loss, val_loss, epoch_elapsed, total_elapsed, current_lr))
                    
                    # CSV에 기록 (validation loss 포함)
                    with open(csv_filepath, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([epoch + 1, avg_epoch_loss, val_loss, epoch_elapsed, total_elapsed, current_lr])
                    
                    # Early stopping 체크
                    early_stopping(val_loss, self.model, path)
                    if early_stopping.early_stop:
                        print("🛑 Early stopping triggered!")
                        # DDP에서 early stopping 신호를 모든 rank에 전파
                        if self.args.ddp:
                            early_stop_tensor = torch.tensor(1, device=self.device)
                            dist.broadcast(early_stop_tensor, src=0)
                            if early_stop_tensor.item() == 1:
                                break
                        else:
                            break
                else:
                    print("Epoch: {} | Avg Loss: {:.7f} | Time: {:.2f}s | Total: {:.2f}s | LR: {:.2e}".format(
                        epoch + 1, avg_epoch_loss, epoch_elapsed, total_elapsed, current_lr))
                    
                    # CSV에 기록 (기본)
                    with open(csv_filepath, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([epoch + 1, avg_epoch_loss, epoch_elapsed, total_elapsed, current_lr])

            # checkpoint 저장 (Early stopping 사용 여부에 따라)
            if not self.args.use_early_stop:
                # Early stopping 사용하지 않을 때만 매 epoch마다 저장
                if self.args.ddp:
                    if self.args.local_rank == 0:
                        self._save_checkpoint(path, epoch)
                    dist.barrier()
                else:
                    self._save_checkpoint(path, epoch)
            
            # Learning rate 조정
            if self.args.no_scheduler:
                # 스케줄러 없음 - 고정 학습률 유지
                pass
            elif self.args.cosine and scheduler is not None:
                scheduler.step()
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.ddp:
                train_loader.sampler.set_epoch(epoch + 1)
        
        # 전체 학습 완료 후 총 학습 시간 출력 및 저장
        total_training_time = time.time() - training_start_time
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print("총 학습 시간: {:.2f}초".format(total_training_time))
            
            # 훈련 정보 저장
            samples_info = ""
            if hasattr(self.args, 'samples_per_file') and self.args.samples_per_file:
                samples_info = f"_samples{self.args.samples_per_file}perfile"
            
            # 테스트 데이터 파일명 추출
            test_dataset_name = self.args.data
            if hasattr(self.args, 'test_data_path') and self.args.test_data_path:
                test_dataset_name = os.path.splitext(os.path.basename(self.args.test_data_path))[0]
            
            # TimesFM과 동일한 폴더 구조 생성
            model_dir = f"results/{self.args.model}"
            results_dir = os.path.join(model_dir, "results")
            training_logs_dir = os.path.join(model_dir, "training_logs")
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(training_logs_dir, exist_ok=True)
            
            training_log_filename = f"training_log_{test_dataset_name}"
            if hasattr(self.args, 'target_feature') and self.args.target_feature is not None:
                training_log_filename += f"_feat{self.args.target_feature}"
            else:
                training_log_filename += "_multivariate"
            training_log_filename += samples_info
            training_log_filename += f"_lr{self.args.learning_rate}_ep{self.args.train_epochs}"
            
            # 스케줄러 정보 추가
            if hasattr(self.args, 'cosine') and self.args.cosine:
                training_log_filename += "_cosine"
            elif hasattr(self.args, 'lradj') and self.args.lradj:
                training_log_filename += f"_{self.args.lradj}"
            
            training_log_filename += ".txt"
            training_log_filename = os.path.join(training_logs_dir, training_log_filename)
            
            with open(training_log_filename, 'a') as f:
                f.write(f"Training Time: {total_training_time:.3f}s ({total_training_time/60:.3f}min)")
                if hasattr(self.args, 'samples_per_file') and self.args.samples_per_file:
                    f.write(f" | Samples/File: {self.args.samples_per_file}")
                if hasattr(self.args, 'target_feature') and self.args.target_feature is not None:
                    f.write(f" | Feature: {self.args.target_feature}")
                else:
                    f.write(f" | Mode: Multivariate (all 6 features)")
                f.write(f" | Epochs: {self.args.train_epochs} | LR: {self.args.learning_rate}\n")
            
        # Early stopping 사용 시 best model 로딩, 아니면 마지막 epoch model 사용
        if self.args.use_early_stop:
            best_model_path = path + '/' + 'checkpoint.pth'
            print(f"🔄 Loading best model from: {best_model_path}")
            
            try:
                # 간단한 checkpoint 로딩 (모든 프로세스가 개별적으로 로딩)
                print(f"📥 Rank {getattr(self.args, 'local_rank', 0)}: Loading checkpoint...")
                checkpoint_state = torch.load(best_model_path, map_location='cpu')
                
                # 모델에 checkpoint 로딩
                self.model.load_state_dict(checkpoint_state, strict=False)
                print(f"✅ Rank {getattr(self.args, 'local_rank', 0)}: Best model loaded successfully!")
                
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                print("⚠️ Continuing with current model state...")
        else:
            print("📝 Using last epoch model (no early stopping)")
        
        # ==========================
        # After training: Point-wise Latent → L1 normalize → KMeans → save centroids & train dist
        # ==========================
        # 기존의 KMeans 관련 코드 블록을 이것으로 교체:
        try:
            anomaly_dir = os.path.join(self.args.checkpoints, setting, 'anomaly')
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                os.makedirs(anomaly_dir, exist_ok=True)

            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print('Computing reconstruction errors on training data for threshold...')
            
            # 새로 추가할 함수 호출
            train_errors = self._compute_reconstruction_errors(train_loader)
            
            train_errors_path = os.path.join(anomaly_dir, 'train_reconstruction_errors.npy')
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                np.save(train_errors_path, train_errors)
                print(f'Training reconstruction errors saved: {train_errors_path}')
        except Exception as e:
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print(f'[WARN] Training error computation failed: {e}')

        return self.model


    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        # 단변량/다변량 모드 확인
        if hasattr(self.args, 'target_feature') and self.args.target_feature is not None:
            feature_names = self._get_feature_names()
            target_feature = self.args.target_feature
            
            if 0 <= target_feature < len(feature_names):
                feature_name = feature_names[target_feature]
            else:
                feature_name = f'feature_{target_feature}'
            
            setting = setting + f'_{feature_name}'
            print(f"Univariate testing with feature: {feature_name.upper()}")
        else:
            feature_names = self._get_feature_names()
            print(f"Multivariate testing with all {len(feature_names)} features ({', '.join(feature_names)})")

        print("info:", self.args.test_seq_len, self.args.input_token_len, self.args.output_token_len, self.args.test_pred_len)
        preds = []
        trues = []
        time_now = time.time()
        inference_start_time = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        
        # Load training reconstruction errors for threshold computation
        anomaly_dir = os.path.join(self.args.checkpoints, setting, 'anomaly')
        train_errors_path = os.path.join(anomaly_dir, 'train_reconstruction_errors.npy')
        
        train_errors = None
        if os.path.exists(train_errors_path):
            train_errors = np.load(train_errors_path)
            print(f'Loaded training reconstruction errors from: {train_errors_path}')
            print(f'Training error statistics: mean={np.mean(train_errors):.6f}, std={np.std(train_errors):.6f}')
        else:
            print(f'Warning: Training reconstruction errors not found at {train_errors_path}')
        
        # Stride 정보 가져오기 및 디버깅 정보 출력
        stride = getattr(test_data, 'stride', 1)
        print(f"\n=== Anomaly Detection Debug Info ===")
        print(f"Test dataset size: {len(test_data)}")
        if hasattr(test_data, 'labels') and test_data.labels is not None:
            print(f"Test data.labels shape: {test_data.labels.shape}")
            unique, counts = np.unique(test_data.labels, return_counts=True)
            print(f"Label distribution: {dict(zip(unique, counts))}")
        else:
            print(f"Test data.labels: None")
        print(f"Test stride: {stride}")
        print(f"Number of batches: {len(test_loader)}")
        print(f"Batch size: {test_loader.batch_size}")
        print(f"===================================\n")

        with torch.no_grad():
            # tqdm으로 진행률 표시 (DDP일 때는 rank 0에서만 표시)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                test_loader_with_progress = tqdm(test_loader, desc="Testing", unit="batch")
            else:
                test_loader_with_progress = test_loader

            # reconstruction error-based anomaly scoring 준비
            recon_error_scores = []    # 시점별 재구성 에러 기반 점수
            point_labels = []  # 시점별 라벨 (point anomaly용)
            # ========== 디버깅 카운터 추가 ==========
            debug_total_windows = 0
            debug_batch_count = 0
            # =======================================

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader_with_progress):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Reconstruction mode: 전체 시퀀스를 재구성
                reconstructed = self.model(batch_x, batch_x_mark, batch_y_mark)
                # 원본과 재구성 결과 비교
                outputs = reconstructed.detach().cpu()
                batch_x_cpu = batch_x.detach().cpu()
                pred = outputs  # 재구성된 시퀀스
                true = batch_x_cpu  # 원본 시퀀스

                preds.append(pred)
                trues.append(true)  # 원본 시퀀스 (ground truth)

                # 공통 길이 계산
                bs = batch_x_cpu.shape[0]
                seq_len = batch_x_cpu.shape[1]

                # 재구성 에러 기반 점수(각 시점 평균 |x - x_hat|) 계산
                per_t_mae = torch.mean(torch.abs(outputs - batch_x_cpu), dim=2)  # [B, T]

                # ========== 디버깅 출력 추가 (첫 3 batch만) ==========
                if i < 3:
                    print(f"\n=== Batch {i} Debug ===")
                    print(f"  batch_x_cpu.shape: {batch_x_cpu.shape}")
                    print(f"  outputs.shape: {outputs.shape}")
                    print(f"  per_t_mae.shape: {per_t_mae.shape}")
                    print(f"  bs: {bs}, seq_len: {seq_len}")
                    print(f"  test_loader.batch_size: {test_loader.batch_size}")
                # ===================================================

                # ========== 수정된 Label 매칭 로직 ==========
                # 각 sample별로 처리
                for b in range(bs):
                    # 현재 window의 실제 시작 인덱스 계산
                    window_start = (i * test_loader.batch_size + b) * stride
                    window_end = window_start + seq_len
                    # ========== 디버깅 출력 추가 (첫 batch의 첫 3 samples만) ==========
                    if i == 0 and b < 3:
                        print(f"\n  Sample {b}:")
                        print(f"    window_start: {window_start}")
                        print(f"    window_end: {window_end}")
                        print(f"    per_t_mae[{b}, :].shape: {per_t_mae[b, :].shape}")
                        print(f"    per_t_mae[{b}, :] length: {len(per_t_mae[b, :].numpy())}")
                    # ================================================================
                    
                    # Error 저장
                    before_len = len(recon_error_scores)
                    recon_error_scores.extend(per_t_mae[b, :].numpy())
                    after_len = len(recon_error_scores)
                    
                    # ========== 디버깅: 추가된 개수 확인 ==========
                    if i == 0 and b < 3:
                        print(f"    Added {after_len - before_len} points")
                    # ===========================================
                    # Error 저장
                    #recon_error_scores.extend(per_t_mae[b, :].numpy())
                    
                    # Label 저장 (stride 반영)
                    if hasattr(test_data, 'labels') and test_data.labels is not None:
                        if window_end <= len(test_data.labels):
                            # 정상 범위
                            point_labels.extend(test_data.labels[window_start:window_end])
                        else:
                            # 범위 벗어나면 가능한 만큼만
                            available = test_data.labels[window_start:min(window_end, len(test_data.labels))]
                            point_labels.extend(available)
                            # 나머지는 0으로 패딩
                            remaining = seq_len - len(available)
                            if remaining > 0:
                                point_labels.extend([0] * remaining)
                # ========== 수정 끝 ==========
                
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                # ========== 디버깅 카운터 업데이트 ==========
                debug_total_windows += bs
                debug_batch_count += 1
                # =========================================
            # ========== 최종 디버깅 출력 ==========
            print(f"\n=== Final Debug Info ===")
            print(f"Total batches processed: {debug_batch_count}")
            print(f"Total windows processed: {debug_total_windows}")
            print(f"Expected points (windows × seq_len): {debug_total_windows * seq_len}")
            print(f"Actual points collected: {len(recon_error_scores)}")
            print(f"Ratio: {len(recon_error_scores) / (debug_total_windows * seq_len) if debug_total_windows > 0 else 0:.2f}x")
            # =======================================
        # inference 완료 후 총 시간 계산
        total_inference_time = time.time() - inference_start_time
        print("총 Inference 시간: {:.2f}초".format(total_inference_time))
        
        # 수집 결과 확인
        print(f"\nCollection Results:")
        print(f"  Reconstruction errors collected: {len(recon_error_scores)}")
        print(f"  Labels collected: {len(point_labels)}")
        if len(point_labels) > 0:
            unique_labels, label_counts = np.unique(point_labels, return_counts=True)
            print(f"  Collected label distribution: {dict(zip(unique_labels, label_counts))}")
        
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()

        # Reconstruction mode에서는 covariate 처리 불필요 (전체 feature 재구성)
        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
        print(f'MSE: {mse:.6f}, MAE: {mae:.6f}')

        anomaly_eval = {}
        anomaly_eval['num_points'] = len(recon_error_scores)
        anomaly_eval['num_labels'] = len(point_labels)

        # Threshold-based anomaly detection using reconstruction errors
        auroc = None
        if len(point_labels) > 0 and len(point_labels) == len(recon_error_scores):
            try:
                # Use reconstruction errors directly as anomaly scores (higher error = more anomalous)
                anomaly_scores = np.array(recon_error_scores)
                labels_raw = np.array(point_labels)
                
                # Convert multiclass labels to binary (0 = normal, 1/2/3/... = anomaly → 1)
                labels = (labels_raw > 0).astype(int)
                
                print(f"Label conversion: multiclass {np.unique(labels_raw)} → binary {np.unique(labels)}")
                print(f"  Normal (0): {np.sum(labels == 0)}, Anomaly (>0): {np.sum(labels == 1)}")
                
                auroc = roc_auc_score(labels, anomaly_scores)
                print(f'AUROC (point anomaly, reconstruction error): {auroc:.6f}')
                print(f'  - Points: {len(anomaly_scores)}, Labels: {len(labels)}')
                
                # Print reconstruction error statistics
                print(f'Test reconstruction error statistics:')
                print(f'  Mean: {np.mean(anomaly_scores):.6f}, Std: {np.std(anomaly_scores):.6f}')
                print(f'  Min: {np.min(anomaly_scores):.6f}, Max: {np.max(anomaly_scores):.6f}')
                print(f'  95th percentile: {np.percentile(anomaly_scores, 95):.6f}')
                print(f'  99th percentile: {np.percentile(anomaly_scores, 99):.6f}')
                
            except Exception as e:
                print(f'[WARN] AUROC computation failed: {e}')
        elif len(point_labels) > 0:
            print(f'[WARN] Length mismatch: {len(recon_error_scores)} scores vs {len(point_labels)} labels')
        anomaly_eval['auroc'] = float(auroc) if auroc is not None else None

        # ========== F1 최적 임계값 및 클래스별 지표 계산 ==========
        try:
            if len(point_labels) > 0 and len(point_labels) == len(recon_error_scores):
                labels_raw = np.array(point_labels).astype(int)
                y_score = np.array(recon_error_scores).astype(float)
                
                # Convert multiclass labels to binary (0 = normal, 1/2/3/... = anomaly → 1)
                y_true = (labels_raw > 0).astype(int)
                
                pos_count = int(np.sum(y_true))
                neg_count = int(len(y_true) - pos_count)

                if pos_count > 0 and neg_count > 0:
                    from sklearn.metrics import precision_recall_curve
                    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_true, y_score)

                    if len(thresholds_arr) > 1000:
                        step = len(thresholds_arr) // 1000
                        thresholds_arr = thresholds_arr[::step]
                        print(f"Threshold count reduced from original to {len(thresholds_arr)} for faster computation")
                        
                    def point_adjust(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> np.ndarray:
                        y_pred_pa = y_pred_arr.copy()
                        in_seg = False
                        seg_start = 0
                        n = len(y_true_arr)
                        for i in range(n + 1):
                            if i < n and y_true_arr[i] == 1 and (i == 0 or y_true_arr[i-1] == 0):
                                seg_start = i
                            if (i == n or y_true_arr[i] == 0) and (i > 0 and y_true_arr[i-1] == 1):
                                seg_end = i
                                if np.any(y_pred_arr[seg_start:seg_end] == 1):
                                    y_pred_pa[seg_start:seg_end] = 1
                        return y_pred_pa

                    best_f1 = -1.0
                    best_threshold = None
                    best_precision = 0.0
                    best_recall = 0.0
                    best_tp = best_fp = best_tn = best_fn = 0

                    eps = 1e-12
                    for thr in thresholds_arr:
                        y_pred = (y_score >= thr).astype(int)
                        y_pred_pa = point_adjust(y_true, y_pred)
                        tp = int(np.sum((y_pred_pa == 1) & (y_true == 1)))
                        fp = int(np.sum((y_pred_pa == 1) & (y_true == 0)))
                        tn = int(np.sum((y_pred_pa == 0) & (y_true == 0)))
                        fn = int(np.sum((y_pred_pa == 0) & (y_true == 1)))
                        precision = float(tp / (tp + fp + eps))
                        recall = float(tp / (tp + fn + eps))
                        f1_pa = float(2 * precision * recall / (precision + recall + eps))
                        if f1_pa > best_f1:
                            best_f1 = f1_pa
                            best_threshold = float(thr)
                            best_precision = precision
                            best_recall = recall
                            best_tp, best_fp, best_tn, best_fn = tp, fp, tn, fn

                    if best_threshold is not None and best_f1 >= 0:
                        anomaly_eval.update({
                            'best_threshold': best_threshold,
                            'best_f1': float(best_f1),
                            'best_precision': float(best_precision),
                            'best_recall': float(best_recall),
                            'tp': int(best_tp),
                            'fp': int(best_fp),
                            'tn': int(best_tn),
                            'fn': int(best_fn),
                            'support_pos': pos_count,
                            'support_neg': neg_count
                        })

                        print(f"Best PA-F1: {best_f1:.6f} @ threshold={best_threshold:.6f} (P={pos_count}, N={neg_count})")
                        
                        # ========== Per-type Anomaly Evaluation ==========
                        unique_types = np.unique(labels_raw)
                        print(f"DEBUG unique_types: {unique_types}, len: {len(unique_types)}")
                        if len(unique_types) > 2:
                            type_names = {0: 'normal', 1: 'along_track', 2: 'in_plane', 3: 'out_plane'}
                            per_type_results = {}
                            
                            for atype in sorted(unique_types):
                                if atype == 0:
                                    continue
                                
                                tname = type_names.get(int(atype), f'type_{int(atype)}')
                                y_true_type = (labels_raw == atype).astype(int)
                                type_pos = int(y_true_type.sum())
                                type_neg = int(len(y_true_type) - type_pos)
                                
                                if type_pos == 0:
                                    continue
                                
                                try:
                                    type_auroc = float(roc_auc_score(y_true_type, y_score))
                                except:
                                    type_auroc = None
                                
                                type_best_f1 = -1.0
                                type_best_thr = 0.0
                                type_best_p = type_best_r = 0.0
                                type_tp = type_fp = type_tn = type_fn = 0
                                
                                for thr in thresholds_arr:
                                    y_pred_t = (y_score >= thr).astype(int)
                                    y_pred_pa_t = point_adjust(y_true_type, y_pred_t)
                                    tp_t = int(((y_pred_pa_t == 1) & (y_true_type == 1)).sum())
                                    fp_t = int(((y_pred_pa_t == 1) & (y_true_type == 0)).sum())
                                    tn_t = int(((y_pred_pa_t == 0) & (y_true_type == 0)).sum())
                                    fn_t = int(((y_pred_pa_t == 0) & (y_true_type == 1)).sum())
                                    p_t = tp_t / (tp_t + fp_t + eps)
                                    r_t = tp_t / (tp_t + fn_t + eps)
                                    f1_t = 2 * p_t * r_t / (p_t + r_t + eps)
                                    if f1_t > type_best_f1:
                                        type_best_f1 = f1_t
                                        type_best_thr = float(thr)
                                        type_best_p = p_t
                                        type_best_r = r_t
                                        type_tp, type_fp = tp_t, fp_t
                                        type_tn, type_fn = tn_t, fn_t
                                
                                per_type_results[tname] = {
                                    'auroc': type_auroc,
                                    'best_f1': float(type_best_f1),
                                    'best_precision': float(type_best_p),
                                    'best_recall': float(type_best_r),
                                    'best_threshold': type_best_thr,
                                    'tp': type_tp, 'fp': type_fp,
                                    'tn': type_tn, 'fn': type_fn,
                                    'support_pos': type_pos,
                                    'support_neg': type_neg,
                                }
                                auroc_str = f"{type_auroc:.4f}" if type_auroc is not None else "N/A"
                                print(f"  [{tname}] PA-F1={type_best_f1:.4f} "
                                    f"AUROC={auroc_str} "
                                    f"P={type_best_p:.4f} R={type_best_r:.4f} "
                                    f"(pos={type_pos})")
                            
                            if per_type_results:
                                anomaly_eval['per_type'] = per_type_results
                                type_dist = {}
                                for atype in sorted(unique_types):
                                    tname = type_names.get(int(atype), f'type_{int(atype)}')
                                    type_dist[tname] = int((labels_raw == atype).sum())
                                anomaly_eval['label_distribution'] = type_dist
                                print(f"  Label distribution: {type_dist}")
                        # ========== Per-type Evaluation End ==========

                        # Suggest threshold based on training data statistics
                        if train_errors is not None:
                            # Common threshold strategies
                            mean_train = np.mean(train_errors)
                            std_train = np.std(train_errors)
                            percentile_95 = np.percentile(train_errors, 95)
                            percentile_99 = np.percentile(train_errors, 99)
                            
                            print(f"Suggested thresholds based on training data:")
                            print(f"  Mean + 2*Std: {mean_train + 2*std_train:.6f}")
                            print(f"  Mean + 3*Std: {mean_train + 3*std_train:.6f}")
                            print(f"  95th percentile: {percentile_95:.6f}")
                            print(f"  99th percentile: {percentile_99:.6f}")
                            print(f"  Optimal F1 threshold: {best_threshold:.6f}")
                    else:
                        print('[WARN] Could not compute PA-F1; skipping')
                else:
                    print('[WARN] Labels are single-class; skipping F1-based metrics')
        except Exception as e:
            print(f'[WARN] F1/threshold metrics computation failed: {e}')
        
        # 테스트 데이터 파일명 추출
        test_dataset_name = self.args.data
        if hasattr(self.args, 'test_data_path') and self.args.test_data_path:
            test_dataset_name = os.path.splitext(os.path.basename(self.args.test_data_path))[0]
        
        # TimesFM과 동일한 폴더 구조 생성
        model_dir = f"results/{self.args.model}"
        results_dir = os.path.join(model_dir, "results")
        plots_dir = os.path.join(model_dir, "plots")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # 단변량/다변량 모드 JSON 결과 저장
        import json
        from datetime import datetime
        
        # 실험 수행 시간 기록
        experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        feature_names = self._get_feature_names()
        
        # Loss function 정보 구성
        loss_info = {
            "loss_function": self.args.loss
        }
        if self.args.loss == 'Physics':
            loss_info.update({
                "lambda_recon": getattr(self.args, 'physics_lambda_recon', 1.0),
                "lambda_smooth": getattr(self.args, 'physics_lambda_smooth', 0.1),
                "lambda_angular": getattr(self.args, 'physics_lambda_angular', 0.1),
                "lambda_bound": getattr(self.args, 'physics_lambda_bound', 0.01)
            })
        
        results = {
            "experiment_timestamp": experiment_timestamp,
            "test_csv": test_dataset_name,
            "target_feature": getattr(self.args, 'target_feature', None),
            "feature_name": feature_names[self.args.target_feature] if hasattr(self.args, 'target_feature') and self.args.target_feature is not None and 0 <= self.args.target_feature < len(feature_names) else "multivariate",
            "total_samples": len(preds),
            "batch_size": getattr(self.args, 'batch_size', 32),
            "mse_total": float(mse),
            "mae_total": float(mae),
            "inference_time": float(total_inference_time),
            "throughput": float(len(preds)/total_inference_time if total_inference_time > 0 else 0),
            "model": self.args.model,
            "loss": loss_info,
            "learning_rate": float(self.args.learning_rate),
            "train_epochs": self.args.train_epochs,
            "samples_per_file": getattr(self.args, 'samples_per_file', None),
            "test_pred_len": self.args.test_pred_len,
            "anomaly": {
                "method": "reconstruction_error_threshold",
                "train_errors_path": train_errors_path if os.path.exists(train_errors_path) else None,
                "auroc": anomaly_eval.get('auroc', None),
                "num_points": anomaly_eval.get('num_points', 0),
                "num_labels": anomaly_eval.get('num_labels', 0),
                "evaluation_type": "point_anomaly_with_reconstruction_error",
                "best_threshold": anomaly_eval.get('best_threshold', None),
                "best_f1": anomaly_eval.get('best_f1', None),
                "best_precision": anomaly_eval.get('best_precision', None),
                "best_recall": anomaly_eval.get('best_recall', None),
                "tp": anomaly_eval.get('tp', None),
                "fp": anomaly_eval.get('fp', None),
                "tn": anomaly_eval.get('tn', None),
                "fn": anomaly_eval.get('fn', None),
                "support_pos": anomaly_eval.get('support_pos', None),
                "support_neg": anomaly_eval.get('support_neg', None),
                "per_type": anomaly_eval.get('per_type', None),
                "label_distribution": anomaly_eval.get('label_distribution', None)
            }
        }
        
        # Training statistics 추가 (available인 경우)
        if train_errors is not None:
            results["anomaly"]["train_error_stats"] = {
                "mean": float(np.mean(train_errors)),
                "std": float(np.std(train_errors)),
                "min": float(np.min(train_errors)),
                "max": float(np.max(train_errors)),
                "percentile_95": float(np.percentile(train_errors, 95)),
                "percentile_99": float(np.percentile(train_errors, 99))
            }
        
        # Test statistics 추가
        if len(recon_error_scores) > 0:
            results["anomaly"]["test_error_stats"] = {
                "mean": float(np.mean(recon_error_scores)),
                "std": float(np.std(recon_error_scores)),
                "min": float(np.min(recon_error_scores)),
                "max": float(np.max(recon_error_scores)),
                "percentile_95": float(np.percentile(recon_error_scores, 95)),
                "percentile_99": float(np.percentile(recon_error_scores, 99))
            }
        
        # JSON 파일명 생성 (단변량/다변량 모드)
        json_filename = f"results_{test_dataset_name}"

                # Hidden dimension 정보 추가
        if self.args.model == 'simple_autoencoder' or self.args.model == 'lstm_autoencoder':
            json_filename += f"_dmodel{self.args.d_model}"
        elif self.args.model == 'transformer_autoencoder':
            json_filename += f"_dmodel{self.args.d_model}"
        elif self.args.model == 'omni_anomaly':
            json_filename += f"_h{self.args.h_dim}_z{self.args.z_dim}"
        elif self.args.model == 'usad':
            json_filename += f"_hidden{self.args.hidden_dim}_latent{self.args.latent_dim}"
        elif self.args.model == 'tranad':
            json_filename += f"_dmodel{self.args.d_model}"
        elif self.args.model == 'dagmm':
            json_filename += f"_h{self.args.hidden_dim1}_{self.args.hidden_dim2}_{self.args.hidden_dim3}"
        elif self.args.model == 'lstmndt':
            json_filename += f"_lstmh{self.args.lstm_hidden}"
        elif self.args.model == 'anomalytransformer':
            json_filename += f"_dmodel{self.args.d_model}_dff{self.args.d_ff}"
        elif self.args.model == 'caem':
            json_filename += f"_ch{self.args.cae_m_channels_1}_{self.args.cae_m_channels_2}_{self.args.cae_m_channels_3}"
        elif self.args.model == 'mscred':
            json_filename += f"_clstm{'_'.join(map(str, self.args.mscred_conv_lstm_hidden))}"
        elif self.args.model == 'dtaad':
            json_filename += f"_dff{self.args.dtaad_d_ff}"
        elif self.args.model == 'memto':
            json_filename += f"_dmodel{self.args.memto_d_model}_dff{self.args.memto_d_ff}"
        elif self.args.model == 'npsr':
            json_filename += f"_dmodel{self.args.npsr_d_model}"
        elif self.args.model == 'sensitivehue':
            json_filename += f"_dmodel{self.args.sensitive_hue_d_model}_fc{self.args.sensitive_hue_dim_hidden_fc}"

        if hasattr(self.args, 'target_feature') and self.args.target_feature is not None:
            if 0 <= self.args.target_feature < len(feature_names):
                feature_name = feature_names[self.args.target_feature]
                json_filename += f"_{feature_name}"
            else:
                json_filename += f"_feat{self.args.target_feature}"
        else:
            json_filename += "_multivariate"

        # Loss function 정보 추가
        json_filename += f"_loss{self.args.loss}"
        if self.args.loss == 'Physics':
            json_filename += f"_s{self.args.physics_lambda_smooth}_a{self.args.physics_lambda_angular}"
        
        # Timestamp 추가
        json_filename += f"_{experiment_timestamp}"

        json_filename += ".json"
        json_filename = os.path.join(results_dir, json_filename)

        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON 결과 저장: {json_filename}")

        # Plot 저장은 rank 0에서만 수행
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print("시각화 및 plot 저장 시작...")
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse_list = []
        mae_list = []
        for i in range(len(trues)):
            y_true = trues[i]
            y_pred = preds[i]
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            mse_list.append(mse)
            mae_list.append(mae)
        sample_indices = list(range(len(preds))[-5:])

        # 각 샘플별로 별도의 그림 생성 (단변량/다변량 모드에 따라)
        for i, idx in enumerate(sample_indices):
            # 단변량/다변량 모드 확인
            is_univariate = hasattr(self.args, 'target_feature') and self.args.target_feature is not None
            
            if is_univariate:
                # 단변량 모드: 특정 feature만 표시
                feature_names = self._get_feature_names()
                target_feature = self.args.target_feature
                
                if 0 <= target_feature < len(feature_names):
                    feature_name = feature_names[target_feature]
                else:
                    feature_name = f'feature_{target_feature}'
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                # 단변량 데이터 추출 (target_feature에 해당하는 feature만)
                original_vals = trues[idx, :, target_feature]    # [seq_len] - 원본
                reconstructed_vals = preds[idx, :, target_feature]      # [seq_len] - 재구성
                
                # Reconstruction 시각화
                ax.plot(range(len(original_vals)), original_vals, label="Original", color="blue", linewidth=2)
                ax.plot(range(len(reconstructed_vals)), reconstructed_vals, label="Reconstructed", color="red", linewidth=2)
                
                ax.set_xlabel("Time Step")
                ax.set_ylabel(f"{feature_name} Value")
                ax.set_title(f"Univariate Time Series Reconstruction - {feature_name.upper()}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 평균 MSE와 MAE
                avg_mse = mse_list[idx]
                avg_mae = mae_list[idx]
                plt.suptitle(f"Sample #{idx} - {feature_name.upper()} Feature - MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}", fontsize=14)
                
            else:
                # 다변량 모드: 모든 feature 표시
                feature_names = self._get_feature_names()
                num_features = len(feature_names)
                
                # 서브플롯 배치 결정 (feature 수에 따라 동적으로)
                if num_features <= 4:
                    rows, cols = 2, 2
                elif num_features <= 6:
                    rows, cols = 2, 3
                elif num_features <= 9:
                    rows, cols = 3, 3
                else:
                    rows, cols = 4, 4
                
                fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
                if num_features > 1:
                    axes = axes.flatten()  # 2D 배열을 1D로 변환
                else:
                    axes = [axes]  # 단일 subplot인 경우 list로 변환
                
                for feat_idx in range(num_features):  # 모든 feature 표시
                    ax = axes[feat_idx]
                    
                    try:
                        # 해당 feature의 데이터 추출
                        original_vals = trues[idx, :, feat_idx]    # [seq_len] - 원본
                        reconstructed_vals = preds[idx, :, feat_idx]      # [seq_len] - 재구성
                        
                        # Reconstruction 시각화
                        ax.plot(range(len(original_vals)), original_vals, label="Original", color="blue")
                        ax.plot(range(len(reconstructed_vals)), reconstructed_vals, label="Reconstructed", color="red")
                        
                        ax.set_xlabel("Time Step")
                        ax.set_ylabel(feature_names[feat_idx])
                        ax.set_title(f"Feature: {feature_names[feat_idx]}")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                    except IndexError as e:
                        # feature 인덱스가 범위를 벗어난 경우
                        ax.text(0.5, 0.5, f'Feature {feat_idx}\nNot Available', 
                            ha='center', va='center', transform=ax.transAxes, fontsize=12)
                        ax.set_title(f"Feature {feat_idx} - N/A")
                
                # 사용하지 않는 서브플롯 숨기기
                for feat_idx in range(num_features, len(axes)):
                    axes[feat_idx].set_visible(False)
                
                # 평균 MSE와 MAE
                avg_mse = mse_list[idx]
                avg_mae = mae_list[idx]
                plt.suptitle(f"Sample #{idx} - Multivariate - MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}", fontsize=16)
            
            plt.tight_layout(rect=(0, 0, 1, 0.95))  # suptitle을 위한 공간 확보
            
            # 테스트 데이터 파일명 추출
            test_dataset_name = self.args.data
            if hasattr(self.args, 'test_data_path') and self.args.test_data_path:
                test_dataset_name = os.path.splitext(os.path.basename(self.args.test_data_path))[0]
            
            # 파일명 패턴 생성
            base_filename = f'{test_dataset_name}'
            
            # 샘플 정보 추가
            if hasattr(self.args, 'samples_per_file') and self.args.samples_per_file:
                base_filename += f'_samples{self.args.samples_per_file}perfile'
            else:
                base_filename += '_fulldata'
            
            # prediction length 추가
            base_filename += f'_pred{self.args.test_pred_len}'
            
            # feature 정보 추가 (단변량/다변량 모드에 따라)
            if is_univariate:
                # 단변량: feature 이름 포함
                feature_names = self._get_feature_names()
                if 0 <= self.args.target_feature < len(feature_names):
                    feature_name = feature_names[self.args.target_feature]
                else:
                    feature_name = f'feat{self.args.target_feature}'
                base_filename += f'_{feature_name}'
            else:
                # 다변량
                base_filename += '_multivariate'
            
            # learning rate, epoch, 스케줄러 정보 추가
            base_filename += f'_lr{self.args.learning_rate}_ep{self.args.train_epochs}'
            
            # 스케줄러 정보 추가
            if hasattr(self.args, 'cosine') and self.args.cosine:
                base_filename += "_cosine"
            elif hasattr(self.args, 'lradj') and self.args.lradj:
                base_filename += f"_{self.args.lradj}"
            
            # 샘플 인덱스 추가
            filename = f'{base_filename}_sample{idx}.png'
            filename = os.path.join(plots_dir, filename)
            
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot 저장 완료: {filename}")
            
        return