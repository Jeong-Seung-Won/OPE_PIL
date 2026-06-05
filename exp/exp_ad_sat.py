"""
exp/exp_ad_sat.py

Sub-Adjacent Transformer (IJCAI 2024) 전용 실험 클래스.
Exp_AD 상속 — train/test loop override.

Train loss (2단계):
  phase-1 (홀수 epoch): rec_loss
  phase-2 (짝수 epoch): 2*rec_loss - k * loss_attn
  → 원본 solver.py의 loss3 방식

Anomaly score:
  softmax(-loss_attn, temperature) * rec_loss  (per-timestep)
  → window별 mean으로 집계
"""

import os
import csv
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve
from datetime import datetime

from exp.exp_ad import Exp_AD
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')


def _point_adjust(y_true, y_pred):
    y_pa = y_pred.copy()
    n = len(y_true)
    i = 0
    while i < n:
        if y_true[i] == 1:
            j = i
            while j < n and y_true[j] == 1:
                j += 1
            if y_pa[i:j].any():
                y_pa[i:j] = 1
            i = j
        else:
            i += 1
    return y_pa


class Exp_AD_SAT(Exp_AD):

    def _build_model(self):
        from models.sub_adjacent_transformer import Model
        if self.args.ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.device = torch.device(f'cuda:{self.args.local_rank}')
            model = Model(self.args).cuda()
            return DDP(model, device_ids=[self.args.local_rank])
        elif self.args.dp:
            from torch.nn import DataParallel
            self.device = self.args.gpu
            return DataParallel(Model(self.args),
                                device_ids=self.args.device_ids).to(self.device)
        else:
            self.device = self.args.gpu
            return Model(self.args).to(self.device)

    def _raw_model(self):
        if self.args.ddp or self.args.dp:
            return self.model.module
        return self.model

    # ── train ─────────────────────────────────────────────────────────────────
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        feature_names = self._get_feature_names()
        span = getattr(self.args, 'sat_span', [20, 30])
        print(f"SAT train: {len(feature_names)} features, "
              f"span={span}, k={getattr(self.args, 'sat_k', 3.0)}")

        # ── Physics rec criterion (--loss Physics 시) ────────────────────────
        #   SATLoss = 2 * rec_loss - k * loss_attn  (phase 2)
        #   기본:    rec_loss = MSE(output, x)
        #   Physics: rec_loss = SatellitePhysicsLoss(output, x)
        #            = λ_r·MSE + λ_s·smooth + λ_a·angular + λ_b·bound
        # SAT의 attention term은 그대로 유지하고 reconstruction 항만 augment.
        rec_criterion = None
        if str(getattr(self.args, 'loss', 'SATLoss')).lower() == 'physics':
            from utils.physics_loss import SatellitePhysicsLoss
            rec_criterion = SatellitePhysicsLoss(
                lambda_recon  =getattr(self.args, 'physics_lambda_recon',   1.0),
                lambda_smooth =getattr(self.args, 'physics_lambda_smooth',  0.1),
                lambda_angular=getattr(self.args, 'physics_lambda_angular', 0.1),
                lambda_bound  =getattr(self.args, 'physics_lambda_bound',   0.01),
                verbose=True,
            ).to(self.device)
            print(f"[SAT + Physics] reconstruction term augmented with "
                  f"smooth + angular + bound penalties.")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        if self.args.use_early_stop:
            _, val_loader = self._get_data(flag='val')
            early_stopping = EarlyStopping(self.args, verbose=True)

        optimizer = self._select_optimizer()

        scheduler = None
        if self.args.cosine:
            tmax = min(self.args.tmax, self.args.train_epochs)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=tmax, eta_min=1e-6)

        model_dir = f"results/{self.args.model}"
        log_dir   = os.path.join(model_dir, "training_logs")
        os.makedirs(log_dir, exist_ok=True)
        test_name = (os.path.splitext(os.path.basename(self.args.test_data_path))[0]
                     if getattr(self.args, 'test_data_path', None)
                     else self.args.data)
        csv_path = os.path.join(
            log_dir,
            f"training_log_{test_name}_sat"
            f"_lr{self.args.learning_rate}_ep{self.args.train_epochs}.csv"
        )
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'train_loss', 'val_loss', 'phase', 'epoch_time', 'lr'])
        print(f"Training log → {csv_path}")

        t_start = time.time()
        for epoch in range(self.args.train_epochs):
            # 원본처럼 홀수/짝수 epoch로 phase 교체
            phase = 1 if epoch % 2 == 0 else 2

            self.model.train()
            ep_losses = []
            t0 = time.time()

            for batch_x, _, bxm, bym in train_loader:
                batch_x = batch_x.float().to(self.device)
                optimizer.zero_grad()

                output = self._raw_model()(batch_x)
                loss   = self._raw_model().compute_loss(
                    batch_x, output, phase=phase, rec_criterion=rec_criterion)

                loss.backward()
                max_norm = float(getattr(self.args, 'max_grad_norm', 1.0))
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm)
                optimizer.step()
                ep_losses.append(loss.item())

            avg_loss = float(np.mean(ep_losses))
            t_ep     = time.time() - t0
            cur_lr   = optimizer.param_groups[0]['lr']

            if self.args.use_early_stop:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for bx, _, bxm, bym in val_loader:
                        bx     = bx.float().to(self.device)
                        out    = self._raw_model()(bx)
                        vloss  = self._raw_model().compute_loss(
                            bx, out, phase=2, rec_criterion=rec_criterion)
                        val_losses.append(vloss.item())
                avg_val = float(np.mean(val_losses))
                print(f"Epoch {epoch+1:3d} [phase{phase}] | "
                      f"train={avg_loss:.5f} val={avg_val:.5f} | "
                      f"{t_ep:.1f}s | lr={cur_lr:.2e}")
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow(
                        [epoch+1, avg_loss, avg_val, phase, t_ep, cur_lr])
                early_stopping(avg_val, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break
            else:
                print(f"Epoch {epoch+1:3d} [phase{phase}] | "
                      f"train={avg_loss:.5f} | {t_ep:.1f}s | lr={cur_lr:.2e}")
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow(
                        [epoch+1, avg_loss, '', phase, t_ep, cur_lr])
                self._save_checkpoint(path, epoch)

            if self.args.cosine and scheduler:
                scheduler.step()
            elif not self.args.no_scheduler and not self.args.cosine:
                adjust_learning_rate(optimizer, epoch + 1, self.args)

        print(f"Total training time: {time.time()-t_start:.1f}s")

        if self.args.use_early_stop:
            ckpt = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(
                torch.load(ckpt, map_location='cpu'), strict=False)

        # train score 저장
        try:
            anomaly_dir = os.path.join(path, 'anomaly')
            os.makedirs(anomaly_dir, exist_ok=True)
            print("Computing train anomaly scores...")
            train_scores = self._get_scores(train_loader)
            np.save(os.path.join(anomaly_dir,
                                 'train_reconstruction_errors.npy'),
                    train_scores)
            print(f"Train scores: mean={train_scores.mean():.6f} "
                  f"std={train_scores.std():.6f}")
        except Exception as e:
            print(f"[WARN] train score computation failed: {e}")

        return self.model

    def _get_scores(self, loader):
        """per-window mean anomaly score 반환"""
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for batch_x, _, bxm, bym in loader:
                batch_x = batch_x.float().to(self.device)
                scores  = self._raw_model().anomaly_score_batch(batch_x)  # [B, L]
                all_scores.extend(scores.mean(axis=-1).tolist())
        return np.array(all_scores)
    '''
    # ── test ──────────────────────────────────────────────────────────────────
    def test(self, setting):
        import json

        test_data, test_loader = self._get_data(flag='test')

        anomaly_dir    = os.path.join(self.args.checkpoints, setting, 'anomaly')
        train_err_path = os.path.join(anomaly_dir,
                                      'train_reconstruction_errors.npy')
        train_errors   = (np.load(train_err_path)
                          if os.path.exists(train_err_path) else None)

        stride = getattr(test_data, 'stride', 1)
        self.model.eval()

        all_scores, all_labels = [], []

        print("[Exp_AD_SAT] Running test inference...")
        with torch.no_grad():
            for i, (batch_x, _, bxm, bym) in enumerate(
                    tqdm(test_loader, desc='Test')):
                batch_x = batch_x.float().to(self.device)
                scores  = self._raw_model().anomaly_score_batch(batch_x)  # [B, L]
                bs      = batch_x.size(0)

                for b in range(bs):
                    win_idx = i * test_loader.batch_size + b
                    # window별 score → mean
                    all_scores.append(float(scores[b].mean()))

                    if (hasattr(test_data, 'labels')
                            and test_data.labels is not None):
                        win_start = win_idx * stride
                        win_end   = win_start + self.args.seq_len
                        lbl       = test_data.labels
                        seg       = lbl[win_start:min(win_end, len(lbl))]
                        all_labels.append(int((seg > 0).any()))

        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels) if all_labels else None
        anomaly_eval = {}

        if labels_arr is not None and len(labels_arr) == len(scores_arr):
            y_true = (labels_arr > 0).astype(int)
            try:
                auroc = float(roc_auc_score(y_true, scores_arr))
                print(f"AUROC = {auroc:.6f}")
                anomaly_eval['auroc'] = auroc
            except Exception as e:
                print(f"[WARN] AUROC failed: {e}")

            try:
                pos_n, neg_n = int(y_true.sum()), int((y_true == 0).sum())
                if pos_n > 0 and neg_n > 0:
                    _, _, thr_arr = precision_recall_curve(y_true, scores_arr)
                    if len(thr_arr) > 2000:
                        thr_arr = thr_arr[::len(thr_arr) // 2000]

                    best_f1 = best_thr = best_p = best_r = 0.
                    best_tp = best_fp = best_tn = best_fn = 0
                    eps = 1e-12
                    for thr in thr_arr:
                        yp    = (scores_arr >= thr).astype(int)
                        yp_pa = _point_adjust(y_true, yp)
                        tp = int(((yp_pa == 1) & (y_true == 1)).sum())
                        fp = int(((yp_pa == 1) & (y_true == 0)).sum())
                        tn = int(((yp_pa == 0) & (y_true == 0)).sum())
                        fn = int(((yp_pa == 0) & (y_true == 1)).sum())
                        p  = tp / (tp + fp + eps)
                        r  = tp / (tp + fn + eps)
                        f1 = 2 * p * r / (p + r + eps)
                        if f1 > best_f1:
                            best_f1, best_thr = f1, float(thr)
                            best_p, best_r    = p, r
                            best_tp, best_fp  = tp, fp
                            best_tn, best_fn  = tn, fn

                    print(f"PA-F1={best_f1:.6f} thr={best_thr:.6f} "
                          f"P={best_p:.4f} R={best_r:.4f}")
                    anomaly_eval.update({
                        'best_f1':        float(best_f1),
                        'best_threshold': best_thr,
                        'best_precision': float(best_p),
                        'best_recall':    float(best_r),
                        'tp': best_tp, 'fp': best_fp,
                        'tn': best_tn, 'fn': best_fn,
                        'support_pos': pos_n, 'support_neg': neg_n,
                    })
            except Exception as e:
                print(f"[WARN] PA-F1 failed: {e}")

        # JSON 저장
        ts        = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        test_name = (os.path.splitext(os.path.basename(
                         self.args.test_data_path))[0]
                     if getattr(self.args, 'test_data_path', None)
                     else self.args.data)
        model_dir = f"results/{self.args.model}"
        res_dir   = os.path.join(model_dir, "results")
        os.makedirs(res_dir, exist_ok=True)

        results = {
            "experiment_timestamp": ts,
            "model":           self.args.model,
            "test_csv":        test_name,
            "loss":            "rec_loss - k * loss_attn",
            "sat_span":        getattr(self.args, 'sat_span',        [20, 30]),
            "sat_k":           getattr(self.args, 'sat_k',           3.0),
            "sat_temperature": getattr(self.args, 'sat_temperature', 50),
            "sat_linear_attn": getattr(self.args, 'sat_linear_attn', True),
            "d_model":         getattr(self.args, 'd_model',         512),
            "e_layers":        getattr(self.args, 'e_layers',        3),
            "learning_rate":   float(self.args.learning_rate),
            "train_epochs":    self.args.train_epochs,
            "anomaly": {
                "method":         "softmax_attn_x_rec_loss",
                "auroc":          anomaly_eval.get('auroc'),
                "best_f1":        anomaly_eval.get('best_f1'),
                "best_precision": anomaly_eval.get('best_precision'),
                "best_recall":    anomaly_eval.get('best_recall'),
                "best_threshold": anomaly_eval.get('best_threshold'),
                "tp": anomaly_eval.get('tp'), "fp": anomaly_eval.get('fp'),
                "tn": anomaly_eval.get('tn'), "fn": anomaly_eval.get('fn'),
                "support_pos": anomaly_eval.get('support_pos'),
                "support_neg": anomaly_eval.get('support_neg'),
                "score_stats": {
                    "mean": float(scores_arr.mean()),
                    "std":  float(scores_arr.std()),
                    "p95":  float(np.percentile(scores_arr, 95)),
                    "p99":  float(np.percentile(scores_arr, 99)),
                },
            }
        }
        if train_errors is not None:
            results["anomaly"]["train_score_stats"] = {
                "mean": float(train_errors.mean()),
                "std":  float(train_errors.std()),
                "p95":  float(np.percentile(train_errors, 95)),
                "p99":  float(np.percentile(train_errors, 99)),
            }

        json_path = os.path.join(
            res_dir,
            f"results_{test_name}_sat"
            f"_d{self.args.d_model}_e{self.args.e_layers}"
            f"_multivariate_{ts}.json"
        )
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON saved → {json_path}")
        return
    '''