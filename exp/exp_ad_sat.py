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
        self.device = self.args.gpu
        return Model(self.args).to(self.device)

    def _raw_model(self):
        return self.model

    # ── train ─────────────────────────────────────────────────────────────────
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        feature_names = self._get_feature_names()
        span = getattr(self.args, 'sat_span', [20, 30])
        print(f"SAT train: {len(feature_names)} features, "
              f"span={span}, k={getattr(self.args, 'sat_k', 3.0)}")

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
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for batch_x, _, bxm, bym in loader:
                batch_x = batch_x.float().to(self.device)
                scores  = self._raw_model().anomaly_score_batch(batch_x)  # [B, L]
                all_scores.extend(scores.mean(axis=-1).tolist())
        return np.array(all_scores)
    