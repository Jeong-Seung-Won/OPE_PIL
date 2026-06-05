"""
Satellite Orbit Physics-Informed Loss for Anomaly Detection

Based on orbital mechanics constraints:
- Temporal smoothness (급변 탐지)
- Angular momentum conservation: h² = a(1-e²)
- Physical bounds for orbital elements

Features (순서대로):
    0: Semi_major_axis (a)
    1: Eccentricity (e)
    2: Inclination (i)
    3: RAAN (Ω)
    4: Argument_of_perigee (ω)
    5: Mean_anomaly (M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SatellitePhysicsLoss(nn.Module):
    """
    위성 궤도 이상 탐지용 Physics-Informed Loss
    
    Anomaly Injection 영향:
        - Along-track: a 급변
        - In-plane: a, e 급변
        - Out-of-plane: i 급변
    """
    
    # Feature indices (고정)
    IDX = {
        'a': 0,      # Semi_major_axis
        'e': 1,      # Eccentricity
        'i': 2,      # Inclination
        'RAAN': 3,   # RAAN
        'omega': 4,  # Argument_of_perigee
        'M': 5       # Mean_anomaly
    }
    
    def __init__(self, 
                 lambda_recon=1.0,
                 lambda_smooth=0.1,
                 lambda_angular=0.1,
                 lambda_bound=0.01,
                 verbose=True):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_smooth = lambda_smooth
        self.lambda_angular = lambda_angular
        self.lambda_bound = lambda_bound
        self.verbose = verbose
        
        if verbose:
            print(f"🛰️ SatellitePhysicsLoss initialized:")
            print(f"   λ_recon={lambda_recon}, λ_smooth={lambda_smooth}")
            print(f"   λ_angular={lambda_angular}, λ_bound={lambda_bound}")
    
    def forward(self, x_pred, x_true):
        """
        Args:
            x_pred: [batch, seq_len, 6] - 재구성 (criterion 호출 순서에 맞춤)
            x_true: [batch, seq_len, 6] - 원본
        
        Returns:
            L_total: scalar loss
        """
        idx = self.IDX
        
        # ========== 1. Reconstruction Loss ==========
        L_recon = F.mse_loss(x_pred, x_true)
        
        # ========== 2. Temporal Smoothness ==========
        # 이상치 주입 시 a, e, i가 급변 → 2차 미분이 커짐
        
        a_pred = x_pred[:, :, idx['a']]
        e_pred = x_pred[:, :, idx['e']]
        i_pred = x_pred[:, :, idx['i']]
        
        # 2차 미분 (가속도)
        dda = self._second_derivative(a_pred)
        dde = self._second_derivative(e_pred)
        ddi = self._second_derivative(i_pred)
        
        L_smooth = torch.mean(dda**2) + torch.mean(dde**2) + torch.mean(ddi**2)
        
        # ========== 3. Angular Momentum Conservation ==========
        # h² ∝ a(1-e²) → 시퀀스 내 일정해야 함
        # Along-track (Δa), In-plane (Δa, Δe) 이상 시 위반
        
        h_squared = a_pred * (1 - e_pred**2 + 1e-8)
        h_var = torch.var(h_squared, dim=1)  # [batch]
        L_angular = torch.mean(h_var)
        
        # ========== 4. Physical Bounds ==========
        # 정규화된 데이터에서는 효과 제한적이지만, 안전장치로 유지
        L_e_bound = torch.mean(F.relu(-e_pred) + F.relu(e_pred - 0.99))
        L_i_bound = torch.mean(F.relu(-i_pred) + F.relu(i_pred - 180))
        L_bound = L_e_bound + L_i_bound
        
        # ========== Total Loss ==========
        L_total = (self.lambda_recon * L_recon + 
                   self.lambda_smooth * L_smooth +
                   self.lambda_angular * L_angular +
                   self.lambda_bound * L_bound)
        
        return L_total
    
    def forward_with_details(self, x_pred, x_true):
        """
        개별 loss 값도 반환 (디버깅/로깅용)
        """
        idx = self.IDX
        losses = {}
        
        # 1. Reconstruction Loss
        L_recon = F.mse_loss(x_pred, x_true)
        losses['recon'] = L_recon.item()
        
        # 2. Temporal Smoothness
        a_pred = x_pred[:, :, idx['a']]
        e_pred = x_pred[:, :, idx['e']]
        i_pred = x_pred[:, :, idx['i']]
        
        dda = self._second_derivative(a_pred)
        dde = self._second_derivative(e_pred)
        ddi = self._second_derivative(i_pred)
        
        L_smooth = torch.mean(dda**2) + torch.mean(dde**2) + torch.mean(ddi**2)
        losses['smooth'] = L_smooth.item()
        
        # 3. Angular Momentum
        h_squared = a_pred * (1 - e_pred**2 + 1e-8)
        h_var = torch.var(h_squared, dim=1)
        L_angular = torch.mean(h_var)
        losses['angular'] = L_angular.item()
        
        # 4. Bounds
        L_e_bound = torch.mean(F.relu(-e_pred) + F.relu(e_pred - 0.99))
        L_i_bound = torch.mean(F.relu(-i_pred) + F.relu(i_pred - 180))
        L_bound = L_e_bound + L_i_bound
        losses['bound'] = L_bound.item()
        
        # Total
        L_total = (self.lambda_recon * L_recon + 
                   self.lambda_smooth * L_smooth +
                   self.lambda_angular * L_angular +
                   self.lambda_bound * L_bound)
        losses['total'] = L_total.item()
        
        return L_total, losses
    
    def _second_derivative(self, x):
        """2차 미분: x[t+1] - 2*x[t] + x[t-1]"""
        if x.size(1) < 3:
            return torch.zeros(x.size(0), 1, device=x.device)
        return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    
    def compute_anomaly_score(self, x_true, x_pred):
        """
        Point-wise anomaly score (inference용)
        
        Args:
            x_true: [batch, seq_len, 6]
            x_pred: [batch, seq_len, 6]
        
        Returns:
            scores: [batch, seq_len] - 각 시점별 이상 점수
        """
        idx = self.IDX
        
        with torch.no_grad():
            # 1. Reconstruction error (per point)
            recon_error = torch.mean((x_pred - x_true)**2, dim=2)  # [B, T]
            
            # 2. Angular momentum deviation (per point)
            a_pred = x_pred[:, :, idx['a']]
            e_pred = x_pred[:, :, idx['e']]
            
            h_pred = a_pred * (1 - e_pred**2 + 1e-8)
            h_mean = h_pred.mean(dim=1, keepdim=True)
            h_deviation = (h_pred - h_mean)**2
            
            # 3. Gradient magnitude (급변 탐지)
            a_grad = self._gradient_magnitude(a_pred)
            e_grad = self._gradient_magnitude(e_pred)
            i_grad = self._gradient_magnitude(x_pred[:, :, idx['i']])
            
            grad_score = a_grad + e_grad + i_grad
            
            # Combined score
            score = recon_error + 0.3 * h_deviation + 0.3 * grad_score
            
            return score
    
    def _gradient_magnitude(self, x):
        """1차 미분 절대값 (뒤쪽 패딩)"""
        if x.size(1) < 2:
            return torch.zeros_like(x)
        dx = torch.abs(x[:, 1:] - x[:, :-1])
        dx = F.pad(dx, (0, 0, 0, 1), mode='replicate')  # 뒤쪽 패딩
        return dx


class PhysicsAwareCriterion:
    """
    기존 criterion과 호환되는 wrapper
    exp_ad.py의 _select_criterion()에서 사용
    
    Usage:
        criterion = PhysicsAwareCriterion(lambda_smooth=0.1)
        loss = criterion(reconstructed, batch_x)
    """
    
    def __init__(self, 
                 lambda_recon=1.0,
                 lambda_smooth=0.1,
                 lambda_angular=0.1,
                 lambda_bound=0.01):
        
        self.physics_loss = SatellitePhysicsLoss(
            lambda_recon=lambda_recon,
            lambda_smooth=lambda_smooth,
            lambda_angular=lambda_angular,
            lambda_bound=lambda_bound,
            verbose=True
        )
    
    def __call__(self, x_pred, x_true):
        """
        기존 criterion(reconstructed, batch_x) 호출 형태와 호환
        """
        return self.physics_loss(x_pred, x_true)
    
    def get_detailed_loss(self, x_pred, x_true):
        """개별 loss 확인용"""
        return self.physics_loss.forward_with_details(x_pred, x_true)
