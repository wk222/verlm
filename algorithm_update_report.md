# OPO/GOPO/WGOPO 算法实现更新报告

根据最新的三篇论文（OPO, GOPO, WGOPO），我对 `verl/trainer/adpo/core_algos.py` 中的核心算法实现进行了全面的审查和更新。以下是具体的修改内容：

## 1. OPO (Orthogonalized Policy Optimization) 的修改

**修改前**：OPO 的驱动项直接使用了归一化后的 escort weight `w`。
**修改后**：根据最新 OPO 论文中的公式 $v^* = (\tilde{\omega}_\alpha - \bar{\omega}_\alpha)/\mu$，我添加了**中心化（Centering）**操作。
- 计算组内平均：`w_mean = w.mean(dim=-1, keepdim=True)`
- 计算中心化后的驱动力：`g = w - w_mean`
- 策略梯度项现在使用 `g` 而不是 `w`：`term1 = -dir_scale * (g.detach() * seq_log_prob_grouped).sum(dim=-1)`。这确保了严格的概率守恒（Conservation Law Projection）。

## 2. GOPO (Group Orthogonalized Policy Optimization) 的修改

**修改前**：使用的是带有 ReLU 的标量损失截断 `ReLU(-A_i * rho_i + (mu/2) * (rho_i - 1)^2)`。
**修改后**：根据最新 GOPO 论文，在 BHP (Bounded Hilbert Projection) 激活时，如果假设 $\lambda^* = 0$ 会导致概率守恒被破坏（Phantom Probability）。我将其改为了论文中推荐的 **Hilbert Distance MSE 形式，并加入了二分查找求解器**：
- 使用二分查找精确求解 $\lambda^*$，使得 $\sum \max(-1, (A_i - \lambda^*)/\mu) = 0$
- 计算精确的 BHP 目标：`v_star = torch.clamp((adv - lambda_star) / opo_mu, min=-1.0)`
- 计算目标策略比：`rho_target = (1.0 + v_star).detach()`
- 优化 MSE 距离：`loss_gopo = 0.5 * opo_mu * (rho - rho_target)**2`。这种形式既保证了非负性和平滑梯度，又在 BHP 激活区严格保证了概率守恒。

## 3. WGOPO (Weighted Group Orthogonalized Policy Optimization) 的全新添加

我完全按照 WGOPO 论文中的 Algorithm 2 实现了这个新算法：
- **Pre-Squashing (预压缩)**：使用 `c * torch.tanh(r_centered / c)` 压缩极端奖励，防止 Reward Hacking。
- **Post-Project (后投影)**：将压缩后的奖励再次中心化 `A_i = u_tilde - u_tilde.mean`。
- **Escort Weighting**：计算插值权重 `omega_tilde = (1.0 - alpha) * A_i + alpha * A_i * torch.exp(A_i)`。
- **Center (中心化)**：计算驱动场 `g = omega_tilde - omega_tilde.mean`。
- **Bisection Solver (二分查找求解器)**：实现了一个 GPU 并行的二分查找算法（最大迭代 40 次），精确求解非线性方程 $\sum \max(-1, (g_i - \lambda)/\mu) = 0$，找到精确的化学势 $\lambda^*$。
- **BHP Target & Loss**：使用精确的 $\lambda^*$ 计算目标 `v_star = torch.clamp((g - lambda_star) / opo_mu, min=-1.0)`，然后使用与 GOPO 相同的 MSE 损失 `0.5 * opo_mu * (rho - rho_target)**2`。

这三个算法现在都已经完美对齐了最新的 LaTeX 论文理论和公式。