
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import sys
import os

# Add the repo root to sys.path
sys.path.append(r"c:\Users\wzx\Documents\GitHub\verlm")

from verl.trainer.adpo.core_algos import _compute_plackett_luce_loss, adpo_policy_loss

def test_poly_plackett_luce():
    print("Testing Poly-Plackett-Luce Loss...")
    # Mock data
    P, G = 2, 4
    u = torch.randn(P, G, requires_grad=True)
    adv = torch.randn(P, G)
    epsilon = 1.0
    
    # Compute standard loss
    loss_std = _compute_plackett_luce_loss(u, adv, use_poly_loss=False)
    
    # Compute poly loss
    loss_poly = _compute_plackett_luce_loss(u, adv, use_poly_loss=True, epsilon=epsilon)
    
    print(f"Standard Loss: {loss_std.item()}")
    print(f"Poly Loss: {loss_poly.item()}")
    
    # Manual verification for one sample
    # Sort u by adv
    idx = torch.argsort(adv[0], descending=True)
    u_sorted = u[0][idx]
    
    # Calculate p_rank for each position
    # p_rank_k = exp(u_k) / sum(exp(u_j)) for j >= k
    loss_manual = 0.0
    for k in range(G-1):
        log_sum_exp = torch.logsumexp(u_sorted[k:], dim=0)
        log_p = u_sorted[k] - log_sum_exp
        p = torch.exp(log_p)
        
        term = -log_p + epsilon * (1 - p)
        loss_manual += term
    
    loss_manual /= (G-1)
    
    # Check if matches (for the first sample, but the function returns mean over batch)
    # Let's compute for batch
    loss_manual_batch = 0.0
    for i in range(P):
        idx = torch.argsort(adv[i], descending=True)
        u_sorted = u[i][idx]
        loss_sample = 0.0
        for k in range(G-1):
            log_sum_exp = torch.logsumexp(u_sorted[k:], dim=0)
            log_p = u_sorted[k] - log_sum_exp
            p = torch.exp(log_p)
            term = -log_p + epsilon * (1 - p)
            loss_sample += term
        loss_manual_batch += loss_sample / (G-1)
    
    loss_manual_batch /= P
    
    print(f"Manual Poly Loss: {loss_manual_batch.item()}")
    assert torch.allclose(loss_poly, loss_manual_batch), "Poly-PL Loss mismatch!"
    print("Poly-Plackett-Luce Loss Test Passed!")

def test_poly_softmax():
    print("\nTesting Poly-Softmax Loss...")
    # Mock data
    B, G = 2, 4
    # Reshape mode: P=2, G=4. Batch size = 8
    # But adpo_policy_loss takes flat inputs usually, but here we test the logic inside
    # Let's mock the inputs to adpo_policy_loss
    
    # We need to mock config
    config = DictConfig({
        "policy_loss": {
            "loss_variant": "softmax",
            "use_poly_loss": True,
            "poly_epsilon": 2.0,
            "tau": 1.0,
            "num_generations": G
        }
    })
    
    # Mock inputs
    # We need log_prob, old_log_prob, advantages, response_mask
    # We want to trigger the reshape path
    # batch_size = P * G
    batch_size = B * G
    log_prob = torch.randn(batch_size, 10) # (B, SeqLen)
    old_log_prob = torch.randn(batch_size, 10)
    advantages = torch.randn(batch_size)
    response_mask = torch.ones(batch_size, 10)
    
    # Run loss
    loss, metrics = adpo_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config
    )
    
    print(f"Poly-Softmax Loss: {loss.item()}")
    
    # Manual verification
    # Reconstruct u and q
    # seq_log_prob
    seq_log_prob = log_prob.sum(dim=-1) / 10.0 # length norm is True by default
    seq_old_log_prob = old_log_prob.sum(dim=-1) / 10.0
    log_ratio = seq_log_prob - seq_old_log_prob
    u = log_ratio.view(B, G)
    adv = advantages.view(B, G)
    beta_reward = 0.3 # default
    q = F.softmax(adv / beta_reward, dim=-1)
    epsilon = 2.0
    
    log_p = F.log_softmax(u, dim=-1)
    p = torch.exp(log_p)
    
    ce_term = -(q * log_p).sum(dim=-1)
    poly_term = (q * epsilon * (1 - p)).sum(dim=-1)
    expected_loss = (ce_term + poly_term).mean()
    
    print(f"Expected Loss: {expected_loss.item()}")
    assert torch.allclose(loss, expected_loss), "Poly-Softmax Loss mismatch!"
    print("Poly-Softmax Loss Test Passed!")

if __name__ == "__main__":
    try:
        test_poly_plackett_luce()
        test_poly_softmax()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
