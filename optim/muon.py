import torch
from typing import Iterable
import random


class Muon(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[torch.Tensor],
            lr=0.02,
            odd_polinom_coef: Iterable[float] = (3.4445, -4.7750,  2.0315),
            NS_n: int=5,
            ):
        defaults = dict(lr=lr,odd_polinom_coef=odd_polinom_coef, NS_n=NS_n)
        super().__init__(params, defaults)
        self.params = params
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr: float
            lr = group['lr']
            for p in group['params']:
                p: torch.Tensor
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    p.data = p.data - lr * p.grad.data
                    continue
                p.data = p.data - lr * self.orthagonal(p.grad.data, n_step=group['NS_n'])
        return loss

    def orthagonal(self, X: torch.Tensor, odd_polinom_coef=None, n_step: int = 1):
        X = X / (torch.linalg.norm(X) + 1e-16)
        X_sqr = X @ X.mT
        for _ in range(n_step):
            ans = torch.zeros(X.size(), device=X.device)
            summand = X
            if odd_polinom_coef is None:
                odd_polinom_coef = self.defaults['odd_polinom_coef']
            for coef in odd_polinom_coef:
                coef: float
                ans += coef * summand
                summand = X_sqr @ summand
            X = ans
        return X


def orthagonal_loss(X: torch.Tensor):
    N, M = X.shape
    zeros_1 = X @ X.T - torch.eye(N)
    zeros_2 = X.T @ X - torch.eye(M)
    loss = min(
        torch.sum(zeros_1**2).item(),
        torch.sum(zeros_2**2).item()
    )
    return loss

def orthagonal_unit_test_loss(X: torch.Tensor, coefs: Iterable[float]):
    opt = Muon([X], odd_polinom_coef=coefs)
    ort_X = opt.orthagonal(X, n_step=5)
    return orthagonal_loss(ort_X)

def orthagonal_test(n_tests: int = 5, verbose = False):
    for test_number in range(n_tests):
        if verbose:
            print(f"TEST #{test_number}")
        N, M = random.randint(2, 10), random.randint(2, 10)
        if verbose:
            print(f"CREATE MATRIX {N}x{M}")
        X = torch.randn((N, M))
        init_loss = orthagonal_loss(X)
        loss = orthagonal_unit_test_loss(X, (3.4445, -4.7750,  2.0315))
        if verbose:
            print(f"INIT LOSS {init_loss}")
            print(f"LOSS {loss}")

