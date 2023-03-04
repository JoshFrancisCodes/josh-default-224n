from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).
                
                '''
                Algorithm 1: Adam algorithm. (g_t^2 indicates the element-wise square g_t ⊙ g_t. All operations on vectors are element-wise. With B_1^t and B_2^t , we denote B_1 and B2 to the power t.)
                1. Require: α : Stepsize 
                2. Require: β_1, β_2 ∈ [0, 1): Exponential decay rates for the moment estimates 
                3. Require: f(θ): Stochastic objective function with parameters θ 
                4. Require: θ_0: Initial parameter vector 
                5. m_0 ← 0 (Initialize 1st moment vector) 
                6. v_0 ← 0 (Initialize 2nd moment vector) 
                7. t ← 0 (Initialize time step) 
                8. while θ_t not converged do:
                    a. t ← t + 1 
                    b. g_t ← ∇f_t(θ_t−1) (Get gradients w.r.t. stochastic objective function at timestep t) 
                    c. m_t ← β_1 · m_(t−1) + (1 − β_1) · g_t (Update biased first moment estimate) 
                    d. v_t ← β_2 · v_(t−1) + (1 − β_2) · g_t^2 (Update biased second raw moment estimate) 
                    e. mˆ_t ← m_t/(1 − β_1^t ) (Compute bias-corrected first moment estimate) 
                    f. vˆ_t ← v_t/(1 − β_2^t ) (Compute bias-corrected second raw moment estimate) 
                    g. θ_t ← θ_(t−1) − α · mˆ_t/( √ vˆ_t + ϵ) 
                9. return θt (Resulting parameters) 

                Note, at the expense of clarity, there is a more efficient version of the above algorithm where the last three lines in the loop are replaced with the following two lines: 
                e*. α_t ← α · √(1 − β_2^t) /(1 − β_1^t )
                f*. θ_t ← θ_(t−1) − α_t · m_t/( √ v_t + ϵ)
                You should use the more efficient version in your implementation. 
                '''
                
                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"] # 1
                beta1, beta2 = group["betas"] # 2
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                # correct_bias = group["correct_bias"]

                # Initialize state variables
                if len(state) == 0:
                    state["step"] = 0 # 7
                    state["m"] = torch.zeros_like(p.data) # 5
                    state["v"] = torch.zeros_like(p.data) # 6

                # Update state variables
                state["step"] += 1 # a
                m, v = state["m"], state["v"] 
                m.mul_(beta1).add_(grad, alpha=1 - beta1) # c
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # d

                alpha_t = alpha * math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"]) # e*
                p.data.addcdiv_(m, v.sqrt() + eps, value=-alpha_t) # f*
                
                # Compute update with weight decay
                if weight_decay > 0:
                    p.data.mul_(1 - alpha * weight_decay)

        return loss

