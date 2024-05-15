import torch
from collections import defaultdict

#ASAM and SAM implementations are taken from https://github.com/SamsungLabs/ASAM
class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

class Lookbehind_ASAM(ASAM):
    def __init__(self, optimizer, model, rho=0.5, eta=0.01, k_steps=5, alpha=0.5):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.k_steps = k_steps
        self.alpha = alpha
        self.k = 0
        if self.alpha == -1:
            self.scheduler = None
            self.tmp_alphas = []

        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                param_state['cached_slow_params'] = torch.zeros_like(p.data)
                param_state['cached_slow_params'].copy_(p.data)
                if self.alpha == -1:
                    param_state['first_descent_step'] = torch.zeros_like(p.data)


    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def get_current_k(self):
        return self.k

    def _cache_params(self):
        """ Cache the current optimizer parameters
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'].copy_(p.data)

    def _cache_slow_params(self):
        """ Cache the slow optimizer parameters
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_slow_params'].copy_(p.data)

    def _backup_and_load_slow_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_slow_params'])

    def _backup_and_load_cache(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @torch.no_grad()
    def descent_step(self):
        self._backup_and_load_cache()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.alpha == -1 and self.k == 0: #adaptive alpha
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['first_descent_step'] = torch.zeros_like(p.data)
                    param_state['first_descent_step'].copy_(p.data)

        self._cache_params()
        self._clear_and_load_backup()

        self.k += 1

        if self.k >= self.k_steps:
            self.k = 0
            if self.alpha == -1: #adaptive alpha
                self.tmp_alphas = []

            # Lookbehind and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.copy_(param_state['cached_params'])
                    if self.alpha == -1: #adaptive alpha
                        cos_sim = torch.nn.CosineSimilarity(dim=0)
                        tmp_alpha = cos_sim((param_state['first_descent_step']-param_state['cached_slow_params']).flatten(), (p.data-param_state['cached_slow_params']).flatten())
                        tmp_alpha = ((tmp_alpha+1.)/2.).item()
                        self.tmp_alphas.append(tmp_alpha)
                        p.data.mul_(tmp_alpha).add_(param_state['cached_slow_params'], alpha=1.0 - tmp_alpha)
                    else:
                        p.data.mul_(self.alpha).add_(param_state['cached_slow_params'], alpha=1.0 - self.alpha)
                    param_state['cached_params'].copy_(p.data)
                    param_state['cached_slow_params'].copy_(p.data)

class Lookbehind_SAM(Lookbehind_ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
