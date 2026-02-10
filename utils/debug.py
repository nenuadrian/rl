import collections, math, torch

"""
check_optimizer_param_duplicates(self.opt)
param_grad_stats(self.policy)
# quick parameter magnitude check
params = [
    (n, p.detach().abs().mean().item())
    for n, p in self.policy.named_parameters()
]
params = sorted(params, key=lambda x: x[1], reverse=True)
print(">>> top param abs means:", params[:8])
"""


def check_optimizer_param_duplicates(optim):
    ids = collections.Counter()
    for group in optim.param_groups:
        for p in group["params"]:
            ids[id(p)] += 1
    dup = {k: c for k, c in ids.items() if c > 1}
    if dup:
        print(">>> DUPLICATE PARAMS IN OPTIMIZER (param id -> count):", dup)
    else:
        print(">>> no duplicate params in optimizer")


def param_grad_stats(model):
    grads = []
    stats = []
    for name, p in model.named_parameters():
        if p.grad is None:
            g_norm = 0.0
            has_inf = False
            has_nan = False
        else:
            g = p.grad
            g_norm = g.detach().norm().item()
            has_inf = torch.isinf(g).any().item()
            has_nan = torch.isnan(g).any().item()
        stats.append((name, g_norm, has_nan, has_inf))
        grads.append(g_norm)
    # Top contributors
    stats_sorted = sorted(stats, key=lambda x: x[1], reverse=True)
    print(">>> top 10 grad norms (param, norm, has_nan, has_inf):")
    for s in stats_sorted[:10]:
        print(s)
    total_norm = math.sqrt(sum([g * g for g in grads]))
    print(">>> total grad norm (pre-clip, computed):", total_norm)
