"""MuSGD, the optimizer YOLO26 trains with, reimplemented.

`optimizer=auto` in the reference resolves to MuSGD on any run long enough to
matter, so leaving it out would mean this clone reproduces YOLO26's
architecture but not its recipe -- and a comparison against baseline/yolo26
would not say whether a difference came from the model or from the optimizer.

MuSGD is a hybrid. Every parameter gets a normal SGD-with-momentum update.
Matrix-shaped parameters (2D and 4D: every convolution and linear weight) get
a second, *orthogonalised* update on top of it:

    1. keep an exponential moving average of the gradient (momentum)
    2. flatten it to a matrix and push its singular values towards 1 with five
       Newton-Schulz iterations, which is a cheap stand-in for the UV^T of an
       SVD
    3. scale by sqrt(max(1, rows / cols)) and apply it with its own learning
       rate (`muon` times the group's)

Orthogonalising equalises how far each direction of a weight matrix moves, so
no single direction dominates the step. Vectors (biases, BatchNorm) have no
matrix structure to orthogonalise and take the plain SGD path only.

`build_optimizer` reproduces the reference's grouping, including the detail
that the classification heads train at three times the base learning rate.

Reference: Jordan et al., *Muon: An optimizer for hidden layers in neural
networks* (2024), https://kellerjordan.github.io/posts/muon/
"""

import torch
from torch import optim

# Newton-Schulz coefficients, tuned to maximise the convergence slope at zero.
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5

# The reference's split between the two updates, and the head's learning-rate
# multiplier while fine-tuning.
MUON_SCALE, SGD_SCALE = 0.2, 1.0
HEAD_LR_MULTIPLIER = 3.0


def zeropower_via_newtonschulz5(g: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal factor of `g` (a matrix or a batch of them).

    Five fixed quintic Newton-Schulz steps in bfloat16. The result is not
    exactly UV^T -- the singular values land somewhere around 0.5 to 1.5 --
    which is close enough for an optimizer and far cheaper than an SVD.
    """
    assert g.ndim in {2, 3}
    x = g.reshape(-1, g.size(-2), g.size(-1)).bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)      # top singular value <= 1
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    a, b, c = NS_COEFFS
    for _ in range(NS_STEPS):
        aa = x @ x.transpose(-2, -1)
        bb = torch.baddbmm(aa, aa, aa, beta=b, alpha=c)      # b * A + c * A @ A
        x = torch.baddbmm(x, bb, x, beta=a)                  # a * X + B @ X
    if transposed:
        x = x.transpose(-2, -1)
    return x.reshape(g.shape)


def muon_update(grads: list[torch.Tensor], momentums: list[torch.Tensor],
                beta: float = 0.95, nesterov: bool = True) -> list[torch.Tensor]:
    """Momentum, then orthogonalisation, for a whole parameter group at once.

    Matrices that share a row count are zero-padded to a common width and
    orthogonalised in one batched call; the padding stays zero throughout the
    iteration, so it does not change any result.
    """
    torch._foreach_mul_(momentums, beta)
    torch._foreach_add_(momentums, grads, alpha=1 - beta)
    if nesterov:
        updates = list(torch._foreach_mul(momentums, beta))
        torch._foreach_add_(updates, grads, alpha=1 - beta)
    else:
        updates = list(momentums)

    buckets: dict[tuple, list] = {}
    for i, update in enumerate(updates):
        matrix = update.view(len(update), -1) if update.ndim > 2 else update
        transposed = matrix.size(0) > matrix.size(1)
        if transposed:
            matrix = matrix.transpose(0, 1)
        scale = max(1, grads[i].size(-2) / grads[i].size(-1)) ** 0.5
        buckets.setdefault((matrix.size(0), scale, matrix.device, matrix.dtype),
                           []).append((i, matrix, transposed))

    for (_, scale, _, _), items in buckets.items():
        width = max(matrix.size(1) for _, matrix, _ in items)
        stacked = torch.stack([torch.nn.functional.pad(matrix, (0, width - matrix.size(1)))
                               for _, matrix, _ in items])
        stacked = zeropower_via_newtonschulz5(stacked).to(grads[items[0][0]].dtype).mul_(scale)
        for j, (i, matrix, transposed) in enumerate(items):
            out = stacked[j, :, :matrix.size(1)]
            updates[i] = (out.T if transposed else out).reshape(grads[i].shape)
    return updates


class MuSGD(optim.Optimizer):
    """SGD with momentum, plus an orthogonalised Muon step on matrix groups.

    A parameter group with `use_muon=True` gets both updates and therefore two
    momentum buffers; one with `use_muon=False` is plain SGD.
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False,
                 use_muon: bool = False, muon: float = MUON_SCALE, sgd: float = SGD_SCALE):
        super().__init__(params, {"lr": lr, "momentum": momentum, "nesterov": nesterov,
                                  "weight_decay": weight_decay, "use_muon": use_muon})
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]

            for p in params:
                if not self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                    if group["use_muon"]:
                        self.state[p]["momentum_buffer_sgd"] = torch.zeros_like(p)

            if group["use_muon"]:
                updates = muon_update([p.grad for p in params],
                                      [self.state[p]["momentum_buffer"] for p in params],
                                      beta=momentum, nesterov=nesterov)
                torch._foreach_add_(params, updates, alpha=-(lr * self.muon))
                buffers = [self.state[p]["momentum_buffer_sgd"] for p in params]
                lr *= self.sgd
            else:
                buffers = [self.state[p]["momentum_buffer"] for p in params]

            grads = [p.grad for p in params]
            if group["weight_decay"] != 0:
                grads = torch._foreach_add(grads, params, alpha=group["weight_decay"])
            torch._foreach_mul_(buffers, momentum)
            torch._foreach_add_(buffers, grads)
            updates = torch._foreach_add(grads, buffers, alpha=momentum) if nesterov else buffers
            torch._foreach_add_(params, updates, alpha=-lr)
        return loss


def build_optimizer(model, name: str = "musgd", lr: float = 0.01, momentum: float = 0.9,
                    weight_decay: float = 5e-4):
    """Group the parameters the way the reference does, then build the optimizer.

    Four kinds of parameter, decided per module rather than per name:

        matrices     2D/4D weights. Only these take the Muon update, and only
                     when the optimizer is MuSGD; under plain SGD they join
                     the decayed-weight group instead.
        bn weights   BatchNorm scales, no weight decay
        biases       no weight decay
        weights      anything else that is neither, with weight decay

    Each group is then split in two, because the reference trains the
    classification heads (`cv3`, `one2one_cv3`) at three times the base rate.
    """
    use_muon = name.lower() == "musgd"
    norm_types = tuple(v for k, v in torch.nn.__dict__.items()
                       if isinstance(v, type) and "Norm" in k)

    matrices, bn_weights, biases, weights = [], [], [], []
    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if param.ndim in {2, 4} and use_muon:
                matrices.append(param)
            elif "bias" in param_name:
                biases.append(param)
            elif isinstance(module, norm_types):
                bn_weights.append(param)
            else:
                weights.append(param)

    shared = {"lr": lr, "momentum": momentum, "nesterov": True}
    groups = [
        {"params": weights, **shared, "weight_decay": weight_decay, "name": "weight"},
        {"params": bn_weights, **shared, "weight_decay": 0.0, "name": "bn"},
        {"params": biases, **shared, "weight_decay": 0.0, "name": "bias"},
    ]
    if use_muon:
        groups.append({"params": matrices, **shared, "weight_decay": weight_decay,
                       "use_muon": True, "name": "muon"})

    head = model.detect
    boosted = {id(p) for branch in (head.cv3, head.one2one_cv3) for p in branch.parameters()}
    split = []
    for group in groups:
        params = group.pop("params")
        fast = [p for p in params if id(p) in boosted]
        rest = [p for p in params if id(p) not in boosted]
        # Keep empty groups out: an optimizer group with no parameters is
        # harmless but makes the learning-rate schedule harder to read.
        if fast:
            split.append({"params": fast, **group, "lr": group["lr"] * HEAD_LR_MULTIPLIER})
        if rest:
            split.append({"params": rest, **group})

    if use_muon:
        return MuSGD(split, muon=MUON_SCALE, sgd=SGD_SCALE)
    if name.lower() == "sgd":
        return optim.SGD(split)
    raise ValueError(f"unknown optimizer '{name}' (use musgd or sgd)")
