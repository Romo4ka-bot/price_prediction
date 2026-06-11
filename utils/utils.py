import random
import numpy as np
import torch
import torch.nn as nn


# import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_scheduler(optimizer: torch.optim.Optimizer,
                  num_steps: int,
                  decay: str,
                  lr: float,
                  lr_decay_factor: float,
                  wu_ratio: float,
                  decay_ratio: float
                  ) -> torch.optim.lr_scheduler.LRScheduler:
    wu_steps = int(wu_ratio * num_steps)
    decay_steps = int(decay_ratio * num_steps)
    num_steps = num_steps - decay_steps - wu_steps
    schedulers = [
        torch.optim.lr_scheduler.LinearLR(optimizer,
                                          total_iters=wu_steps,
                                          start_factor=1e-8,
                                          end_factor=1.0),
        torch.optim.lr_scheduler.ConstantLR(optimizer,
                                            total_iters=num_steps,
                                            factor=1)
    ]
    if decay == 'linear':
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(optimizer,
                                              total_iters=decay_steps,
                                              start_factor=1.0,
                                              end_factor=lr_decay_factor)
        )
    elif decay == 'cosine':
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=decay_steps,
                                                       eta_min=lr * lr_decay_factor)
        )
    else:
        raise NotImplementedError()
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=[wu_steps, wu_steps + num_steps]
    )


def get_param_groups(
        model: nn.Module,
        lr: float,
        weight_decay: float,
        no_weight_decay: tuple[str] = ('embed', 'norm', 'bias')
        # lr_decay_by_block: float = None,
) -> list:
    # if lr_decay_by_block is None:
    decay = set()
    no_decay = set()
    # else:
    #     raise NotImplementedError
    #     groups = [
    #         {
    #             'no_decay': set(),
    #             'decay': set()
    #         }
    #         for _ in range((len(model.blocks) + 2))
    #     ]

    for name, param in model.named_parameters():
        # if not param.requires_grad:
        #     continue

        # if lr_decay_by_block is None:
        if any(skip_name in name for skip_name in no_weight_decay):
            no_decay.add(name)
        else:
            decay.add(name)
        # else:
        #     if 'block' in name:
        #         i = int(name.split('.')[1]) + 1
        #         if any(skip_name in name for skip_name in no_weight_decay):
        #             groups[i]['no_decay'].add(name)
        #         else:
        #             groups[i]['decay'].add(name)
        #     elif 'embed' in name:
        #         if any(skip_name in name for skip_name in no_weight_decay):
        #             groups[0]['no_decay'].add(name)
        #         else:
        #             groups[0]['decay'].add(name)
        #     else:
        #         if any(skip_name in name for skip_name in no_weight_decay):
        #             groups[0]['no_decay'].add(name)
        #         else:
        #             groups[0]['decay'].add(name)

    param_dict = {pn: p for pn, p in model.named_parameters()}

    # if lr_decay_by_block is None:
    assert len(decay & no_decay) == 0
    assert len(param_dict.keys() - (decay | no_decay)) == 0
    return [
        {"params": [param_dict[pn] for pn in sorted(list(decay))],
         "lr": lr,
         "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
         "lr": lr,
         "weight_decay": 0.0},
    ]
    # else:
    #     res = []
    #     t = len(groups) - 1
    #     for i, pg in enumerate(groups):
    #         if i == 0:
    #             inter = pg['no_decay'] & pg['decay']
    #             union = pg['no_decay'] | pg['decay']
    #         else:
    #             inter &= pg['no_decay'] & pg['decay']
    #             union |= pg['no_decay'] | pg['decay']
    #
    #         res.extend([
    #             {"params": [param_dict[pn] for pn in sorted(list(pg['decay']))],
    #              "lr": lr * lr_decay_by_block ** (t - i),
    #              "weight_decay": weight_decay},
    #             {"params": [param_dict[pn] for pn in sorted(list(pg['no_decay']))],
    #              "lr": lr * lr_decay_by_block ** (t - i),
    #              "weight_decay": 0.0},
    #         ])
    #     assert len(inter) == 0
    #     assert len(param_dict.keys() - union) == 0
    #     return res


def accuracy(pred: torch.Tensor, label: torch.Tensor) -> float:
    pred = pred.argmax(1)
    return (pred == label).float().mean().item()


def mape(pred: torch.Tensor, label: torch.Tensor, epsilon: float = 1e-8) -> float:
    return torch.abs(
        (label - pred) / (label + epsilon)
    ).mean().item()


def logcosh_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    # x = pred - target
    # loss = x * x.tanh()
    loss = torch.cosh(pred - target).log()
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise NotImplementedError


class LogCoshLoss(nn.modules.loss._Loss):
    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return logcosh_loss(input, target, reduction=self.reduction)


if __name__ == '__main__':
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=10)
    for epoch in range(12):
        print(epoch, opt.param_groups[0]['lr'])
        scheduler.step()
