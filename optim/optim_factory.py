""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


from torch.optim import Optimizer

def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


def create_two_optimizer(args, model, filter_bias_and_bn=True):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       ((not any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": args.weight_decay,
            "lr": args.lr1
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       ((any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": 0.0,
            "lr": args.lr1
        },
        {
            "params": [p for n, p in model.visual_encoder.named_parameters() if
                       ((not any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": args.weight_decay,
            "lr": args.lr2
        },
        {
            "params": [p for n, p in model.visual_encoder.named_parameters() if
                       ((any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": 0.0,
            "lr": args.lr2
        },

    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    return optimizer
