import torch


def build_optimizer(args, model):

    ed_params = filter(lambda x: id(x), model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [ {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
