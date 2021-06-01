import os
import collections
import torch


def check_args(args, rank=0):
    args.setting_file = os.path.join(args.checkpoint_dir, args.setting_file)
    args.log_file = os.path.join(args.checkpoint_dir, args.log_file)
    if os.path.exists(args.setting_file) and rank == 0:
        os.remove(args.setting_file)
    if os.path.exists(args.log_file) and rank == 0:
        os.remove(args.log_file)
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(args.setting_file, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            print('------------ Options -------------')
            for k in args.__dict__:
                v = args.__dict__[k]
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            print('------------ End -------------')


def torch_init_model(model, init_checkpoint, key=None):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    if key is not None:
        state_dict = state_dict[key]
    if type(state_dict) != collections.OrderedDict:
        state_dict = state_dict.state_dict()
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))
