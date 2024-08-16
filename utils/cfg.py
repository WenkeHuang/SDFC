from yacs.config import CfgNode as CN
from utils.utils import log_msg


# simplify cfg
def simplify_cfg(args, cfg):
    dump_cfg = CN()
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.OPTIMIZER = cfg.OPTIMIZER
    if args.server in list(cfg['Server'].keys()):
        dump_cfg['Server'] = CN()
        dump_cfg['Server'][args.server] = CN()
        dump_cfg['Server'][args.server] = cfg['Server'][args.server]
    if args.optim in list(cfg['Optim'].keys()):
        dump_cfg['Optim'] = CN()
        dump_cfg['Optim'][args.optim] = CN()
        dump_cfg['Optim'][args.optim] = cfg['Optim'][args.optim]

    if args.attack_type != 'None':
        dump_cfg['attack'] = CN()
        dump_cfg['attack'].bad_client_rate = cfg['attack'].bad_client_rate
        dump_cfg['attack'].noise_data_rate = cfg['attack'].noise_data_rate
        dump_cfg['attack'][args.attack_type] = cfg['attack'][args.attack_type]
    return dump_cfg


def show_cfg(args, cfg):
    # dump_cfg = CN()
    # dump_cfg.DATASET = cfg.DATASET
    # dump_cfg.OPTIMIZER = cfg.OPTIMIZER
    # dump_cfg.Optim = args.[optm]
    # if args.attack_type != 'None':
    #     dump_cfg['attack'] = cfg['attack']
    print(log_msg("CONFIG:\n{}".format(cfg.dump()), "INFO"))
    return None

CFG = CN()
'''Federated dataset'''
CFG.DATASET = CN()
CFG.DATASET.dataset = "fl_cifar10"  #
CFG.DATASET.communication_epoch = 50
CFG.DATASET.n_classes = 10

CFG.DATASET.parti_num = 10
CFG.DATASET.online_ratio = 1.0
CFG.DATASET.train_val_domain_ratio = 0.9
CFG.DATASET.val_scale = 256
CFG.DATASET.backbone = "resnet18"
CFG.DATASET.pretrained = False
CFG.DATASET.aug = "weak"
CFG.DATASET.val_aug = "strong"
CFG.DATASET.beta = 0.5

'''attack'''
CFG.attack = CN()
CFG.attack.bad_client_rate = 0.3
CFG.attack.noise_data_rate = 0.5

CFG.attack.byzantine = CN()
CFG.attack.byzantine.evils = 'PairFlip'  # PairFlip SymFlip AddNoise RandomNoise None
CFG.attack.byzantine.dataset_type = 'single_domain'

CFG.attack.byzantine.dev_type = 'std'
CFG.attack.byzantine.lamda = 10.0
CFG.attack.byzantine.threshold_diff = 1e-5

CFG.attack.backdoor = CN()
CFG.attack.backdoor.evils = 'base_backdoor'  # base_backdoor semantic_backdoor
CFG.attack.backdoor.backdoor_label = 2
CFG.attack.backdoor.trigger_position = [
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 5], [0, 0, 6],
    [0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 2, 4], [0, 2, 5], [0, 2, 6], ]
CFG.attack.backdoor.trigger_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]
CFG.attack.backdoor.semantic_backdoor_label = 3

'''Federated OPTIMIZER'''
CFG.OPTIMIZER = CN()
CFG.OPTIMIZER.type = 'SGD'
CFG.OPTIMIZER.momentum = 0.9
CFG.OPTIMIZER.weight_decay = 1e-5
CFG.OPTIMIZER.local_epoch = 10
CFG.OPTIMIZER.local_train_batch = 64
CFG.OPTIMIZER.local_test_batch = 64
CFG.OPTIMIZER.val_batch = 64
CFG.OPTIMIZER.local_train_lr = 1e-3

''''''
CFG.Server = CN()

CFG.Server.FedOpt = CN()
CFG.Server.FedOpt.global_lr = 0.5

CFG.Server.FedCPA = CN()
CFG.Server.FedCPA.top_rate = 0.5 # 0.1 0.2 1.0

CFG.Server.FLTrust = CN()
CFG.Server.FLTrust.public_epoch = 20

CFG.Server.CRFL = CN()
CFG.Server.CRFL.param_clip_thres = 15
CFG.Server.CRFL.epoch_index_weight = 2
CFG.Server.CRFL.epoch_index_bias = 10
CFG.Server.CRFL.sigma = 0.01
CFG.Server.CRFL.scale_factor = 1.0

CFG.Server.RLR = CN()
CFG.Server.RLR.server_lr = 1.0 # 1.0 1e-1 1e-2 1e-4
CFG.Server.RLR.robustLR_threshold = 0.5 # 0.1 1.0 5.0 10.0

CFG.Server.SageFlow = CN()
CFG.Server.SageFlow.eth = 2.0 # 1.0 0.5 4.0
CFG.Server.SageFlow.delta = 1.0 # 0.5 2.0

'''Optim'''
CFG.Optim = CN()

CFG.Optim.FedAvG = CN()

CFG.Optim.FedProx = CN()
CFG.Optim.FedProx.mu = 0.01


CFG.Optim.PrevNormalFedFish = CN()
CFG.Optim.PrevNormalFedFish.lamda = 1.0

CFG.Optim.LocalFedFish = CN()
CFG.Optim.LocalFedFish.lamda = 1.0

CFG.Optim.ValFedFish = CN()
CFG.Optim.ValFedFish.lamda = 1.0


CFG.Optim.PrevAbsFedFish = CN()
CFG.Optim.PrevAbsFedFish.lamda = 1.0

CFG.Optim.PrevAbsFedGradient = CN()
CFG.Optim.PrevAbsFedGradient.lamda = 1.0
# CFG.Optim.PrevFedFish = CN()
# CFG.Optim.PrevFedFish.lamda = 0.01
# CFG.Optim.PrevWeightFedFish = CN()
# CFG.Optim.PrevWeightFedFish.lamda = 0.01
