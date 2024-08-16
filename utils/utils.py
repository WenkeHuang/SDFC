import functools
from collections import Counter
import numpy as np
import torch
import os

import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "TEST": 31,
        "OOD": 33,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg

def merge_from_list(args, opt):
    if len(opt) % 2 != 0:
        raise ValueError("Override list has odd length: {}; it must be a list of pairs".format(len(opt)))

    for full_key, v in zip(opt[0::2], opt[1::2]):
        if not hasattr(args, full_key):
            raise AttributeError(f"Attribute '{full_key}' does not exist in args.")
        setattr(args, full_key, v)

    return args

def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    :param path: the complete path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)


def set_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad


def ini_client_domain(rand_domain_select, domains_list, parti_num):
    domains_len = len(domains_list)

    if rand_domain_select:

        max_num = 10
        is_ok = False
        while not is_ok:
            selected_domain_list = np.random.choice(domains_list, size=parti_num - domains_len, replace=True, p=None)
            selected_domain_list = list(selected_domain_list) + domains_list
            selected_domain_list = list(selected_domain_list)
            result = dict(Counter(selected_domain_list))
            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True
    else:
        selected_domain_dict = {}
        for domain in domains_list:
            selected_domain_dict[domain] = parti_num // domains_len
        # selected_domain_dict = {'MNIST': 6, 'USPS': 4, 'SVHN': 3, 'SYN': 7}  # base
        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)
        selected_domain_list = np.random.permutation(selected_domain_list)
    result = Counter(selected_domain_list)
    print(log_msg(selected_domain_list))
    print(log_msg(result))
    return selected_domain_list


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "TEST": 31,
        "ROBUST": 33,
        "OOD": 34,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def cal_client_weight(online_clients_list, client_domain_list, freq):
    client_weight = {}
    for index, item in enumerate(online_clients_list):  # 遍历循环当前的参与者
        client_domain = client_domain_list[item]
        client_freq = freq[index]
        client_weight[str(item) + ':' + client_domain] = round(client_freq, 3)
    return client_weight


def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = row[offset:offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size
