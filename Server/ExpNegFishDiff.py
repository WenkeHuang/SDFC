from Server.utils.server_methods import ServerMethod
import torch
import numpy as np


class ExpNegFishDiff(ServerMethod):
    NAME = 'ExpNegFishDiff'

    def __init__(self, args, cfg):
        super(ExpNegFishDiff, self).__init__(args, cfg)


    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        # priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        fish_diff_dict = kwargs['fish_diff_dict']
        fish_diff_sum_dict = {}

        for _, net_id in enumerate(online_clients_list):
            fish_diff_sum = []
            fish_diff = fish_diff_dict[net_id]
            for para_name, fish_diff_item in fish_diff.items():
                fish_diff_sum.append(torch.sum(fish_diff_item))
            fish_diff_sum_dict[net_id] = torch.mean(torch.tensor(fish_diff_sum))

        freq = []
        for _, net_id in enumerate(online_clients_list):
            freq.append(fish_diff_sum_dict[net_id].cpu())

        new_freq = (freq - np.min(freq)) / (np.max(freq) - np.min(freq))
        new_freq = np.exp(-new_freq) / np.sum(np.exp(-new_freq))

        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=new_freq, except_part=[], global_only=False)
        return freq
