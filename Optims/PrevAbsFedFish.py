from Optims.utils.federated_optim import FederatedOptim
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import copy
import torch

class PrevAbsFedFish(FederatedOptim):
    NAME = 'PrevAbsFedFish'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(PrevAbsFedFish, self).__init__(nets_list, client_domain_list, args, cfg)
        self.prev_nets_list = []
        self.local_fish_dict = {}
        self.val_fish_dict = {}
        self.lamda = cfg.Optim[self.NAME].lamda

    def ini(self):
        for j in range(self.cfg.DATASET.parti_num):
            self.prev_nets_list.append(copy.deepcopy(self.nets_list[j]))
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for index, _ in enumerate(self.nets_list):
            self.nets_list[index].load_state_dict(global_w)
            self.prev_nets_list[index].load_state_dict(global_w)

    def loc_update(self,priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))
        # self.online_clients_list = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients_list = total_clients
        if self.epoch_index ==0:
            for i in self.online_clients_list:
                self.local_fish_dict[i]=self.fish_calculate(self.prev_nets_list[i],priloader_list[i])
                self.val_fish_dict[i]=self.fish_calculate(self.prev_nets_list[i],self.val_loader)
                self.fish_diff_dict[i]=self.fish_diff_calculate(self.local_fish_dict[i], self.val_fish_dict[i])
        for i in self.online_clients_list:
            self._train_net(i, self.nets_list[i], priloader_list[i], self.fish_diff_dict[i],self.prev_nets_list[i])

        self.copy_nets2_prevnets()

        for i in self.online_clients_list:
            self.local_fish_dict[i] = self.fish_calculate(self.prev_nets_list[i], priloader_list[i])
            self.val_fish_dict[i] = self.fish_calculate(self.prev_nets_list[i], self.val_loader)
            self.fish_diff_dict[i] = self.fish_diff_calculate(self.local_fish_dict[i], self.val_fish_dict[i])
        return  None
    def fish_diff_calculate(self, local_fish, val_fish):
        fish_diff = {}
        for para_name, _ in local_fish.items():
            local_fish_item = local_fish[para_name]
            val_fish_item = val_fish[para_name]
            fish_diff_item = torch.abs(local_fish_item - val_fish_item)
            fish_diff[para_name] = fish_diff_item
        return fish_diff
    
    def _train_net(self, index, net, train_loader,fish_diff,prev_net):

        net = net.to(self.device)
        prev_net = prev_net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9,weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))

        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                losses = []
                for para_name, fish_diff_item in fish_diff.items():
                    prev_parameter = dict(prev_net.named_parameters())[para_name]
                    curr_parameter = dict(net.named_parameters())[para_name]
                    fish_diff_item = fish_diff[para_name]
                    loss_consolidation = (fish_diff_item * (curr_parameter - prev_parameter) ** 2).sum()
                    losses.append(loss_consolidation)
                loss_reg = sum(losses)
                loss += self.lamda * loss_reg
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
                optimizer.step()

    def fish_calculate(self,net,loader):
        average_grads = {name: torch.zeros_like(param) for name, param in
                         net.named_parameters()}
        num_batches = 0
        for step, (image, label) in enumerate(loader):
            num_batches += 1
            image, label = image.to(self.device), label.to(self.device)
            output = net(image)
            pre = F.log_softmax(output, dim=1)
            log_liklihoods = (pre[:, label])
            net.zero_grad()
            log_liklihoods.mean().backward(retain_graph=True)
            for name, param in net.named_parameters():
                if param.grad is not None:
                    average_grads[name] += param.grad

            net.zero_grad()

        for name in average_grads:
            average_grads[name] /= num_batches
        fish_dict = {}
        for name, grad in average_grads.items():
            fish_dict[name] = average_grads[name].clone() ** 2

        return fish_dict
