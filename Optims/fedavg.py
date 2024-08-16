import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from Optims.utils.federated_optim import FederatedOptim

class FedAvG(FederatedOptim):
    NAME = 'FedAvG'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedAvG, self).__init__(nets_list, client_domain_list, args, cfg)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self,priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))
        self.online_clients_list  = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()

        for i in self.online_clients_list:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        return  None

    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
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
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index,loss)
                optimizer.step()
