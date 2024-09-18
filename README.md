# Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning

> Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning,            
> Wenke Huang, Mang Ye, Zekun Shi, Bo Du, Dacheng Tao
> *ECCV, 2024*
> [Link]()

## News
* [2024-09-18] Repo creation and code release. 

## Abstract
Federated learning presents massive potential for privacy- friendly vision task collaboration. However, the federated visual performance is deeply affected by backdoor attacks, where malicious clients optimize on triggered samples to mislead the global model into targeted mispredictions. 
Existing backdoor defensive solutions are normally based on two assumptions: data homogeneity and minority malicious ratio for the elaborate client-wise defensive rules. 
To address existing limitations, we argue that heterogeneous clients and backdoor attackers both bring divergent optimization directions and thus it is hard to discriminate them precisely. 
In this paper, we argue that parameters appear in different important degrees towards distinct distribution and instead consider meaningful and meaningless parameters for the ideal target distribution. 
We propose the Self-Driven Fisher Calibration (SDFC), which utilizes the Fisher Information to calculate the parameter importance degree for the local agnostic and global validation distribution and regulate those elements with large important differences. 
Furthermore, we allocate high aggregation weight for clients with relatively small overall parameter differences, which encourages clients with close local distribution to the global distribution, to contribute more to the federation. 
This endows SDFC to handle backdoor attackers in heterogeneous federated learning. Various vision task performances demonstrate the effectiveness of SDFC.

## Citation
```
@inproceedings{SDFC_ECCV24,
    title    = {Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning},
    author    = {Huang, Wenke and Ye, Mang and Shi, Zekun and Du, Bo and Tao, Dacheng},
    booktitle = {ECCV},
    year      = {2024}
}
```

## Relevant Projects
[3] Rethinking Federated Learning with Domain Shift: A Prototype View - CVPR 2023 [[Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf)][[Code](https://github.com/WenkeHuang/RethinkFL)]

[2] Federated Graph Semantic and Structural Learning - IJCAI 2023 [[Link](https://marswhu.github.io/publications/files/FGSSL.pdf)][[Code](https://github.com/wgc-research/fgssl)]

[1] Learn from Others and Be Yourself in Heterogeneous Federated Learning - CVPR 2022 [[Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf)][[Code](https://github.com/WenkeHuang/FCCL)]
