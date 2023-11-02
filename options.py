import argparse
import json
import os

import torch

def args_parser():
    parser = argparse.ArgumentParser()
    #dataset and model
    parser.add_argument(
        '--dataset',
        type = str,
        default = 'femnist',
        help = 'name of the dataset: mnist, cifar10, femnist'
    )
    parser.add_argument(
        '--model',
        type = str,
        default = 'cnn',
        help='name of model. mnist: logistic, lenet; cifar10: resnet18, cnn_complex; femnist: lenet'
    )
    parser.add_argument(
        '--input_channels',
        type = int,
        default = 1,
        help = 'input channels. femnist、mnist:1, cifar10 :3'
    )
    parser.add_argument(
        '--output_channels',
        type = int,
        default = 62,
        help = 'output channels. femnist:62'
    )
    #nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 10,
        help = 'batch size when trained on client'
    )
    # -------------云聚合轮次、边缘聚合轮次、本地更新轮次
    parser.add_argument(
        '--num_communication',
        type = int,
        default=5,
        help = 'number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type = int,
        default=20,
        help = 'number of edge aggregation (K_2)'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=20,
        help='number of local update (K_1)'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.06,
        help = 'learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type = float,
        default= '1',
        help = 'lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type = int,
        default=1,
        help= 'lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type = float,
        default = 0,
        help = 'SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type = float,
        default = 0,
        help= 'The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type = int,
        default = 0,
        help = 'verbose for print progress bar'
    )
    #setting for federeated learning
    parser.add_argument(
        '--iid',
        type = int,
        default = 1,
        help = 'distribution of the data, 1,0,-1,-2(one-class)'
    )
    parser.add_argument(
        '--edgeiid',
        type=int,
        default = 1,
        help='distribution of the data under edges, 1 (edgeiid),0 (edgeniid) (used only when iid = -2)'
    )
    parser.add_argument(
        '--frac',
        type = float,
        default = 1,
        help = 'fraction of participated clients'
    )
    # -------------客户端数、边缘服务器数、客户端训练样本量
    parser.add_argument(
        '--num_clients',
        type = int,
        default = 20,
        help = 'number of all available clients'
    )
    parser.add_argument(
        '--num_edges',
        type = int,
        default= 2,
        help= 'number of edges'
    )
    parser.add_argument(
        '--num_sample_per_client',
        default= -1,
        type=int,
        help='>=0: number of samples per client， -1: all samples'
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 1,
        help = 'random seed (defaul: 1)'
    )

    # editer: Sensorjang 20230925
    dataset_root = os.path.join(os.getcwd(), 'train_data')
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    parser.add_argument(
        '--dataset_root',
        type = str,
        default = dataset_root,
        help = 'dataset root folder'
    )
    parser.add_argument(
        '--show_dis',
        type= int,
        default= 1,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--classes_per_client',
        type=int,
        default = 2,
        help='under artificial non-iid distribution, the classes per client'
    )
    parser.add_argument(
        '--gpu',
        type = int,
        default=0,
        help = 'GPU to be selected, 0, 1, 2, 3'
    )

    parser.add_argument(
        '--mtl_model',
        default=0,
        type = int
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )

    # editer: Sensorjang 20230925
    parser.add_argument(
        '--test_on_all_samples',
        type = int,
        default = 1,
        help = '1 means test on all samples, 0 means test samples will be split averagely to each client'
    )

    # editer: Sensorjang 20230925
    # 定义edges及其下属clients的映射关系
    parser.add_argument(
        '--active_mapping',
        type = int,
        default = 0,
        help = '1 means mapping is active, 0 means mapping is inactive'
    )
    mapping = {
        "0": [0, 1, 2, 3],
        "1": [4]
    }
    # 将映射关系转换为JSON格式
    mapping_json = json.dumps(mapping)
    parser.add_argument(
        '--mapping',
        type = str,
        default = mapping_json,
        help = 'mapping of edges and their clients'
    )


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
