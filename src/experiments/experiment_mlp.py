from src.utils.load_datasets import load_dataset
from src.utils.train_4node import train_mlp_4node, classify_nodes_mlp
from src.utils.train_4graph import train_mlp_4graph, classify_graph_mlp
from src.models.mlp import MLP_4node, MLP_4graph

import argparse
import torch
import yaml
import os
import numpy as np
import pandas as pd
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):

    # check if 'outputs'&'checkpoint' directories exist, if not create
    outputs_dir = os.path.join(os.getcwd(), '../results')
    checkpoints_dir = os.path.join(os.getcwd(), '../checkpoints')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if not args.graph:    # node classification

        # load dataset
        random.seed(2020)  # fix random seed so that that the splitting of one dataset keeps the same
        np.random.seed(2020)
        torch.manual_seed(2020)
        torch.backends.cudnn.deterministic = True
        print("********** LOAD DATASET : {} **********".format(args.dataset.upper()))
        g, features, labels, train_mask, valid_mask, test_mask = load_dataset(args)

        # read parameters from config file
        path = '../configs/config.yaml'
        config_file = os.path.join(os.getcwd(), path)
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        h_feats = config['mlp']['hidden_features']
        in_feats = features.shape[1]
        out_feats = torch.max(labels).item() + 1

        # set random seed  for training (with the given parameter)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        print("********** BUILD MLP **********")
        # build network
        mlp = MLP_4node(args.depth, in_feats, h_feats, out_feats).to(device)

        print("********** TRAIN MLP **********")
        # train network
        _ = train_mlp_4node(mlp, features, labels, train_mask, valid_mask, args)

        print("********** SAVE CHECKPOINT **********")
        checkpoint_path = str(args.depth) + 'layerMLP_' + args.dataset + '_randomseed'+ str(args.random_seed) + '.pkl'
        checkpoint_file = os.path.join(checkpoints_dir, checkpoint_path)
        torch.save(mlp.state_dict(), checkpoint_file)

        print("********** TEST MLP **********")
        # test with original features
        correctly_classified_nodes_mask = classify_nodes_mlp(mlp, features, labels, test_mask)

        print("********** SAVE RESULTS **********")
        # save results
        result_df = pd.DataFrame({'random_seed': args.random_seed,
                                  'model': 'MLP'}, index=[0])

        test_nodes_index = np.arange(g.number_of_nodes())[test_mask]
        for i,j in enumerate(test_nodes_index):     # column name of dataframe: test node id in whole dataset -> j
            result_df[str(j)] = correctly_classified_nodes_mask[i]

        # check if the file exists
        outputs_file = os.path.join(outputs_dir, args.dataset+'_results.csv')
        if os.path.exists(outputs_file):
            result_df.to_csv(outputs_file, mode='a', header=False, index=False)
        else:
            result_df.to_csv(outputs_file,index=False)

    else:        # graph classification
        # load dataset
        random.seed(2020)  # fix random seed so that that the splitting of one dataset keeps the same
        np.random.seed(2020)
        torch.manual_seed(2020)
        torch.backends.cudnn.deterministic = True
        print("********** LOAD DATASET : {} **********".format(args.dataset.upper()))
        statistics, dataset, train_dataloader, valid_dataloader, test_dataloader = load_dataset(args)

        # read parameters from config file
        path = '../configs/config.yaml'
        config_file = os.path.join(os.getcwd(), path)
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        h_feats = config['mlp']['hidden_features']
        in_feats = statistics[0]
        out_feats = statistics[1].item()

        # set random seed  for training (with the given parameter)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        print("********** BUILD MLP **********")
        # build network
        mlp = MLP_4graph(args.depth, in_feats, h_feats, out_feats).to(device)

        print("********** TRAIN MLP **********")
        # train network
        _ = train_mlp_4graph(mlp, train_dataloader, valid_dataloader, args)

        print("********** SAVE CHECKPOINT **********")
        checkpoint_path = str(args.depth) + 'layerMLP_' + args.dataset + '_randomseed' + str(args.random_seed) + '.pkl'
        checkpoint_file = os.path.join(checkpoints_dir, checkpoint_path)
        torch.save(mlp.state_dict(), checkpoint_file)

        print("********** TEST MLP **********")
        # test with original features
        correctly_classified_graph_mask = classify_graph_mlp(test_dataloader, mlp)

        print("********** SAVE RESULTS **********")
        # save results
        result_df = pd.DataFrame({'dataset_name': args.dataset,
                                  'random_seed': args.random_seed,
                                  'model': 'MLP'}, index=[0])

        # compute testset size, batch size of last batch may smaller than others
        num_graphs_in_testset = 0
        for _,( _ ,graph_labels) in enumerate(test_dataloader):
            num_graphs_in_testset += len(graph_labels)
        for i in np.arange(num_graphs_in_testset):
            result_df[str(i)] = correctly_classified_graph_mask[i]

        # check if the file exists
        outputs_file = os.path.join(outputs_dir, args.dataset+'_results.csv')
        if os.path.exists(outputs_file):
            result_df.to_csv(outputs_file, mode='a', header=False, index=False)
        else:
            result_df.to_csv(outputs_file, index=False)



if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="GNN-N: 100-layer GCN")

    parser.add_argument('--dataset', default='cora', help='dataset names')
    parser.add_argument('--graph', action='store_true', help='graph classification ?')
    parser.add_argument('--depth', type=int, default=3, help='how many layers does MLP have')
    parser.add_argument('--random_seed', default=0, type=int)
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")