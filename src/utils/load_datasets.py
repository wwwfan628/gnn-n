from ogb.nodeproppred import DglNodePropPredDataset
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data.gnn_benckmark import Coauthor, AmazonCoBuy
from dgl.data import TUDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import dgl
import os
import yaml


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_citation(args):
    if args.dataset == 'cora':
        dataset = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        dataset = PubmedGraphDataset()
    g = dataset[0]
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)
    train_mask = g.ndata['train_mask'].to(device)
    valid_mask = g.ndata['val_mask'].to(device)
    test_mask = g.ndata['test_mask'].to(device)
    return g, features, labels, train_mask, valid_mask, test_mask


def index_to_mask(index_set, size):
    mask = torch.zeros(size, dtype=torch.bool)
    index = np.array(list(index_set))
    mask[index] = 1
    return mask


def load_copurchase_coauthors(args):
    # coauthor/co-purchase splits:
    # 20 per class for training
    # 30 per classes for validation
    # rest labels for testing
    if 'amazon' in args.dataset:
        name = args.dataset.split('_')[-1]
        dataset = AmazonCoBuy(name)
    elif 'coauthors' in args.dataset:
        name = args.dataset.split('_')[-1]
        dataset = Coauthor(name)
    g = dataset.data[0]
    features = torch.FloatTensor(g.ndata['feat'])
    features = (features.t() / torch.sum(features, dim=1)).t()  # normalize so that sum of each row equals 1
    labels = torch.LongTensor(g.ndata['label'])

    indices = []
    num_classes = torch.max(labels).item() + 1
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    test_index = torch.cat([i[50:] for i in indices], dim=0)
    test_index = test_index[torch.randperm(test_index.size(0))]

    train_mask = index_to_mask(train_index, size=g.number_of_nodes())
    val_mask = index_to_mask(val_index, size=g.number_of_nodes())
    test_mask = index_to_mask(test_index, size=g.number_of_nodes())
    return g, features, labels, train_mask, val_mask, test_mask


def load_ogbn(args):
    dataset = DglNodePropPredDataset(name=args.dataset)
    split_idx = dataset.get_idx_split()
    if args.dataset == 'ogbn-mag':
        train_index, valid_index, test_index = split_idx["train"]['paper'], split_idx["valid"]['paper'], split_idx["test"]['paper']
        g, labels = dataset[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        labels = labels['paper']
        labels = torch.squeeze(labels)  # squeeze tensor to one dimension
        features = torch.FloatTensor(g.ndata['feat']['paper'])

        train_mask = index_to_mask(train_index, size=g.num_nodes('paper'))
        val_mask = index_to_mask(valid_index, size=g.num_nodes('paper'))
        test_mask = index_to_mask(test_index, size=g.num_nodes('paper'))
    else:
        train_index, valid_index, test_index = split_idx["train"], split_idx["valid"], split_idx["test"]
        g, labels = dataset[0]
        labels = torch.squeeze(labels)    # squeeze tensor to one dimension
        features = torch.FloatTensor(g.ndata['feat'])

        train_mask = index_to_mask(train_index, size=g.number_of_nodes())
        val_mask = index_to_mask(valid_index, size=g.number_of_nodes())
        test_mask = index_to_mask(test_index, size=g.number_of_nodes())
    return g, features, labels, train_mask, val_mask, test_mask


def collate_tu(batch):
    graphs, labels = map(list, zip(*batch))
    # batch graphs and cast to PyTorch tensor
    for graph in graphs:
        for (key, value) in graph.ndata.items():
            graph.ndata[key] = torch.FloatTensor(value.float()).to(device)
        for (key, value) in graph.edata.items():
            graph.edata[key] = torch.FloatTensor(value.float()).to(device)
    batched_graphs = dgl.batch(graphs)
    # cast to PyTorch tensor
    batched_labels = torch.LongTensor(np.array(labels).flatten()).to(device)
    return batched_graphs, batched_labels


def load_tu(args):
    # seperate dataset name
    dataset_name = args.dataset.split('_')[-1]
    dataset_name = dataset_name.capitalize()
    # read ratio of training,validation and test set from configuration file
    path = '../configs/config.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    train_ratio = config['tu']['train_ratio']
    validate_ratio = config['tu']['validate_ratio']
    batch_size = config['tu']['batch_size']

    dataset = TUDataset(name=dataset_name)    # load dataset
    # Use provided node feature by default. If no feature provided, use node label instead.
    # If neither labels provided, use degree for node feature.
    for g in dataset.graph_lists:
        if 'node_attr' in g.ndata.keys():
            g.ndata['feat'] = g.ndata['node_attr']
        elif 'node_labels' in g.ndata.keys():
            g.ndata['feat'] = g.ndata['node_labels']
        else:
            g.ndata['feat'] = g.in_degrees().view(-1, 1).float()

    # print information of node features
    if 'node_attr' in g.ndata.keys():
        print("Use default node feature!")
    elif 'node_labels' in g.ndata.keys():
        print("Use node labels as node features!")
    else:
        print("Use node degree as node features!")

    statistics = dataset.statistics()  # includes input/output feature size

    train_size = int(train_ratio * len(dataset))
    valid_size = int(validate_ratio * len(dataset))
    test_size = int(len(dataset) - train_size - valid_size)

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_size, valid_size, test_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_tu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tu)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tu)
    # output dataset so that feature modification can be made directly on the whole dataset
    return statistics, dataset, train_dataloader, valid_dataloader, test_dataloader


def load_ogbg(args):
    # read ratio of training,validation and test set from configuration file
    path = '../configs/config.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    batch_size = config['ogbg']['batch_size']

    dataset = DglGraphPropPredDataset(name=args.dataset)

    statistics = np.zeros(2, dtype=int)   # statistics[0]:input feature size   statistics[1]:number of classes
    # gather info to compute statistics of dataset
    statistics[0] = dataset.graphs[0].ndata['feat'].shape[1]
    statistics[1] = dataset.num_classes

    split_idx = dataset.get_idx_split()

    train_dataloader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, collate_fn=collate_dgl)
    valid_dataloader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False, collate_fn=collate_dgl)
    test_dataloader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False, collate_fn=collate_dgl)
    # output dataset so that feature modification can be made directly on the whole dataset
    return statistics, dataset, train_dataloader, valid_dataloader, test_dataloader

def load_dataset(args):

    if args.dataset in 'cora, citeseer, pubmed':
        g, features, labels, train_mask, valid_mask, test_mask = load_citation(args)
        return g, features, labels, train_mask, valid_mask, test_mask
    elif args.dataset in 'amazon_photo, amazon_computers, coauthors_cs, coauthors_physics':
        g, features, labels, train_mask, valid_mask, test_mask = load_copurchase_coauthors(args)
        return g, features, labels, train_mask, valid_mask, test_mask
    elif 'ogbn' in args.dataset:
        g, features, labels, train_mask, valid_mask, test_mask = load_ogbn(args)
        return g, features, labels, train_mask, valid_mask, test_mask

    elif 'tu' in args.dataset:
        statistics, dataset, train_dataloader, valid_dataloader, test_dataloader = load_tu(args)
        return statistics, dataset, train_dataloader, valid_dataloader, test_dataloader

    elif 'ogbg' in args.dataset:
        statistics, dataset, train_dataloader, valid_dataloader, test_dataloader = load_ogbg(args)
        return statistics, dataset, train_dataloader, valid_dataloader, test_dataloader