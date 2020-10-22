import torch
import time
import numpy as np
import torch.nn.functional as F
import yaml
import os
from ogb.nodeproppred import Evaluator


def evaluate_mlp_ogbn(model, features, labels, mask, args):
    evaluator = Evaluator(name= args.dataset)
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss_test = F.nll_loss(logp[mask], labels[mask])
        _, indices = torch.max(logits, dim=1)
        acc = evaluator.eval({'y_true': labels[mask].unsqueeze(1),'y_pred': indices[mask].unsqueeze(1)})['acc']
    return acc, loss_test


def evaluate_mlp_4node(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss_test = F.nll_loss(logp[mask], labels[mask])
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices[mask] == labels[mask])
        acc = correct.item() * 1.0 / len(labels[mask])
    return acc, loss_test


def classify_nodes_mlp(model, features, labels, mask):
    num_nodes_in_testset = torch.sum(mask==1)
    model.eval()
    with torch.no_grad():
        logits = model(features)
        _, indices = torch.max(logits, dim=1)
        correctly_classified_nodes_mask = np.zeros(num_nodes_in_testset)
        correctly_classified_nodes_mask[indices[mask] == labels[mask]] = 1
    return correctly_classified_nodes_mask


def train_mlp_4node(model, features, labels, train_mask, valid_mask, args):
    path = '../configs/config.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['mlp']['train_lr']  # learning rate
    max_epoch = config['mlp']['train_max_epoch']  # maximal number of training epochs
    # used for early stop
    patience = config['mlp']['train_patience']
    best_acc = -1
    best_loss = float('inf')
    cur_step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        model.train()
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)
        if 'ogbn' in args.dataset:
            acc, loss_valid = evaluate_mlp_ogbn(model, features, labels, valid_mask, args)
        else:
            acc, loss_valid = evaluate_mlp_4node(model, features, labels, valid_mask)
        print("Epoch {:04d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(epoch + 1, loss.item(), acc, np.mean(dur)))

        # early stop
        if acc > best_acc or best_loss > loss_valid:
            best_acc = np.max((acc, best_acc))
            best_loss = np.min((best_loss, loss_valid))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break
    return best_acc