import torch
import time
import numpy as np
import torch.nn as nn
import yaml
import os
from ogb.graphproppred import Evaluator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def classify_graph_mlp(test_dataloader, model):
    correctly_classified_graph_mask = []
    for batch_idx, (batch_graph, graph_labels) in enumerate(test_dataloader):
        model.eval()
        with torch.no_grad():
            logits = model(batch_graph, batch_graph.ndata['feat'].float())
            ypred = torch.argmax(logits, dim=1)
            mask = np.zeros(len(graph_labels))
            mask[ypred == graph_labels.squeeze()] = 1
            correctly_classified_graph_mask.append(mask)

    correctly_classified_graph_mask = np.concatenate(correctly_classified_graph_mask, axis=0)
    return correctly_classified_graph_mask


def evaluate_mlp_ogbg_mol(valid_dataloader, model, loss_fcn, args):
    evaluator = Evaluator(name=args.dataset)
    val_loss_list = []
    y_true = []
    y_pred = []
    for batch_idx, (batch_graph, graph_labels) in enumerate(valid_dataloader):
        model.eval()
        with torch.no_grad():
            logits = model(batch_graph, batch_graph.ndata['feat'].float())
            ypred = torch.argmax(logits, dim=1)
            logits_true = torch.zeros(logits.shape)
            for i, j in enumerate(graph_labels):
                logits_true[i, j] = 1
            loss = loss_fcn(logits, logits_true).item()
        val_loss_list.append(loss)
        y_true.append(graph_labels.detach())
        y_pred.append(ypred.unsqueeze(dim=1).detach())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    mean_val_loss = np.array(val_loss_list).mean()
    metric_value = evaluator.eval({"y_true": y_true, "y_pred": y_pred})['rocauc']
    return metric_value, mean_val_loss



def evaluate_mlp_tu(valid_dataloader, model, loss_fcn, batch_size):
    val_loss_list = []
    correct_label = 0
    for batch_idx, (batch_graph, graph_labels) in enumerate(valid_dataloader):
        model.eval()
        with torch.no_grad():
            logits = model(batch_graph, batch_graph.ndata['feat'])
            loss = loss_fcn(logits, graph_labels).item()
            ypred = torch.argmax(logits, dim=1)
            correct = torch.sum(ypred == graph_labels)
            correct_label += correct.item()
        val_loss_list.append(loss)
    mean_val_loss = np.array(val_loss_list).mean()
    acc = correct_label / (len(valid_dataloader) * batch_size)
    return acc, mean_val_loss


def train_mlp_4graph(model, train_dataloader, valid_dataloader, args):

    path = '../configs/config.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['mlp']['train_lr']  # learning rate
    max_epoch = config['mlp']['train_max_epoch']  # maximal number of training epochs
    clip = config['mlp']['clip']
    if 'tu' in args.dataset:
        batch_size = config['tu']['batch_size']
    elif 'ogbg' in args.dataset:
        batch_size = config['ogbg']['batch_size']

    # used for early stop
    patience = config['mlp']['train_patience']
    best_acc = -1
    best_loss = float('inf')
    cur_step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dur = []

    if 'tu' in args.dataset:
        loss_fcn = nn.CrossEntropyLoss()
    elif args.dataset == 'ogbg-molhiv':
        loss_fcn = nn.BCEWithLogitsLoss()

    for epoch in range(max_epoch):
        t0 = time.time()

        model.train()
        loss_list = []
        for (batch_idx, (batch_graph, graph_labels)) in enumerate(train_dataloader):
            model.zero_grad()
            logits = model(batch_graph, batch_graph.ndata['feat'].float())
            if 'tu' in args.dataset:
                loss = loss_fcn(logits, graph_labels)
            elif args.dataset == 'ogbg-molhiv':
                logits_true = torch.zeros(logits.shape)
                for i,j in enumerate(graph_labels):
                    logits_true[i,j] = 1
                loss = loss_fcn(logits, logits_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()

        dur.append(time.time() - t0)
        if 'tu' in args.dataset:
            metric_value, valid_loss = evaluate_mlp_tu(valid_dataloader, model, loss_fcn, batch_size)
        elif args.dataset == 'ogbg-molhiv':
            metric_value, valid_loss = evaluate_mlp_ogbg_mol(valid_dataloader, model, loss_fcn, args)
        print("Epoch {:04d} | Train Loss {:.4f} | Valid Loss {:.4f} | Valid Metric Value {:.4f} | Time(s) {:.4f}".format(epoch + 1, loss_data, valid_loss, metric_value, np.mean(dur)))
        # early stop
        if metric_value > best_acc or best_loss > valid_loss:
            best_acc = np.max((metric_value, best_acc))
            best_loss = np.min((best_loss, valid_loss))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break
    return best_acc