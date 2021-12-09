import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from util import load_data, load_hyper_data, separate_data
from models.graphcnn import GraphCNN
import matplotlib.pyplot as plt

SCALE=1
criterion = nn.CrossEntropyLoss()


def train(args, model, device, train_graphs, hyper_train_graphs, train_motif2A, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        hyper_batch_graph = [hyper_train_graphs[idx] for idx in selected_idx]
        batch_motif2A = [train_motif2A[idx] for idx in selected_idx]

        output = model(batch_graph, hyper_batch_graph, batch_motif2A)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, hyper_graphs, test_motif2A, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx], [hyper_graphs[j] for j in sampled_idx],
                            [test_motif2A[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


def test(args, model, device, train_graphs, hyper_train_graph, train_motif2A,
         test_graphs, hyper_test_graphs, test_motif2A, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs, hyper_train_graph, train_motif2A)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs, hyper_test_graphs, test_motif2A)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    return acc_train, acc_test


def main(dataset, epoch=800, num_layers=5, num_mlp_layers=3, lr=0.01, batch_size=32, filename='', degree_as_tag=False):
    # Training settings

    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default=dataset,
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=epoch,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=lr,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=10,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=num_layers,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=num_mlp_layers,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true", default=degree_as_tag,
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    args = parser.parse_args()

    check_path = os.path.join('checkpoints', dataset)
    if not os.path.exists(check_path):
        os.mkdir(check_path)
    if filename == '':
        filename = os.path.join(check_path, time.strftime("%m%d_%H%M"))

    filepath = os.path.join(filename, 'params.txt')
    if not os.path.exists(filename):
        os.mkdir(filename)
    with open(filepath, 'w') as f:
        f.write(str(vars(args)))
        f.close()

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes, motf2As, len_tagset = load_data(args.dataset, degree_as_tag,SCALE)
    hyper_graphs, _ = load_hyper_data(args.dataset, 0, SCALE)

    max_accs = []
    for i in range(args.fold_idx):
        ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
        train_graphs, hyper_train_graphs, train_motif2A, test_graphs, hyper_test_graphs, test_motif2A \
            = separate_data(graphs, hyper_graphs, motf2As, args.seed, i)

        model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1],
                         args.hidden_dim, num_classes, args.final_dropout, args.learn_eps,
                         args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        max_acc_test = 0
        avg_losses, acc_trains, acc_tests = [], [], []
        for epoch in range(1, args.epochs + 1):
            scheduler.step()

            start_time = time.time()
            avg_loss = train(args, model, device, train_graphs, hyper_train_graphs, train_motif2A, optimizer, epoch)
            train_time = time.time() - start_time

            start_time = time.time()
            acc_train, acc_test = test(args, model, device, train_graphs, hyper_train_graphs, train_motif2A,
                                       test_graphs, hyper_test_graphs, test_motif2A, epoch)
            test_time = time.time() - start_time

            if acc_test > max_acc_test:
                max_acc_test = acc_test

            avg_losses.append(avg_loss)
            acc_trains.append(acc_train)
            acc_tests.append(acc_test)
            print("fold: %f accuracy train: %f test: %f max_test: %f train_time: %f test_time: %f" % (
            i, acc_train, acc_test, max_acc_test, train_time, test_time))

        if not filename == "":
            filepath = os.path.join(filename, str(i) + '.txt')
            if not os.path.exists(filename):
                os.mkdir(filename)
            with open(filepath, 'w') as f:
                for epoch in range(1, args.epochs + 1):
                    f.write("%f %f %f %f %f %f" % (epoch, avg_losses[epoch - 1],
                                                   acc_trains[epoch - 1], acc_tests[epoch - 1], train_time, test_time))
                    f.write("\n")
                f.close()

        epoches = range(0, args.epochs)

        max_accs.append(max_acc_test)

        print(model.atts)

    max_accs = np.array(max_accs)
    print(max_accs)
    print(np.mean(max_accs), np.std(max_accs))
    return max_accs, np.mean(max_accs), np.std(max_accs)


if __name__ == '__main__':

    datasets = ['Synthetic']
    param_dicts = {
        'MUTAG': {'epoch': 3, 'num_layer': 4, 'num_mlp_layer': 1, 'lr': 0.01, 'batch_size': 32,
                  'degree_as_tag': False},
        'PTC_MR': {'epoch': 3, 'num_layer': 6, 'num_mlp_layer': 3, 'lr': 0.01, 'batch_size': 32,
                   'degree_as_tag': False},
        'PTC_MM': {'epoch': 400, 'num_layer': 5, 'num_mlp_layer': 3, 'lr': 0.01, 'batch_size': 32,
                   'degree_as_tag': False},
        'PTC_FR': {'epoch': 400, 'num_layer': 5, 'num_mlp_layer': 3, 'lr': 0.01, 'batch_size': 32,
                   'degree_as_tag': False},
        'PTC_FM': {'epoch': 400, 'num_layer': 5, 'num_mlp_layer': 3, 'lr': 0.01, 'batch_size': 32,
                   'degree_as_tag': False},
        'COX2': {'epoch': 3, 'num_layer': 4, 'num_mlp_layer': 2, 'lr': 1e-2, 'batch_size': 8, 'degree_as_tag': False},
        'NCI1': {'epoch': 3, 'num_layer': 6, 'num_mlp_layer': 3, 'lr': 0.01, 'batch_size': 32,
                 'degree_as_tag': False},
        'PROTEINS': {'epoch': 3, 'num_layer': 6, 'num_mlp_layer': 3, 'lr': 0.01, 'batch_size': 16,
                     'degree_as_tag': False},
        'IMDB-MULTI': {'epoch': 3, 'num_layer': 6, 'num_mlp_layer': 1, 'lr': 0.01, 'batch_size': 32,
                       'degree_as_tag': True},
        'IMDB-BINARY': {'epoch': 3, 'num_layer': 6, 'num_mlp_layer': 3, 'lr': 0.01, 'batch_size': 32,
                        'degree_as_tag': True},
        'Synthetic': {'epoch': 3, 'num_layer': 3, 'num_mlp_layer': 1, 'lr': 0.01, 'batch_size': 32,
                      'degree_as_tag': False}
    }

    for i in range(3,4):
        SCALE=i
        accs, mean_accs, std_accs = [], [], []

        for dataset in datasets:
            acc, mean_acc, std_acc = main(dataset, param_dicts[dataset]['epoch'],
                                          param_dicts[dataset]['num_layer'],
                                          param_dicts[dataset]['num_mlp_layer'],
                                          param_dicts[dataset]['lr'],
                                          param_dicts[dataset]['batch_size'],
                                          filename='',
                                          degree_as_tag=param_dicts[dataset]['degree_as_tag'])
            accs.append(acc)
            mean_accs.append(mean_acc)
            std_accs.append(std_acc)
            plt.close('all')

        for i in range(len(datasets)):
            print(accs[i], mean_accs[i], std_accs[i])




