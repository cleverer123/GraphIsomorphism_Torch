import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse 
from load_proteins import load_data_table, separate_data
from gin import GraphIsomorphismNetwork
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def train(args, model, train_graphs):
    model.train()
    total_loss = 0
    num_corrects = 0
    for i in range(args.iters_per_epoch):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]

        optimizer.zero_grad()
        
        output = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(args.device)
    
        loss = criterion(output, labels)

        total_loss += loss

        # if (output >= 0.5 and labels[i] == 1) or (output < 0.5 and labels[i] == 0):
        #     num_corrects += 1

        loss.backward()
        optimizer.step()

    return total_loss/args.iters_per_epoch

def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, train_graphs, test_graphs):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test       
# def eval(model, data, labels):
#     model.eval()
#     total_loss = 0
#     num_corrects = 0
#     for i in range(len(data)):       
#         optimizer.zero_grad()
#         prediction = model(data[i])
#         loss = criterion(prediction, labels[i])
#         total_loss += loss
#         if (prediction >= 0.5 and labels[i] == 1) or (prediction < 0.5 and labels[i] == 0):
#             num_corrects += 1

    
#     return total_loss / len(data), 100 * num_corrects / len(data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="PROTEINS",
                        help='name of dataset (default: PROTEINS)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
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
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                        help='output file')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0) 
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    graph_list, num_classes = load_data_table()

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graph_list, args.seed, args.fold_idx)

    model = GraphIsomorphismNetwork(
        args.num_layers, 
        args.num_mlp_layers, 
        train_graphs[0].node_features.shape[1], 
        args.hidden_dim, 
        num_classes, 
        args.final_dropout, 
        args.learn_eps, 
        args.graph_pooling_type, 
        args.neighbor_pooling_type, 
        device
        ).to(device)

    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.CrossEntropyLoss() #nn.MSELoss()
  

    train_loss_list = []
    # test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(args.epochs):
        # shuffled_data, shuffled_labels = shuffle_data(train_data, train_labels)

        train_loss = train(args, model, train_graphs)        
        train_acc, test_acc = test(args, model, train_graphs, test_graphs)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)        
        test_acc_list.append(test_acc)

        print('Epoch [{}/{}], train_loss: {:.4f}, train_acc:{:.3f}, test_acc:{:.3f}'
                    .format(epoch + 1, args.epochs, train_loss, train_acc, test_acc))

    print('Best test acc:', max(test_acc_list[50:]))

    plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
    # plt.plot(range(len(test_loss_list)), test_loss_list)
    plt.savefig('loss_1.jpg')

    plt.figure()
    plt.plot(range(len(train_acc_list)), train_acc_list, label='train_acc')
    plt.plot(range(len(test_acc_list)), test_acc_list, label='test_acc')
    plt.savefig('acc_1.jpg')

