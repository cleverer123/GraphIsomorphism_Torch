import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_proteins import load_data
from gin import GraphIsomorphismNetwork0
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def shuffle_data(x, y):
    zipped = list(zip(x, y))
    np.random.shuffle(zipped)
    x, y = zip(*zipped)
    return x, y


def preprocess(molecule_list):

    gin = GraphIsomorphismNetwork0(4, 20)
    processed_data = []

    for molecule in molecule_list:
        processed_data.append(gin.predict(molecule))
    
    max_attribute = 0# maximum value of attribute value 

    # 归一化
    for pdata in processed_data:
        pdata[0:3] /= pdata[0:3].max()
        
        max_attribute = max(max_attribute, pdata[3].cpu().numpy())

    for pdata in processed_data:
        pdata[3] /= torch.from_numpy(max_attribute)

    return processed_data

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fun0 = nn.Linear(4, 100)
        self.fun1 = nn.Linear(100, 50)
        self.fun2 = nn.Linear(50, 25)
        self.fun3 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.fun0(x)
        x = F.relu(x)
        x = self.fun1(x)
        x = F.relu(x)
        x = self.fun2(x)
        x = F.relu(x)
        x = self.fun3(x)
        x = F.sigmoid(x)
        return x

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MLP, self).__init__()
#         self.module = nn.Sequential(
# 			nn.Linear(input_dim, 100),
#             nn.ReLU(inplace=True),
# 			nn.Linear(100, 50),
#             nn.ReLU(inplace=True),
#             nn.Linear(50, 25),
#             nn.ReLU(inplace=True),
# 			nn.Linear(25, output_dim),
#             nn.Sigmoid()
# 		)
    
#     def forward(self, x):
#         return self.module(x)

def train(model, data, labels):
    model.train()
    total_loss = 0
    num_corrects = 0
    for i in range(len(data)):       
        optimizer.zero_grad()
        output = model(data[i])
    
        loss = criterion(output, labels[i])
        total_loss += loss

        if (output >= 0.5 and labels[i] == 1) or (output < 0.5 and labels[i] == 0):
            num_corrects += 1

        loss.backward()
        optimizer.step()

    return total_loss/len(data), 100 * num_corrects / len(data)
            
def eval(model, data, labels):
    model.eval()
    total_loss = 0
    num_corrects = 0
    for i in range(len(data)):       
        optimizer.zero_grad()
        prediction = model(data[i])
        loss = criterion(prediction, labels[i])
        total_loss += loss
        if (prediction >= 0.5 and labels[i] == 1) or (prediction < 0.5 and labels[i] == 0):
            num_corrects += 1

    
    return total_loss / len(data), 100 * num_corrects / len(data)

if __name__ == "__main__":
    
    molecule_list, label_list = load_data()
    print('---Data Loaded---')
    processed_data = preprocess(molecule_list)
    print('---Data Processed---')

    # split data into train data and test data
    shuffled_data, shuffled_labels = shuffle_data(processed_data, label_list)
    test_data = shuffled_data[0:200]
    test_labels = shuffled_labels[0:200]
    train_data = shuffled_data[200:]
    train_labels = shuffled_labels[200:]
    

    input_dim = 4
    hidden_dim = 100
    output_dim = 1

    device = torch.device('cuda')
    model = MLP().to(device)
    # model = MLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.BCELoss()#nn.MSELoss()
    # criterion = F.cross_entropy

    epoch_size = 100

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(epoch_size):
        shuffled_data, shuffled_labels = shuffle_data(train_data, train_labels)

        train_loss, train_acc = train(model, train_data, train_labels)        
        test_loss, test_acc = eval(model, test_data, test_labels)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)        
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print('Epoch [{}/{}], train_loss: {:.4f}, train_acc:{:.3f}, test_loss: {:.4f}, test_acc:{:.3f}'
                    .format(epoch + 1, epoch_size, train_loss, train_acc, test_loss, test_acc))

    print('Best test acc:', max(test_acc_list))

    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.plot(range(len(test_loss_list)), test_loss_list)
    plt.savefig('loss_1.jpg')

    plt.figure()
    plt.plot(range(len(train_acc_list)), train_acc_list)
    plt.plot(range(len(test_acc_list)), test_acc_list)
    plt.savefig('acc_1.jpg')

