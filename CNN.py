
#imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms



#Create CNN
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10): # 28x28 image as input, 0-9 (10 classes) as output.
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #same conv
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) #halfs the dimension size.
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # same conv
        self.fc1 = nn.Linear(16*7*7, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# model = CNN()
# x = torch.rand(64, 1, 28, 28)
# print(x.shape)
# out = model(x)
# print(out[0])
# exit()

#set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#HyperParams
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

#Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

#init model

model = CNN().to(device)


#Loss nd optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)



#Train our network

for epoch in range(num_epochs):
    for batch_id, (batch, targets) in enumerate(train_loader):

        batch = batch.to(device = device)
        targets = targets.to(device = device)

        # batch = batch.reshape(batch.shape[0], -1) # flatten the input for our model...no need now, we did it before in forward!

        #forward
        scores = model(batch)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #Sg or adam step ( updating weights )
        optimizer.step()


#check accuracy on training nd test to see how good our model is

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  #to let the model know that this is evaluation mode which can impact how calculations are done

    with torch.no_grad():
         for x, y in loader:
             x = x.to(device=device)
             y = y.to(device = device)
             # x = x.reshape( x.shape[0], -1)
             scores = model(x)
            # (64, 10)
             _, predictions = scores.max(1) # index of maximum value for the 2nd dimension

             num_correct += (predictions == y).sum()
             num_samples += predictions.size(0)

         print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)