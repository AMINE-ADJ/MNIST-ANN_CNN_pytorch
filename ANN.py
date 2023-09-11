
#imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


#Create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes): # 28x28 image as input, 0-9 (10 classes) as output.
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# testInput = torch.rand(128, 784) 128 c'est le batch size...hna i ve just one batch.
# model = NN(784,10)
# predic = model(testInput)
# print(predic[0])
# print(testInput[0])

#set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#HyperParams
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

#Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# figure = plt.figure(figsize=(8, 8))
# img, label = train_dataset[2]
# plt.title(label)
# plt.axis("off")
# plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# print(img)
# print(label)

#init model

model = NN(input_size = input_size, num_classes = num_classes).to(device)


#Loss nd optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)



#Train our network

for epoch in range(num_epochs):
    for batch_id, (batch, targets) in enumerate(train_loader):

        batch = batch.to(device = device)
        targets = targets.to(device = device)

        batch = batch.reshape(batch.shape[0], -1) # flatten the input for our model.

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
             x = x.reshape( x.shape[0], -1)
             scores = model(x)
            # (64, 10)
             _, predictions = scores.max(1) # index of maximum value for the 2nd dimension

             num_correct += (predictions == y).sum()
             num_samples += predictions.size(0)

         print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)