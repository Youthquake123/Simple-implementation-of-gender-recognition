import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets

# Parameter Settings
epochs_num = 10  # Number of training iterations
batch_size = 32  # Batch size
learning_rate = 0.0001  # Learning rate
momentum = 0.9  # Momentum method to update parameters
use_gpu = 0  # Whether to use GPU-accelerated training
is_train = 0  # Choose between training the model or directly testing the existing model

# Define Converter
transform = transforms.Compose([
    transforms.CenterCrop(200),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Define the model
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)  # Convolution layer
        self.bn1 = nn.BatchNorm2d(16)  # Standardize input operations on each small batch of data
        self.relu1 = nn.ReLU()  # Activation function
        self.pool1 = nn.MaxPool2d(2, 2)  # Maximum pooling layer

        self.conv2 = nn.Conv2d(16, 32, 3)  # Convolution layer
        self.bn2 = nn.BatchNorm2d(32)  # The input data is more dispersed
        self.relu2 = nn.ReLU()  # Activation function
        self.pool2 = nn.MaxPool2d(2, 2)  # Maximum pooling layer

        self.fc1 = nn.Linear(32 * 23 * 23, 120)  # Full connection layer with 32 inputs and 120 outputs
        self.bn3 = nn.BatchNorm1d(120)  # Standardized operation
        self.relu3 = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(0.5)  # Randomly discard neurons in the neural network with a certain probability

        self.fc2 = nn.Linear(120, 2)  # Fully connected layer, classifier effect
        self.softmax2 = nn.Softmax(dim=1)  # Output normalization

    # Forward propagation function
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 23 * 23)
        x = self.relu3(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax2(x)
        return x


# Initialize the model
net = GenderClassifier()

if is_train:
    # Define loss functions and optimizers
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr, momentum)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Load data
    '''When data is loaded, the ImageFolder class automatically labels subfolders in the folder 
    and pairs the path of each image with its corresponding label to form the training dataset'''
    train_set = datasets.ImageFolder('train', transform=transform)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    # Output the category name and corresponding label
    class_to_idx = train_set.class_to_idx
    for class_name, class_label in class_to_idx.items():
        print('The label of ' + class_name + ' : ' + str(class_label))

    start = time.time()
    counter = []
    counter_temp = 0
    loss_history = []
    correct_cnt = 0
    record_interval = 70
    # Train the network multiple iterations
    for epoch in range(1, epochs_num + 1):
        for i, data in enumerate(train_loader, 0):
            img, label = data
            if use_gpu:  # GPU acceleration
                img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()  # Clear the gradient of optimizer
            output = net(img)  # Forward propagation calculates iterative results
            loss = criterion(output, label)  # Calculate the loss function
            loss.backward()  # Back propagation
            optimizer.step()  # Parameter update

            _, predict = torch.max(output, 1)
            correct_cnt += (predict == label).sum()  # Compare the predicted value with the actual value

            # Store loss values and accuracy
            if i % record_interval == record_interval - 1:
                counter_temp += record_interval * batch_size
                counter.append(counter_temp)
                loss_history.append(loss.item())
        print("Iterator times : {}\n Loss : {}\n".format(epoch, loss.item()))
    end = time.time()
    counter = [element / epochs_num for element in counter]
    print('Time : {:.3f}'.format(end - start) + ' s')

    # Draw loss image
    plt.figure()
    plt.plot(counter, loss_history)
    plt.title('Loss Curve')
    plt.xlabel('Data')
    plt.ylabel('Loss')
    plt.show()

    state = {'net': net.state_dict()}
    torch.save(net.state_dict(), '.\modelpara.pth')
    print('Done')

else:
    # Choose whether to use GPU
    if use_gpu:
        net.load_state_dict(torch.load('.\modelpara.pth'))
    else:
        net.load_state_dict(torch.load('.\modelpara.pth', map_location='cpu'))

    # Load data
    test_set = datasets.ImageFolder('test', transform=transform)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    start = time.time()
    counter = []
    counter_temp = 0
    correct_history = []
    correct_cnt = 0
    record_interval = 1
    # Test the model
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            img, label = data
            if use_gpu:
                img, label = img.cuda(), label.cuda()
            output = net(img)

            _, predict = torch.max(output, 1)
            correct_cnt += (predict == label).sum()  # Compare the predicted value with the actual value

            if i % record_interval == record_interval - 1:
                counter_temp += record_interval * batch_size
                counter.append(counter_temp)
                correct_history.append(correct_cnt.item() / (record_interval * batch_size))
                correct_cnt = 0
    end = time.time()

    # Draw accuracy images
    plt.figure()
    plt.plot(counter, correct_history)
    plt.title('Test_acc Curve')
    plt.xlabel('Data')
    plt.ylabel('Accuracy')
    plt.show()

    # Accuracy of test results
    print('Test_acc : {:.4f}'.format(float(correct_history[-1])))
    print('Time : {:.3f} s'.format(end - start))
