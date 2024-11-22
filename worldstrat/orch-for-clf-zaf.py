import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import json
from torchvision.models import resnet18 as rnet18
# Define the path to your data folders
#orchard_path = "mountdata/euroorchardsSR"
#forest_path = "mountdata/euroforestsSR"

orchard_path = 'mountdata/zaforchardsSR-unnormalized'
forest_path = 'mountdata/zafforestsSR-unnormalized'

# Define hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 30

# Check if CUDA/GPU is available (optional but recommended for faster training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

anomalies = ['mountdata/zaforchards2-50m/Image_Lat_-22.240004789245813_Lon_29.037818903248997.tif', 'mountdata/zaforchards2-50m/Image_Lat_-23.190591509848193_Lon_30.02648230297659.tif', 'mountdata/zaforchards2-50m/Image_Lat_-23.206826468411446_Lon_30.02780007043587.tif', 'mountdata/zaforchards2-50m/Image_Lat_-23.729701304546758_Lon_30.581030198896638.tif',
             'mountdata/zaforchards2-50m/Image_Lat_-23.75421204450379_Lon_30.040246170137554.tif', 'mountdata/zaforchards2-50m/Image_Lat_-24.36739322183993_Lon_30.740867775354953.tif', 'mountdata/zaforchards2-50m/Image_Lat_-24.36793308158871_Lon_30.78216057550199.tif', 'mountdata/zaforchards2-50m/Image_Lat_-24.36912426208262_Lon_30.70107562081649.tif',
             'mountdata/zaforchards2-50m/Image_Lat_-24.369803520775992_Lon_30.706244153045976.tif',
             'mountdata/zaforchards2-50m/Image_Lat_-24.369990509567586_Lon_30.697764023623456.tif', 'mountdata/zaforchards2-50m/Image_Lat_-25.004242898951123_Lon_31.056674929441822.tif', 'mountdata/zaforchards2-50m/Image_Lat_-25.083577710663594_Lon_31.059082643615447.tif', 'mountdata/zaforchards2-50m/Image_Lat_-25.355981316561753_Lon_30.87019654213516.tif',
             'mountdata/zaforchards2-50m/Image_Lat_-25.356378638983212_Lon_30.749683721086058.tif', 'mountdata/zaforchards2-50m/Image_Lat_-25.399142856245106_Lon_31.853747261479064.tif', 'mountdata/zaforchards2-50m/Image_Lat_-25.42440065060146_Lon_31.10614799508328.tif', 'mountdata/zaforchards2-50m/Image_Lat_-32.15294327380358_Lon_18.886208968452273.tif',
             'mountdata/zaforchards2-50m/Image_Lat_-32.266605173370685_Lon_18.979176591008986.tif', 'mountdata/zaforchards2-50m/Image_Lat_-32.80791159156275_Lon_18.7056015650465.tif', 'mountdata/zaforchards2-50m/Image_Lat_-33.424104508473484_Lon_19.219134895787988.tif', 'mountdata/zaforchards2-50m/Image_Lat_-33.42747627667723_Lon_25.494554865159728.tif',
             'mountdata/zaforchards2-50m/Image_Lat_-33.76617526466205_Lon_23.486430250833436.tif', 'mountdata/zaforchards2-50m/Image_Lat_-33.81678984759524_Lon_23.75666510651492.tif', 'mountdata/zafforests2-50m/Image_Lat_-22.750359611743477_Lon_30.076954472051707.tif', 'mountdata/zafforests2-50m/Image_Lat_-24.893520791446683_Lon_31.057298004159495.tif', 'mountdata/zafforests2-50m/Image_Lat_-25.382506281880506_Lon_29.54393489111359.tif', 'mountdata/zafforests2-50m/Image_Lat_-25.38360708957488_Lon_29.41490838960512.tif',
 'mountdata/zafforests2-50m/Image_Lat_-25.663925262990297_Lon_31.10501855033152.tif', 'mountdata/zafforests2-50m/Image_Lat_-26.299960611462208_Lon_25.683737847721723.tif', 'mountdata/zafforests2-50m/Image_Lat_-27.963598186452764_Lon_32.084100119337684.tif',
             'mountdata/zafforests2-50m/Image_Lat_-28.315390968743124_Lon_32.45090360985396.tif', 'mountdata/zafforests2-50m/Image_Lat_-28.4597026176856_Lon_32.415726424788375.tif', 'mountdata/zafforests2-50m/Image_Lat_-28.937286098156267_Lon_30.947486221372245.tif', 'mountdata/zafforests2-50m/Image_Lat_-28.98745723719802_Lon_29.846632525779782.tif', 'mountdata/zafforests2-50m/Image_Lat_-29.803715878804166_Lon_30.26411985853742.tif', 'mountdata/zafforests2-50m/Image_Lat_-32.68005997185722_Lon_27.002089587012645.tif']

anoms = [x[9:] + '.npy' for x in anomalies]
print(anoms[0])
dropout_prob = .1
class ResModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResModel, self).__init__()
        
        # Load the pre-trained MobileNetV2 model
        resnet18 = rnet18(pretrained=True)
        
       # Remove the classification head of ResNet18
        self.features = nn.Sequential(*list(resnet18.children())[:-1])

        # Add custom layers for your task
#        self.fc = nn.Sequential(
#            nn.Flatten(),
#            nn.Linear(512, num_classes)
#        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),  # You can adjust the output size of the linear layer
            nn.ReLU(),           # Add activation function
            nn.Dropout(p=dropout_prob),  # Dropout layer
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        # Feature extraction with ResNet18
        x = self.features(x)

        # Fully connected layers
        x = self.fc(x)

        return x

model = ResModel().to(device)

shape = (160,160,3)

from torch.utils.data.sampler import SubsetRandomSampler

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.file_list[idx])
        if img_name in anoms:
          #return torch.zeros(50,50,3), -1
          return torch.zeros(160,160,3)
        image = np.squeeze(np.load(img_name, allow_pickle=True))
        #image = np.squeeze(np.load(img_name))[:,53:103,53:103]#.unsqueeze(0).unsqueeze(0)
        #print(image)
        #print(image.shape)
        #print(image)
        #if self.transform:
        #    image = self.transform(image)
        label = 0 if "orchard" in self.data_dir else 1  # Assuming "orchard" is class 0 and "forest" is class 1
        return image, label, img_name

# Data augmentation and normalization
transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create data loaders for training and validation
#orchard_dataset = CustomDataset(orchard_path, transform)
#forest_dataset = CustomDataset(forest_path, transform)
dataset = CustomDataset(orchard_path, transform) + CustomDataset(forest_path, transform)
split_ratio = 0.8
num_data = len(dataset)
split = int(np.floor(split_ratio * num_data))
indices = list(range(num_data))

indices = [idx for idx in indices if dataset[idx][1] != -1]

np.random.shuffle(indices)


# Create data samplers for training and testing sets
train_indices, test_indices = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create data loaders for training and testing
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import torch.nn.functional as F
num_epochs = 30

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch, (images, labels, img_name) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_accuracy = 100 * correct / total
    return total_loss / (batch + 1), train_accuracy

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    misclassified_paths = []
    with torch.no_grad():
        for batch, (images, labels, img_name) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            incorrect_indices = (predicted != labels).nonzero()
            misclassified_paths.extend([img_name[i] for i in incorrect_indices])


    val_accuracy = 100 * correct / total
    return total_loss / (batch + 1), val_accuracy, misclassified_paths

trnl = []
vall = []
trna=[]
vala=[]
epochs = [x for x in range(num_epochs)]
mistakes = []
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy, misclf = validate(model, test_loader, criterion)
    trnl.append(train_loss)
    vall.append(val_loss)
    trna.append(train_accuracy)
    vala.append(val_accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    if epoch == 20:
        print('epoch 20 erros')
    if epoch == num_epochs-1:
        print('last epoch errors')
        print(misclf)
        mistakes = misclf
        with open("zaf_mistakes_SR.json", 'w') as f:
            json.dump(mistakes, f, indent=2) 
print("Training complete.")
import matplotlib.pyplot as plt
plt.plot(epochs,trnl,label='train loss')
plt.plot(epochs,vall,label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig('zaf-SR-loss.png')

plt.plot(epochs,trna,label='train acc')
plt.plot(epochs,vala,label='val acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.savefig('zaf-SR-acc.png')

orchard_path = 'mountdata/zaforchards2-50m'
forest_path = 'mountdata/zafforests2-50m'

from PIL import Image
import rasterio as rio

#import cv2
#import gdal
from torch.utils.data.sampler import SubsetRandomSampler
#import imagecodecs
from skimage import io

anomalies = []
class CustomDatasetS2(Dataset):
    def __init__(self, data_dir, transform=None, num_channel=12):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)
        self.num_channel = num_channel

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.file_list[idx])
        if img_name.endswith('.tif'):
            if img_name in anomalies:
                pass
            with rio.open(img_name) as img :
                image= img.read()
                image = image.astype('float32')
    
                image = image[:self.num_channel,2:52,2:52]
                image = np.transpose(image,(1,2,0))
                #print
                if image.shape != (50,50,self.num_channel):
                    print('anomaly')
                    anomalies.append(img_name)
                    return torch.zeros(50, 50,self.num_channel), -1
                    #return torch.zeros(10, 10,12), -1
                
                if self.transform:
                    image = self.transform(image)
            label = 0 if "orchard" in self.data_dir else 1  # Assuming "orchard" is class 0 and "forest" is class 1
            return image, label, img_name

transform = transforms.Compose([transforms.ToTensor()])
bicubtransform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((160, 160), interpolation=Image.BICUBIC), transforms.ToTensor()])

datasets2 = CustomDatasetS2(orchard_path, transform, 3) + CustomDatasetS2(forest_path, transform, 3)
#split_ratio = 0.8
#num_data = len(datasets2)
#split = int(np.floor(split_ratio * num_data))
#indices = list(range(num_data))
#indices = [idx for idx in indices if datasets2[idx][1] != -1]
#np.random.shuffle(indices)


# Create data samplers for training and testing sets
train_indices, test_indices = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create data loaders for training and testing
train_loader = DataLoader(datasets2, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(datasets2, batch_size=batch_size, sampler=test_sampler)

import rasterio as rio
model = ResModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training loop with validation
trnl = []
vall = []
trna=[]
vala=[]
epochs = [x for x in range(num_epochs)]
mistakes_s2 = []
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy, misclf = validate(model, test_loader, criterion)
    trnl.append(train_loss)
    vall.append(val_loss)
    trna.append(train_accuracy)
    vala.append(val_accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    if epoch == 20:
        print('epoch 20 erros')
    if epoch == num_epochs-1:
        print('last epoch errors')
        print(misclf)
        mistakes_s2 = misclf
        with open("zaf_mistakes_s2.json", 'w') as f:
            json.dump(mistakes_s2, f, indent=2) 
print("Training complete.")
import matplotlib.pyplot as plt
plt.plot(epochs,trnl,label='train loss')
plt.plot(epochs,vall,label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig('zaf-s2-loss.png')

plt.plot(epochs,trna,label='train acc')
plt.plot(epochs,vala,label='val acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.savefig('zaf-s2-acc.png')

class CustomDatasetS2(Dataset):
    def __init__(self, data_dir, transform=None, num_channel=12):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)
        self.num_channel = num_channel

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.file_list[idx])
        if img_name.endswith('.tif'):
            if img_name in anomalies:
                pass
            with rio.open(img_name) as img :
                image= img.read()
                image = image.astype('float32')
                image = (image * 255).astype('uint8')
                image = image[:self.num_channel,2:52,2:52]
                image = np.transpose(image,(1,2,0))
                #print
                if image.shape != (50,50,self.num_channel):
                    print('anomaly')
                    anomalies.append(img_name)
                    return torch.zeros(50, 50,self.num_channel), -1
                    #return torch.zeros(10, 10,12), -1

                if self.transform:
                    image = self.transform(image)
            label = 0 if "orchard" in self.data_dir else 1  # Assuming "orchard" is class 0 and "forest" is class 1
            return image, label, img_name
datasets2bicub = CustomDatasetS2(orchard_path, bicubtransform, 3) + CustomDatasetS2(forest_path, bicubtransform, 3)

# Create data loaders for training and testing
train_loader = DataLoader(datasets2bicub, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(datasets2bicub, batch_size=batch_size, sampler=test_sampler)

model = ResModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training loop with validation
trnl = []
vall = []
trna=[]
vala=[]
epochs = [x for x in range(num_epochs)]
mistakes_bicub = []
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy, misclf = validate(model, test_loader, criterion)
    trnl.append(train_loss)
    vall.append(val_loss)
    trna.append(train_accuracy)
    vala.append(val_accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    if epoch == 20:
        print('epoch 20 erros')
    if epoch == num_epochs-1:
        print('last epoch errors')
        print(misclf)
        mistakes_bicub = misclf

        with open("zaf_mistakes_bicub.json", 'w') as f:
            json.dump(mistakes_bicub, f, indent=2)


print("Training complete.")
import matplotlib.pyplot as plt
plt.plot(epochs,trnl,label='train loss')
plt.plot(epochs,vall,label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig('zaf-bicub-loss.png')

plt.plot(epochs,trna,label='train acc')
plt.plot(epochs,vala,label='val acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.savefig('zaf-bicub-acc.png')



