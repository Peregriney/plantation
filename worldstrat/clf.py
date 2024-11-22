import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time

import matplotlib.pyplot as plt
import tifffile

bucket_name = 'mountdata'
# Check if CUDA/GPU is available (optional but recommended for faster training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

augment = False

def preprocess_data_gcs_multi(bucket_name, prefix, region):
    # Initialize lists to store images and labels
    images = []
    labels = []
    paths = []
    imagesCHM = []

    countries = os.listdir(os.path.join(bucket_name,prefix))
    # Iterate through the blobs

    if region == 'all':
        print('all countries included')
    elif region == 'east':
        print('manualEast')
        countries = ['manualEast']
    elif region == 'west':
        print('manualWest, CIV, Benin')
        if 'forest' in prefix:
            countries = ['manualWest','benin']
        else:
            countries = ['manualWest','CIV','benin']
    elif region == 'central':
        print('manualCentral')
        countries = ['manualCentral']

    elif region == 'val':
        print('sdptv2')
        if 'forest' in prefix:
            countries = []
        else:
            countries = ['sdptmanual-v2']
        #    countries = ['oil','coco']
    
    for country in countries:
        for blob in os.listdir(os.path.join(bucket_name,prefix,country)):
            #print(blob)
            if 'orchard' in prefix:
                label = 0  # Assign class 0 if 'orchard' is in the filename
                prefixCHM = orchardpathCHM
            else:
                label = 1  # Assign class 1 otherwise
                prefixCHM = forestpathCHM

            fp = os.path.join(bucket_name,prefixCHM,country,blob[:-12]+'.tif')
            if os.path.exists(fp):
                # Read and preprocess the image
                with open(os.path.join(bucket_name,prefix,country,blob), 'rb') as f:
                    # Get the byte data of the image
                    img = tifffile.imread(f)
                    img = img[:40,:40]
                    img = img.astype('float32')
                    #print(img.shape)
                    # Append the preprocessed image and label to the lists
                    if img.shape == (40,40,24):
                        images.append(img)
                        labels.append(label)
                        paths.append(os.path.join(bucket_name, prefix, country, blob[:-12]+'.tif'))
                        with open(fp, 'rb') as f2:
                            # Get the byte data of the image
                            img_bytes2 = f2.read()
                            img2 = cv2.imdecode(np.frombuffer(img_bytes2, np.uint8), cv2.IMREAD_GRAYSCALE)
                            img2 = img2[10:234, 10:234]  # Clip to central 40x40 pixels
                    
                            # Normalize pixel values to range [0, 1]
                            img2 = img2.astype('float32')# / 255.0
                    
                            # Append the preprocessed image and label to the lists
                            imagesCHM.append(img2)
                    else:
                        print('not 40x40',fp)
            else:
                print(fp)
    

    # Convert lists to numpy arrays
    images = np.array(images)
    imagesCHM = np.array(imagesCHM)
    labels = np.array(labels)

    return images, imagesCHM, labels, paths
'''
def preprocess_data_gcs(bucket_name, prefix):
  """Preprocesses data from Google Cloud Storage with augmentation.

  Args:
      bucket_name: Name of the bucket containing the data.
      prefix: Prefix of the folders containing images (e.g., "data/").

  Returns:
      A tuple of (images, labels):
          images: A numpy array of preprocessed images.
          labels: A numpy array of labels.
  """

  # Initialize lists to store images and labels
  images = []
  labels = []
  paths = []

  countries = os.listdir(os.path.join(bucket_name, prefix))
  # Iterate through the blobs
  for country in countries:
    for blob in os.listdir(os.path.join(bucket_name, prefix, country)):
      # Determine label based on prefix
      if 'orchard' in prefix:
        label = 0
      else:
        label = 1

      # Read and preprocess the image
      with open(os.path.join(bucket_name, prefix, country, blob), 'rb') as f:
        img_bytes = f.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        img = img[10:234, 10:234]  # Clip to central 50x50 pixels

        # Normalize pixel values (optional)
        img = img.astype('float32') / 255.0
        images.append(img)
        labels.append(label)
        paths.append(os.path.join(bucket_name, prefix, country, blob))
        # Create augmented versions
        if augment:
            augmented_images = []
            for angle in (90, 180, 270):  # Rotate by 90, 180, and 270 degrees
              rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE * angle)
              augmented_images.append(rotated)
              paths.append(os.path.join(bucket_name, prefix, country, blob+'-rot'))
            flipped = cv2.flip(img, 1)  # Horizontally flip
            augmented_images.append(flipped)
            paths.append(os.path.join(bucket_name, prefix, country, blob+'flip'))
    
            # Add original and augmented images to final list
            images.extend(augmented_images)
            labels.extend([label] * len(augmented_images))  # Duplicate label for each image

  # Convert lists to numpy arrays
  #images = np.array(images)
  #labels = np.array(labels)
  return images, labels, paths
'''



orchardpath = 's2NDVItsorchard-400m'
orchardpathCHM = 'chmorchard-224m'
forestpath = 's2NDVItsforest-400m'
forestpathCHM = 'chmforest-224m'

distshift = True

if not distshift:
    
    # Preprocess the data from GCS
    print('beginning the orchard data preprocessing')
    print(time.perf_counter())
    images, imagesCHM, labels, paths = preprocess_data_gcs_multi(bucket_name, orchardpath)
    
    print('finished the orchard data preprocessing')
    print(time.perf_counter())
    
    images2, imagesCHM2, labels2, paths2 = preprocess_data_gcs_multi(bucket_name, forestpath)
    
    print('finished the forest data preprocessing')
    print(time.perf_counter())


else:
    print('regional shifting')
    #for region in ['val', 'east', 'west', 'central']
    region = 'val'
    print('forest val processing ', region)
    print(time.perf_counter())
    images2, imagesCHM2, labels2, paths2 = preprocess_data_gcs_multi(bucket_name, forestpath, region)
    #print('finished the forest data preprocessing')
    #print(time.perf_counter())
    print('beginning the orchard data preprocessing ', region)
    print(time.perf_counter())
    images, imagesCHM, labels, paths = preprocess_data_gcs_multi(bucket_name, orchardpath, region)
    print('finished the orchard data preprocessing')
    print(time.perf_counter())

    region = 'east'
    imagese, imagesCHMe, labelse, pathse = preprocess_data_gcs_multi(bucket_name, orchardpath, region)
    images2e, imagesCHM2e, labels2e, paths2e = preprocess_data_gcs_multi(bucket_name, forestpath, region)

    region = 'west'
    imagesw, imagesCHMw, labelsw, pathsw = preprocess_data_gcs_multi(bucket_name, orchardpath, region)
    images2w, imagesCHM2w, labels2w, paths2w = preprocess_data_gcs_multi(bucket_name, forestpath, region)

    region = 'central'
    images_c, imagesCHM_c, labels_c, paths_c = preprocess_data_gcs_multi(bucket_name, orchardpath, region)
    images2_c, imagesCHM2_c, labels2_c, paths2_c = preprocess_data_gcs_multi(bucket_name, forestpath, region)

    print('finished all processing')
    print(time.perf_counter())


import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, grayscale_images, band24_images, labels):
        self.grayscale_images = grayscale_images
        self.band24_images = band24_images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        grayscale_image = self.grayscale_images[idx]
        band24_image = self.band24_images[idx]
        label = self.labels[idx]
        return grayscale_image, band24_image, label

    def concatenate(self, datasets):
        """
        Concatenates multiple PairedImageDatasets into a longer dataset.
        Args:
            datasets (list): List of PairedImageDataset instances to concatenate.
        Returns:
            PairedImageDataset: Concatenated dataset.
        """
        # Extract data from the current dataset
        grayscale_images_concat = self.grayscale_images[:]
        band24_images_concat = self.band24_images[:]
        labels_concat = self.labels[:]

        # Iterate over each additional dataset and concatenate their data
        for dataset in datasets:
            #grayscale_images_concat.cat(dataset.grayscale_images)
            #band24_images_concat.cat(dataset.band24_images)
            #labels_concat.cat(dataset.labels)
            #grayscale_images_concat
            self.grayscale_images = torch.cat((grayscale_images_concat,dataset.grayscale_images),dim=0)
            #band24_images_concat
            self.band24_images = torch.cat((band24_images_concat,dataset.band24_images),dim=0)
            #labels_concat #
            self.labels = np.concatenate((labels_concat,dataset.labels),axis=0)

        # Create and return a new concatenated PairedImageDataset
        concatenated_dataset = PairedImageDataset(grayscale_images_concat, band24_images_concat, labels_concat)
        return concatenated_dataset


if distshift:
    imgse = np.concatenate((imagese,images2e))
    imgsCHMe = np.concatenate((imagesCHMe,imagesCHM2e))
    labse = np.concatenate((labelse,labels2e))
    pthse = np.concatenate((pathse,paths2e))
    train_imagese, val_imagese, train_imagesCHMe, val_imagesCHMe, train_labelse, val_labelse = train_test_split(imgse, imgsCHMe, labse, test_size=0.2, random_state=42)

    imgsw = np.concatenate((imagesw,images2w))
    imgsCHMw = np.concatenate((imagesCHMw,imagesCHM2w))
    labsw = np.concatenate((labelsw,labels2w))
    pthsw = np.concatenate((pathsw,paths2w))
    train_imagesw, val_imagesw, train_imagesCHMw, val_imagesCHMw, train_labelsw, val_labelsw = train_test_split(imgsw, imgsCHMw, labsw, test_size=0.2, random_state=42)

    imgsc = np.concatenate((images_c,images2_c))
    imgsCHMc = np.concatenate((imagesCHM_c,imagesCHM2_c))
    labsc = np.concatenate((labels_c,labels2_c))
    pthsc = np.concatenate((paths_c,paths2_c))
    train_imagesc, val_imagesc, train_imagesCHMc, val_imagesCHMc, train_labelsc, val_labelsc = train_test_split(imgsc, imgsCHMc, labsc, test_size=0.2, random_state=42)

    sdptimgs = images
    sdptimgsCHM = imagesCHM
    sdptlabs = labels

    train_images_tensore = torch.tensor(train_imagese.astype('float32')).permute(0, 3, 1, 2)
    train_images_tensorw = torch.tensor(train_imagesw.astype('float32')).permute(0, 3, 1, 2)
    train_images_tensorc = torch.tensor(train_imagesc.astype('float32')).permute(0, 3, 1, 2)
    sdpt_images_tensor = torch.tensor(sdptimgs.astype('float32')).permute(0, 3, 1, 2)

    channel_indices = [5, 11, 17, 23]
    train_images_tensore = train_images_tensore[:, channel_indices, :, :]
    train_images_tensorw = train_images_tensorw[:, channel_indices, :, :]
    train_images_tensorc = train_images_tensorc[:, channel_indices, :, :]
    sdpt_images_tensor = sdpt_images_tensor[:, channel_indices, :, :]

    train_labels_tensore = torch.tensor(train_labelse)
    train_labels_tensorw = torch.tensor(train_labelsw)
    train_labels_tensorc = torch.tensor(train_labelsc)
    sdpt_labels_tensor = torch.tensor(sdptlabs)

    val_images_tensore = torch.tensor(val_imagese).permute(0, 3, 1, 2) #.unsqueeze(1)  # Add a channel dimension
    val_images_tensore = val_images_tensore[:, channel_indices, :, :]
    val_images_tensorw = torch.tensor(val_imagesw).permute(0, 3, 1, 2) #.unsqueeze(1)  # Add a channel dimension
    val_images_tensorw = val_images_tensorw[:, channel_indices, :, :]
    val_images_tensorc = torch.tensor(val_imagesc).permute(0, 3, 1, 2) #.unsqueeze(1)  # Add a channel dimension
    val_images_tensorc = val_images_tensorc[:, channel_indices, :, :]
    val_labels_tensore = torch.tensor(val_labelse)
    val_labels_tensorw = torch.tensor(val_labelsw)
    val_labels_tensorc = torch.tensor(val_labelsc)

    train_imagesCHM_tensore = torch.tensor(train_imagesCHMe,dtype=torch.float32).unsqueeze(1).float()#.permute(2, 0, 1)#.unsqueeze(1)  # Add a channel dimension
    val_imagesCHM_tensore = torch.tensor(val_imagesCHMe,dtype=torch.float32).unsqueeze(1).float()#.permute(2, 0, 1)#.unsqueeze(1)  # Add a channel dimension
    train_imagesCHM_tensorw = torch.tensor(train_imagesCHMw,dtype=torch.float32).unsqueeze(1).float()#.permute(2, 0, 1)#.unsqueeze(1)  # Add a channel dimension
    val_imagesCHM_tensorw = torch.tensor(val_imagesCHMw,dtype=torch.float32).unsqueeze(1).float()#.permute(2, 0, 1)#.unsqueeze(1)  # Add a channel dimension
    train_imagesCHM_tensorc = torch.tensor(train_imagesCHMc,dtype=torch.float32).unsqueeze(1).float()#.permute(2, 0, 1)#.unsqueeze(1)  # Add a channel dimension
    val_imagesCHM_tensorc = torch.tensor(val_imagesCHMc,dtype=torch.float32).unsqueeze(1).float()#.permute(2, 0, 1)#.unsqueeze(1)  # Add a channel dimension
    sdpt_imagesCHM_tensor = torch.tensor(sdptimgsCHM,dtype=torch.float32).unsqueeze(1).float()#.permute(2, 0, 1)#.unsqueeze(1)  # Add a channel dimension

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
            
    # Use the custom dataset class
    train_datasete = PairedImageDataset(train_images_tensore, train_imagesCHM_tensore, train_labelse)
    val_datasete = PairedImageDataset(val_images_tensore, val_imagesCHM_tensore, val_labelse)
    train_loadere = DataLoader(train_datasete, batch_size=32, shuffle=True)
    val_loadere = DataLoader(val_datasete, batch_size=32)

#    etestw_loader = DataLoader(PairedImageDataset(torch.cat((train_images_tensorw,val_images_tensorw)), torch.cat((train_labelsw,val_labelsw)) ), batch_size=32)    
#    etestc_loader = DataLoader(PairedImageDataset(torch.cat((train_images_tensorc,val_images_tensorc)), torch.cat((train_labelsc,val_labelsc)) ), batch_size=32)    

    train_datasetw = PairedImageDataset(train_images_tensorw, train_imagesCHM_tensorw, train_labelsw)
    val_datasetw = PairedImageDataset(val_images_tensorw, val_imagesCHM_tensorw, val_labelsw)
    train_loaderw = DataLoader(train_datasetw, batch_size=32, shuffle=True)
    val_loaderw = DataLoader(val_datasetw, batch_size=32)
    
    train_datasetc = PairedImageDataset(train_images_tensorc, train_imagesCHM_tensorc, train_labelsc)
    val_datasetc = PairedImageDataset(val_images_tensorc, val_imagesCHM_tensorc, val_labelsc)
    train_loaderc = DataLoader(train_datasetc, batch_size=32, shuffle=True)
    val_loaderc = DataLoader(val_datasetc, batch_size=32)

    sdpt_dataset = PairedImageDataset(sdpt_images_tensor, sdpt_imagesCHM_tensor, sdptlabs)
    sdpt_loader = DataLoader(sdpt_dataset, batch_size=32, shuffle=True)

    full_val_ds = sdpt_dataset.concatenate([val_datasete,val_datasetw,val_datasetc])
    #(torch.cat(sdpt_images_tensor, val_images_tensore, val_images_tensorw, val_images_tensorc), 
    #                                 torch.cat(sdpt_imagesCHM_tensor, val_imagesCHM_tensore, val_imagesCHM_tensorw, val_imagesCHM_tensorc), 
    #                                 torch.cat(sdpt_labels, val_labelse, val_labelsw, val_labelsc))
    full_val_loader = DataLoader(full_val_ds,batch_size=32,shuffle=True)


import torch.nn.init as init
class MultiResNet(nn.Module):
    def __init__(self):
        super(MultiResNet, self).__init__()

        # First ResNet for grayscale images
        self.resnet_gray = models.resnet18(pretrained=True)
        self.resnet_gray.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Second ResNet for 24-band images
        self.resnet_24band = models.resnet18(pretrained=True)
        self.resnet_24band.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Get the number of features in the final layer
        num_features = self.resnet_gray.fc.in_features
        num_features24 = self.resnet_24band.fc.in_features

        # Concatenate the features from both streams
        self.fc = nn.Linear(num_features + num_features24, 1)  # Adjust the output size as needed
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function for binary classification

        #self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    def forward(self, x_gray, x_24band):

        # Forward pass through the grayscale ResNet
        x_gray = self.resnet_gray.conv1(x_gray)
        x_gray = self.resnet_gray.bn1(x_gray)
        x_gray = self.resnet_gray.relu(x_gray)
        x_gray = self.resnet_gray.maxpool(x_gray)
        #print('x_gray after maxpool')
        #print(x_gray)
        x_gray = self.resnet_gray.layer1(x_gray)
        x_gray = self.resnet_gray.layer2(x_gray)
        x_gray = self.resnet_gray.layer3(x_gray)
        x_gray = self.resnet_gray.layer4(x_gray)
        #print('x_gray after layers')
        #print(x_gray)
        x_gray = self.resnet_gray.avgpool(x_gray)
        x_gray = torch.flatten(x_gray, 1)
        #print('grayscale CHM at flatten')
        #print(x_gray)

        #print('before forward pass: ')
        #print(x_24band)
        # Forward pass through the 24-band ResNet
        x_24band = self.resnet_24band.conv1(x_24band)
        #print('after conv1 forward pass: ')
        #print(x_24band[0])
        x_24band = self.resnet_24band.bn1(x_24band)
        #print('after bn')
        #print(x_24band[0])
        x_24band = self.resnet_24band.relu(x_24band)
        #print('after relu')
        #print(x_24band[0])
        x_24band = self.resnet_24band.maxpool(x_24band)
        #print('x_24 after maxpool')
        #print(x_24band[0])
        x_24band = self.resnet_24band.layer1(x_24band)
        x_24band = self.resnet_24band.layer2(x_24band)
        x_24band = self.resnet_24band.layer3(x_24band)
        x_24band = self.resnet_24band.layer4(x_24band)
        #print('x_24 after layers')
        #print(x_24band[0])
        x_24band = self.resnet_24band.avgpool(x_24band)
        x_24band = torch.flatten(x_24band, 1)
        #print('sentinel2 at flatten')
        #print(x_24band[0])

        # Concatenate the features from both streams
        x = torch.cat((x_gray, x_24band), dim=1)
        #print('x before clf')
        #print(x)
        # Classification head
        x = self.fc(x)
        #print('x after fc')
        #print(x)

        x = self.sigmoid(x)
        #print('x after clf head: ')
        #print(x)
        #print('after forward pass: ')
        #print(model.parameters())
        return x

# Example usage
model = MultiResNet()
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

from sklearn.metrics import confusion_matrix
def plot_res(trn_acc_list,val_acc_list,trn_loss_list,val_loss_list,all_labels,all_predictions):

    # Extract the training and validation accuracy from the model history
    train_accuracy = trn_acc_list
    val_accuracy = val_acc_list

    # Get the number of epochs
    epochs = range(len(train_accuracy))

    # Create the plot
    plt.plot(epochs, train_accuracy, label='Training accuracy')
    plt.plot(epochs, val_accuracy, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # Extract the training and validation accuracy from the model history
    train_losses = trn_loss_list
    val_losses = val_loss_list

    # Get the number of epochs
    epochs = range(len(train_accuracy))

    # Create the plot
    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, val_losses, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    print('acc %')
    print(correct/total * 100)
    print('total INCORRECT', 'total examples in val', 'number of correct examples in val')
    print(total-correct, total, correct)
#    print('val loss')
#    print(val_loss / len(val_loader.dataset))

    print_cm(all_labels,all_predictions)

def print_cm(all_labels,all_predictions):

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Extract true positives, negatives, false positives, and negatives
    tn, fp, fn, tp = cm.ravel()

    # Calculate Precision, Recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # Print results (you can modify this for your needs)
    print(f"Confusion Matrix:\n {cm}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    #previously: leak-free resulted in overfitting by epoch 3 with a peak val acc ~80%.

###############################
###############################
#########DISTRIBUTION SHIFT TRN EAST
###############################
###############################
#MEAN INTERPOLATION MODEL
print('training start')
trn_acc_list = []
val_acc_list = []
trn_loss_list = []
val_loss_list = []
all_predictions = []
all_labels = []

criterion = nn.BCEWithLogitsLoss()
# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0
    train_loss = 0.0

    for inputs, inputsCHM, labs in train_loadere:
        #print(inputs.shape) #32,4,40,40
        #print(inputs)

        inputs = inputs.to("cuda")
        inputsCHM = inputsCHM.to("cuda")
        labs = torch.Tensor(labs).unsqueeze(1).to("cuda")


        # Calculate mean of non-NaN values in each channel

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data
            #if mask.sum() > 0:
            #    print("Inputs after interpolation:")
            #    print(channel_data)
            # Check for NaNs in inputs
        if torch.isnan(inputs).any():
            raise ValueError("NaNs found in inputs after interpolation")

        # Check for NaNs in inputsCHM and labs
        if torch.isnan(inputsCHM).any():
            raise ValueError("NaNs found in inputsCHM")
        if torch.isnan(labs).any():
            raise ValueError("NaNs found in labs")

        #inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)

        optimizer.zero_grad()
        outputs = model(inputsCHM,inputs)
        if torch.isnan(outputs).any():
            print(f"Inputs: {inputs[0][0]}")
            #print(f"InputsCHM: {inputsCHM}")
            print(f"Model outputs: {outputs}")
            raise ValueError("NaNs found in model outputs")

        loss = criterion(outputs, labs.float())
        if torch.isnan(loss).any():
            raise ValueError("NaNs found in loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
        correct += (predicted == labs).sum().item()
    trn_acc_list.append(correct/total * 100)
    trn_loss_list.append(train_loss/len(train_loadere.dataset))


    if epoch == 19:
        torch.save(model.state_dict(), 'resnet-multimodal-EASTtrain.pt')


    # Validate the model
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():

        for inputs, inputsCHM, labs in val_loadere:

            inputs= inputs.to("cuda")
            inputsCHM = inputsCHM.to("cuda")
            labs = torch.Tensor(labs).unsqueeze(1).to("cuda")
            #normalized_inputs = normalizer(inputs)

            has_nan = torch.isnan(inputs).any(dim=1)  # Check along batch dimension (assuming dim=0 is batch)
            # Filter inputs to exclude those with NaNs
            has_nan_anywhere = torch.any(torch.any(has_nan, dim=1), dim=1)

            inputs = inputs[~has_nan_anywhere]
            inputsCHM = inputsCHM[~has_nan_anywhere]
            labs = labs[~has_nan_anywhere]

            outputs = model(inputsCHM, inputs)
            loss = criterion(outputs, labs.float())
            val_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labs.size(0)
            correct += (predicted == labs).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
        val_acc_list.append(correct/total * 100)
        val_loss_list.append(val_loss / len(val_loadere.dataset))

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss/len(train_loadere.dataset):.4f}, '
          f'Train Acc: {trn_acc_list[-1]:.4f}, '
          #f'Train Acc: {trn_acc_list[-1])*100:.4f}, '
          f'Val Loss: {val_loss/len(val_loadere.dataset):.4f}, '
          f'Val Acc: {(correct/total)*100:.2f}' )

plot_res(trn_acc_list,val_acc_list,trn_loss_list,val_loss_list,all_labels,all_predictions)



val_loss = 0.0
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, inputsCHM, labs in val_loaderc:

        inputs= inputs.to(device)
        inputsCHM = inputsCHM.to(device)
        labs = torch.Tensor(labs).unsqueeze(1).to(device)
        '''
        has_nan = torch.isnan(inputs).any(dim=1)
        has_nan_anywhere = torch.any(torch.any(has_nan, dim=1), dim=1)
        inputs = inputs[~has_nan_anywhere]
        inputsCHM = inputsCHM[~has_nan_anywhere]
        labs = labs[~has_nan_anywhere]
        '''

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data

        outputs = model(inputsCHM, inputs)
        loss = criterion(outputs, labs.float())
        val_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
#        print(predicted)
        correct += (predicted == labs).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    print('acc %')
    print(correct/total * 100)
    print('total INCORRECT', 'total examples in val', 'number of correct examples in val')
    print(total-correct, total, correct)
    print('val loss')
    print(val_loss / len(val_loaderc.dataset))
    print_cm(all_predictions,all_labels)



val_loss = 0.0
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, inputsCHM, labs in val_loaderw:

        inputs= inputs.to(device)
        inputsCHM = inputsCHM.to(device)
        labs = torch.Tensor(labs).unsqueeze(1).to(device)
        '''
        has_nan = torch.isnan(inputs).any(dim=1)
        has_nan_anywhere = torch.any(torch.any(has_nan, dim=1), dim=1)
        inputs = inputs[~has_nan_anywhere]
        inputsCHM = inputsCHM[~has_nan_anywhere]
        labs = labs[~has_nan_anywhere]
        '''

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data

        outputs = model(inputsCHM, inputs)
        loss = criterion(outputs, labs.float())
        val_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
#        print(predicted)
        correct += (predicted == labs).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    print('acc %')
    print(correct/total * 100)
    print('total INCORRECT', 'total examples in val', 'number of correct examples in val')
    print(total-correct, total, correct)
    print('val loss')
    print(val_loss / len(val_loaderw.dataset))
    print_cm(all_predictions,all_labels)



val_loss = 0.0
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, inputsCHM, labs in full_val_loader:

        inputs= inputs.to(device)
        inputsCHM = inputsCHM.to(device)
        labs = torch.Tensor(labs).unsqueeze(1).to(device)
        '''
        has_nan = torch.isnan(inputs).any(dim=1)
        has_nan_anywhere = torch.any(torch.any(has_nan, dim=1), dim=1)
        inputs = inputs[~has_nan_anywhere]
        inputsCHM = inputsCHM[~has_nan_anywhere]
        labs = labs[~has_nan_anywhere]
        '''

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data

        outputs = model(inputsCHM, inputs)
        loss = criterion(outputs, labs.float())
        val_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
#        print(predicted)
        correct += (predicted == labs).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    print('acc %')
    print(correct/total * 100)
    print('total INCORRECT', 'total examples in val', 'number of correct examples in val')
    print(total-correct, total, correct)
    print('val loss')
    print(val_loss / len(full_val_loader.dataset))
    print_cm(all_predictions,all_labels)



model = MultiResNet()
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

###############################
###############################
#########DISTRIBUTION SHIFT TRN CENTRAL
###############################
###############################
#MEAN INTERPOLATION MODEL
print('training start')
trn_acc_list = []
val_acc_list = []
trn_loss_list = []
val_loss_list = []
all_predictions = []
all_labels = []

criterion = nn.BCEWithLogitsLoss()
# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0
    train_loss = 0.0

    for inputs, inputsCHM, labs in train_loaderc:
        #print(inputs.shape) #32,4,40,40
        #print(inputs)

        inputs = inputs.to("cuda")
        inputsCHM = inputsCHM.to("cuda")
        labs = torch.Tensor(labs).unsqueeze(1).to("cuda")


        # Calculate mean of non-NaN values in each channel

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data
            #if mask.sum() > 0:
            #    print("Inputs after interpolation:")
            #    print(channel_data)
            # Check for NaNs in inputs
        if torch.isnan(inputs).any():
            raise ValueError("NaNs found in inputs after interpolation")

        # Check for NaNs in inputsCHM and labs
        if torch.isnan(inputsCHM).any():
            raise ValueError("NaNs found in inputsCHM")
        if torch.isnan(labs).any():
            raise ValueError("NaNs found in labs")

        #inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)

        optimizer.zero_grad()
        outputs = model(inputsCHM,inputs)
        if torch.isnan(outputs).any():
            print(f"Inputs: {inputs[0][0]}")
            #print(f"InputsCHM: {inputsCHM}")
            print(f"Model outputs: {outputs}")
            raise ValueError("NaNs found in model outputs")

        loss = criterion(outputs, labs.float())
        if torch.isnan(loss).any():
            raise ValueError("NaNs found in loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
        correct += (predicted == labs).sum().item()
    trn_acc_list.append(correct/total * 100)
    trn_loss_list.append(train_loss/len(train_loaderc.dataset))


    if epoch == 19:
        torch.save(model.state_dict(), 'resnet-multimodal-centraltrain.pt')


    # Validate the model
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():

        for inputs, inputsCHM, labs in val_loaderc:

            inputs= inputs.to("cuda")
            inputsCHM = inputsCHM.to("cuda")
            labs = torch.Tensor(labs).unsqueeze(1).to("cuda")
            #normalized_inputs = normalizer(inputs)

            has_nan = torch.isnan(inputs).any(dim=1)  # Check along batch dimension (assuming dim=0 is batch)
            # Filter inputs to exclude those with NaNs
            has_nan_anywhere = torch.any(torch.any(has_nan, dim=1), dim=1)

            inputs = inputs[~has_nan_anywhere]
            inputsCHM = inputsCHM[~has_nan_anywhere]
            labs = labs[~has_nan_anywhere]

            outputs = model(inputsCHM, inputs)
            loss = criterion(outputs, labs.float())
            val_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labs.size(0)
            correct += (predicted == labs).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
        val_acc_list.append(correct/total * 100)
        val_loss_list.append(val_loss / len(val_loaderc.dataset))

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss/len(train_loaderc.dataset):.4f}, '
          f'Train Acc: {trn_acc_list[-1]:.4f}, '
          #f'Train Acc: {trn_acc_list[-1])*100:.4f}, '
          f'Val Loss: {val_loss/len(val_loaderc.dataset):.4f}, '
          f'Val Acc: {(correct/total)*100:.2f}' )

plot_res(trn_acc_list,val_acc_list,trn_loss_list,val_loss_list,all_labels,all_predictions)



val_loss = 0.0
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, inputsCHM, labs in val_loadere:

        inputs= inputs.to(device)
        inputsCHM = inputsCHM.to(device)
        labs = torch.Tensor(labs).unsqueeze(1).to(device)
        '''
        has_nan = torch.isnan(inputs).any(dim=1)
        has_nan_anywhere = torch.any(torch.any(has_nan, dim=1), dim=1)
        inputs = inputs[~has_nan_anywhere]
        inputsCHM = inputsCHM[~has_nan_anywhere]
        labs = labs[~has_nan_anywhere]
        '''

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data

        outputs = model(inputsCHM, inputs)
        loss = criterion(outputs, labs.float())
        val_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
#        print(predicted)
        correct += (predicted == labs).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    print('acc %')
    print(correct/total * 100)
    print('total INCORRECT', 'total examples in val', 'number of correct examples in val')
    print(total-correct, total, correct)
    print('val loss')
    print(val_loss / len(val_loadere.dataset))
    print_cm(all_predictions,all_labels)



val_loss = 0.0
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, inputsCHM, labs in val_loaderw:

        inputs= inputs.to(device)
        inputsCHM = inputsCHM.to(device)
        labs = torch.Tensor(labs).unsqueeze(1).to(device)
        '''
        has_nan = torch.isnan(inputs).any(dim=1)
        has_nan_anywhere = torch.any(torch.any(has_nan, dim=1), dim=1)
        inputs = inputs[~has_nan_anywhere]
        inputsCHM = inputsCHM[~has_nan_anywhere]
        labs = labs[~has_nan_anywhere]
        '''

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data

        outputs = model(inputsCHM, inputs)
        loss = criterion(outputs, labs.float())
        val_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
#        print(predicted)
        correct += (predicted == labs).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    print('acc %')
    print(correct/total * 100)
    print('total INCORRECT', 'total examples in val', 'number of correct examples in val')
    print(total-correct, total, correct)
    print('val loss')
    print(val_loss / len(val_loaderw.dataset))
    print_cm(all_predictions,all_labels)



val_loss = 0.0
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, inputsCHM, labs in full_val_loader:

        inputs= inputs.to(device)
        inputsCHM = inputsCHM.to(device)
        labs = torch.Tensor(labs).unsqueeze(1).to(device)

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data

        outputs = model(inputsCHM, inputs)
        loss = criterion(outputs, labs.float())
        val_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
#        print(predicted)
        correct += (predicted == labs).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    print('acc %')
    print(correct/total * 100)
    print('total INCORRECT', 'total examples in val', 'number of correct examples in val')
    print(total-correct, total, correct)
    print('val loss')
    print(val_loss / len(full_val_loader.dataset))
    print_cm(all_predictions,all_labels)


model = MultiResNet()
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

###############################
###############################
#########DISTRIBUTION SHIFT TRN WEST
###############################
###############################
#MEAN INTERPOLATION MODEL
print('training start')
trn_acc_list = []
val_acc_list = []
trn_loss_list = []
val_loss_list = []
all_predictions = []
all_labels = []

criterion = nn.BCEWithLogitsLoss()
# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0
    train_loss = 0.0

    for inputs, inputsCHM, labs in train_loaderw:
        #print(inputs.shape) #32,4,40,40
        #print(inputs)

        inputs = inputs.to("cuda")
        inputsCHM = inputsCHM.to("cuda")
        labs = torch.Tensor(labs).unsqueeze(1).to("cuda")


        # Calculate mean of non-NaN values in each channel

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data
            #if mask.sum() > 0:
            #    print("Inputs after interpolation:")
            #    print(channel_data)
            # Check for NaNs in inputs
        if torch.isnan(inputs).any():
            raise ValueError("NaNs found in inputs after interpolation")

        # Check for NaNs in inputsCHM and labs
        if torch.isnan(inputsCHM).any():
            raise ValueError("NaNs found in inputsCHM")
        if torch.isnan(labs).any():
            raise ValueError("NaNs found in labs")

        #inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)

        optimizer.zero_grad()
        outputs = model(inputsCHM,inputs)
        if torch.isnan(outputs).any():
            print(f"Inputs: {inputs[0][0]}")
            #print(f"InputsCHM: {inputsCHM}")
            print(f"Model outputs: {outputs}")
            raise ValueError("NaNs found in model outputs")

        loss = criterion(outputs, labs.float())
        if torch.isnan(loss).any():
            raise ValueError("NaNs found in loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
        correct += (predicted == labs).sum().item()
    trn_acc_list.append(correct/total * 100)
    trn_loss_list.append(train_loss/len(train_loaderw.dataset))


    if epoch == 19:
        torch.save(model.state_dict(), 'resnet-multimodal-westtrain.pt')


    # Validate the model
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():

        for inputs, inputsCHM, labs in val_loaderw:

            inputs= inputs.to("cuda")
            inputsCHM = inputsCHM.to("cuda")
            labs = torch.Tensor(labs).unsqueeze(1).to("cuda")
            #normalized_inputs = normalizer(inputs)

            has_nan = torch.isnan(inputs).any(dim=1)  # Check along batch dimension (assuming dim=0 is batch)
            # Filter inputs to exclude those with NaNs
            has_nan_anywhere = torch.any(torch.any(has_nan, dim=1), dim=1)

            inputs = inputs[~has_nan_anywhere]
            inputsCHM = inputsCHM[~has_nan_anywhere]
            labs = labs[~has_nan_anywhere]

            outputs = model(inputsCHM, inputs)
            loss = criterion(outputs, labs.float())
            val_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labs.size(0)
            correct += (predicted == labs).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
        val_acc_list.append(correct/total * 100)
        val_loss_list.append(val_loss / len(val_loaderw.dataset))

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss/len(train_loaderw.dataset):.4f}, '
          f'Train Acc: {trn_acc_list[-1]:.4f}, '
          #f'Train Acc: {trn_acc_list[-1])*100:.4f}, '
          f'Val Loss: {val_loss/len(val_loaderw.dataset):.4f}, '
          f'Val Acc: {(correct/total)*100:.2f}' )

plot_res(trn_acc_list,val_acc_list,trn_loss_list,val_loss_list,all_labels,all_predictions)



val_loss = 0.0
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, inputsCHM, labs in val_loadere:

        inputs= inputs.to(device)
        inputsCHM = inputsCHM.to(device)
        labs = torch.Tensor(labs).unsqueeze(1).to(device)
        '''
        has_nan = torch.isnan(inputs).any(dim=1)
        has_nan_anywhere = torch.any(torch.any(has_nan, dim=1), dim=1)
        inputs = inputs[~has_nan_anywhere]
        inputsCHM = inputsCHM[~has_nan_anywhere]
        labs = labs[~has_nan_anywhere]
        '''

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data

        outputs = model(inputsCHM, inputs)
        loss = criterion(outputs, labs.float())
        val_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
#        print(predicted)
        correct += (predicted == labs).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    print('acc %')
    print(correct/total * 100)
    print('total INCORRECT', 'total examples in val', 'number of correct examples in val')
    print(total-correct, total, correct)
    print('val loss')
    print(val_loss / len(val_loadere.dataset))
    print_cm(all_predictions,all_labels)



val_loss = 0.0
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, inputsCHM, labs in val_loaderc:

        inputs= inputs.to(device)
        inputsCHM = inputsCHM.to(device)
        labs = torch.Tensor(labs).unsqueeze(1).to(device)
        '''
        has_nan = torch.isnan(inputs).any(dim=1)
        has_nan_anywhere = torch.any(torch.any(has_nan, dim=1), dim=1)
        inputs = inputs[~has_nan_anywhere]
        inputsCHM = inputsCHM[~has_nan_anywhere]
        labs = labs[~has_nan_anywhere]
        '''

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data

        outputs = model(inputsCHM, inputs)
        loss = criterion(outputs, labs.float())
        val_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
#        print(predicted)
        correct += (predicted == labs).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    print('acc %')
    print(correct/total * 100)
    print('total INCORRECT', 'total examples in val', 'number of correct examples in val')
    print(total-correct, total, correct)
    print('val loss')
    print(val_loss / len(val_loaderc.dataset))
    print_cm(all_predictions,all_labels)



val_loss = 0.0
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, inputsCHM, labs in full_val_loader:

        inputs= inputs.to(device)
        inputsCHM = inputsCHM.to(device)
        labs = torch.Tensor(labs).unsqueeze(1).to(device)

        for c in range(inputs.size(1)):  # Assuming the channel dimension is 1
            channel_data = inputs[:, c, :, :]
            mask = torch.isnan(channel_data)
            #if mask.sum() > 0:
            #    print("Inputs before interpolation:")
            #    print(channel_data)
            mean_val = torch.nanmean(channel_data, dim=0, keepdim=True)

            channel_data[mask] = mean_val.expand_as(channel_data)[mask]
            inputs[:, c, :, :] = channel_data

        outputs = model(inputsCHM, inputs)
        loss = criterion(outputs, labs.float())
        val_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
#        print(predicted)
        correct += (predicted == labs).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    print('acc %')
    print(correct/total * 100)
    print('total INCORRECT', 'total examples in val', 'number of correct examples in val')
    print(total-correct, total, correct)
    print('val loss')
    print(val_loss / len(full_val_loader.dataset))
    print_cm(all_predictions,all_labels)

