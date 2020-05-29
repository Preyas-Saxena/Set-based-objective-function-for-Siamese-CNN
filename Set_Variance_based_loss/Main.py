
import torch
import Data_processing
from torch.utils.data import DataLoader
from torchvision import transforms
import Model_pt2
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(2)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'

# rootdir='/home/preyas/Desktop/Thesis/polyvore/'
rootdir='/data/leuven/329/vsc32927/polyvore/'

train_file='train_no_dup.json'
valid_file='valid_no_dup.json'

#Actual Values from the training dataset:
mean_data=[0.7170, 0.6794, 0.6613]
std_data=[0.2358, 0.2511, 0.2574]

normalize = transforms.Normalize(mean=mean_data, std=std_data)
transform_list=transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])

learning_rate=5e-5

#Output is a variable containing tuples of (image,pos_image) pairs:
data=Data_processing.style_labels(datadir=rootdir,filename=train_file)
train_loader=DataLoader(data,batch_size=12,shuffle=True)

val_data=Data_processing.style_labels(datadir=rootdir,filename=valid_file)
val_loader=DataLoader(val_data,batch_size=5,shuffle=False)

#Import the model:
vgg = Model_pt2.initialize_model("vgg", 128, feature_extract=False, use_pretrained=True)
vgg.to(device)

#Defining the optimizer:
optimizer=optim.Adam(vgg.parameters(),lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=0.75)

#Training the model parameters:
print('Learning rate is:', learning_rate)
Model_pt2.varnet(model=vgg, dataloader=train_loader, valid_loader= val_loader, optimizer=optimizer, scheduler=scheduler, epochs=5, transforms=transform_list, device=device, validating=True)
print('Done')

