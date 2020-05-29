
from torch.utils.data import DataLoader
import Data_process
from torchvision import transforms
import Model
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(2)
title_location='/home/preyas/Desktop/Thesis/Related_Docs/type_aware_data/type_aware_polyvore_outfits/'
outfit_location='/home/preyas/Desktop/Thesis/Related_Docs/type_aware_data/type_aware_polyvore_outfits/nondisjoint/'

titles_file='polyvore_outfit_titles.json'
outfit_train_files=['train.json']
outfit_valid_files=['valid.json']

# device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
device='cpu'

mean_data=[0.7170, 0.6794, 0.6613]
std_data=[0.2358, 0.2511, 0.2574]

normalize = transforms.Normalize(mean=mean_data, std=std_data)
transform_list=transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])

#Loading the training data:
data=Data_process.style_classes(outfit_direc= outfit_location, outfit_files= outfit_train_files, titles_direc=title_location, titles_file=titles_file)
train_loader=DataLoader(data,batch_size=10,shuffle=True)

#Loading the validation data:
valid_data=Data_process.style_classes(outfit_direc= outfit_location, outfit_files= outfit_valid_files, titles_direc=title_location, titles_file=titles_file)
valid_loader=DataLoader(valid_data,batch_size=5,shuffle=False)

#Load a trained model from a checkpoint (optional):
PATH='/home/preyas/Desktop/Thesis/Thesis_models/Thesis_varnet_redo/Strict/margin2/05-03-2020_17:29:57_base.ckpt'
checkpoint = torch.load(PATH, map_location= torch.device('cpu'))

#Initializing a deep CNN model:
vgg = Model.initialize_model("vgg", 128, feature_extract=True, use_pretrained=True)
vgg.load_state_dict(checkpoint['model_state_dict'])
vgg.to(device)

#Initializing a multi-layer perceptron for style-classification:
D_in=128
H1=64
D_out=4

fc_module=torch.nn.Sequential(
    torch.nn.Linear(D_in,H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1,D_out),
    torch.nn.Softmax(dim=1),)

fc_module.to(device)

learning_rate=5e-4
print('Learning rate is: ', learning_rate)

#Defining the optimizer and lr scheduler:
optimizer=optim.Adam(fc_module.parameters(),lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=0.75)

#Training the style classification model:
Model.classify(cnn_model=vgg, mlp_model=fc_module, dataloader=train_loader, validloader=valid_loader,optimizer=optimizer, epochs=5, scheduler=scheduler,device=device, outfit_direc=outfit_location, outfit_train_files=outfit_train_files, outfit_valid_files=outfit_valid_files, transform=transform_list, validating=True)
