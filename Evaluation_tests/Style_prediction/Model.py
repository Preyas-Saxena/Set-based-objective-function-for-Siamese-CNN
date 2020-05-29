
import torchvision.models as models
import torch.nn as nn
import torch
import json
from PIL import Image
import math
from datetime import datetime
import statistics

imgdir='/home/preyas/Desktop/Thesis/Related_Docs/type_aware_data/type_aware_polyvore_outfits/images/'
# imgdir='/data/leuven/329/vsc32927/type_aware_polyvore/images/'

# Function for setting the model parameters' requires_grad flag:
def set_params_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Function for initializing the model as per the embedding size required and if extracting features versus training using the initial weights:
def initialize_model(model_name, embedding_size, feature_extract, use_pretrained):
    if model_name == "vgg":
        model = models.vgg16_bn(use_pretrained)
        set_params_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, embedding_size)
    else:
        pass

    return model

def save_model(epoch, model_state_params, optimizer_state_param, loss):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M")
    path_remote = '/home/preyas/Desktop/Thesis/Thesis_models/Thesis_varnet_redo/Strict/margin2/Classify/' + dt_string + '_' + str(epoch) + 'epoch.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_params,
        'optimizer_state_dict': optimizer_state_param,
        'loss': loss}, path_remote)
    return None

def load_train_items(path, transform):
    img= Image.open(path).convert('RGB')
    if transform is not None:
        return transform(img)
    else:
        return img

def outfit_to_tensor(outfit_ids, outfit_direc, outfit_files, transforms):

    outfit_data=[]

    for file in outfit_files:
        # Reading outfit to item mapping file (Need to decide on test/train data and concatenate files accordingly)!:
        outfit_data = outfit_data + json.load(open(str(outfit_direc + file), 'r'))

    # Getting batch size:
    batch_size = outfit_ids.shape[0]

    item_list = []
    outfit_lengths_list = []

    for idx in range(batch_size):
        outfit_id = outfit_ids[idx].item()

        for outfit in outfit_data:
            if str(outfit['set_id']) == str(outfit_id):
                outfit_items = outfit['items']
                outfit_lengths_list.append(len(outfit_items))

                for item in outfit_items:
                    item_id = item['item_id']
                    item_path = imgdir + str(item_id) + '.jpg'
                    item_tensor = load_train_items(item_path, transform=transforms)
                    item_list.append((item_tensor))

    # Stacking items from item_list in the form of a torch tensor- which can be used as an input to DL models:
    stacked_items = torch.stack(item_list)
    return stacked_items, outfit_lengths_list

def set_encoder(embeds, split_sizes):
    outfit_splits = torch.split(embeds, split_size_or_sections=split_sizes, dim=0)  # tuple containing splits
    set_rep = []

    for i in outfit_splits:
        set_rep.append(torch.mean(i, dim=0))

    set_rep = torch.stack(set_rep)
    return set_rep

def target_labels(labels):

    label_list=[]
    for label in labels:
        if label==1:
            label_list.append(torch.tensor([1.,0.,0.,0.]))
        if label==2:
            label_list.append(torch.tensor([0.,1.,0.,0.]))
        if label==3:
            label_list.append(torch.tensor([0.,0.,1.,0.]))
        if label==4:
            label_list.append(torch.tensor([0.,0.,0.,1.]))
    label_list=torch.stack(label_list)

    return label_list

def cross_entropy(pred, ground_truth):

    epsilon=1e-16
    pred=pred+epsilon #to avoid taking log 0

    loss = torch.sum(torch.mul(torch.log10(pred), ground_truth), dim=1)
    loss= loss*-1
    batch_loss = torch.mean(loss)

    return batch_loss

def weighted_cross_entropy(weights, pred, ground_truth):

    epsilon=1e-16
    pred=pred+epsilon #to avoid taking log 0

    prod = torch.mul(torch.log10(pred), ground_truth)
    loss= torch.sum(torch.mul(weights, prod), dim=1)
    loss= loss*-1
    batch_loss = torch.mean(loss)

    return batch_loss

def classify(cnn_model, mlp_model ,dataloader, validloader, optimizer, epochs , scheduler, device, outfit_direc, outfit_train_files, outfit_valid_files, transform, validating):

    for epoch in range(1,epochs+1):
        batch_losses=[]
        scheduler.step()
        print('Current Learning rate is: ', optimizer.param_groups[0]['lr'], 'and', scheduler.get_lr())

        for batch, (outfit_ids, outfit_types) in enumerate(dataloader):

            cnn_model.eval()
            mlp_model.train()

            item_tensors, outfit_lengths = outfit_to_tensor(outfit_ids=outfit_ids, outfit_direc=outfit_direc,outfit_files=outfit_train_files,transforms=transform)

            item_embeds=cnn_model(item_tensors.to(device))
            set_rep=set_encoder(embeds=item_embeds, split_sizes=outfit_lengths)

            output = mlp_model(set_rep.to(device))
            target=target_labels(outfit_types)

            # Setting weights tensor for weighted cross entropy:
            # weights = torch.tensor([[1., 1., 1., 1.]], requires_grad=False)

            cost=cross_entropy(pred=output,ground_truth=target.to(device)) #Calculating loss over this batch
            # print(cost.item())

            if math.isnan(cost):
                break

            with torch.no_grad():
                batch_losses.append(cost.item())

            mlp_model.zero_grad()
            cost.backward()
            optimizer.step()

            # Printing the batch loss after every ith iteration:
            if (batch+1)%20 ==0:
                print('Epoch {}, Step {}/{}, Loss: {:.4f}'.format(epoch, (batch + 1), len(dataloader), cost.item()))

            #Printing cumulative loss after every 50 iterations:
            if (((batch+1)%50 == 0) or ((batch+1)==len(dataloader))):
                cum_loss= sum(batch_losses)/len(batch_losses)
                print('Epoch {}, After {} batches, Loss: {:.4f}'.format(epoch,(batch+1), cum_loss))

            # Validating:
            if (((batch+1)) == len(dataloader)) and validating:
                print('Validation mode')

                cnn_model.eval()
                mlp_model.eval()

                with torch.no_grad():
                    val_batch_losses=[]

                    for val_batch, (val_outfit_ids, val_outfit_types) in enumerate(validloader):

                        val_item_tensors, val_outfit_lengths = outfit_to_tensor(outfit_ids=val_outfit_ids, outfit_direc=outfit_direc, outfit_files=outfit_valid_files, transforms=transform)

                        val_item_embeds = cnn_model(val_item_tensors.to(device))
                        val_set_rep = set_encoder(embeds=val_item_embeds, split_sizes=val_outfit_lengths)

                        val_output = mlp_model(val_set_rep.to(device))
                        val_target = target_labels(val_outfit_types)

                        val_batch_cost = cross_entropy(pred=val_output, ground_truth=val_target.to(device))  # Calculating loss over this batch
                        val_batch_losses.append(val_batch_cost.item())
                    print('Validation loss: {:.4f}'.format(statistics.mean(val_batch_losses)))

        save_model(epoch=epoch, model_state_params=mlp_model.state_dict(), optimizer_state_param=optimizer.state_dict(), loss=sum(batch_losses) / len(batch_losses))
