
import torch
import torch.nn.functional as F
import math
import torchvision.models as models
import torch.nn as nn
from datetime import datetime
from PIL import Image
import numpy as np
import random

# imgdir = '/home/preyas/Desktop/Thesis/images/'
imgdir='/data/leuven/329/vsc32927/images/'

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

#Function for saving model params at various checkpoints throughout training:
def save_model(epoch, model_state_params, optimizer_state_param, scheduler_state_param, loss):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    path_remote = '/data/leuven/329/vsc32927/Thesis_varcoeff/redo/margin3/' + dt_string + '_base.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_params,
        'optimizer_state_dict': optimizer_state_param,
        'scheduler_state_dict': scheduler_state_param,
        'loss': loss}, path_remote)
    return None

#Defining a default image loader given the image's path:
def load_train_items(path, transform):
    img= Image.open(path).convert('RGB')
    if transform is not None:
        return transform(img)
    else:
        return img

#Function for taking a batch of outfit_ids, and their respective number of items and converting into a batch of tensors:
def outfit_to_tensor(outfits, outfit_len, transforms):
    # getting batch size:
    batch_size = outfits.shape[0]

    # Initializing list to stack items from each outfit in the current batch:
    item_list = []
    item_label_list = []

    for idx in range(batch_size):
        outfit_id = outfits[idx].item()
        num_items = outfit_len[idx].item()

        for item in range(1, num_items + 1):
            img_path = imgdir + str(outfit_id) + '/' + str(item) + ".jpg"
            img_tensor = load_train_items(img_path, transform=transforms)
            item_list.append(img_tensor)
            item_label_list.append(torch.tensor(outfit_id))

    # Stacking items from item_list in the form of a torch tensor- which can be used as an input to DL models:
    stacked_items = torch.stack(item_list)
    stacked_item_labels = torch.stack(item_label_list)

    return stacked_items, stacked_item_labels

def adding_multi_negative_samples(items, lengths, labels):

    items_copy = items.clone()
    cumlen_tensor = torch.cumsum(lengths, dim=0)

    indices_batch = [i for i in range(items_copy.shape[0])]
    rand_indices_batch = []
    for idx, len in enumerate(cumlen_tensor):
        if idx == 0:
            indices_set = indices_batch[0:len.item()]  # indices of a set's elements
        else:
            indices_set = indices_batch[cumlen_tensor[idx - 1].item():len.item()]

        len_outfit = lengths[idx].item()
        to_replace = round(len_outfit / 2) #Replacing ~50percent of items to make a set in-consistent

        indices_rem = [x for x in indices_batch if x not in indices_set]
        indices_set_array = np.array(indices_set)
        indices_set_array[random.sample(range(indices_set.__len__()), to_replace)] = random.sample(indices_rem, to_replace)
        indices_set = indices_set_array.tolist()
        rand_indices_batch = rand_indices_batch + indices_set

    itemscopy_rand=items_copy[rand_indices_batch]
    combined_items = torch.cat((items, itemscopy_rand), dim=0)

    combined_lengths = torch.cat((lengths.unsqueeze(0), lengths.unsqueeze(0)), dim=1)
    combined_labels = torch.cat((labels.unsqueeze(0), torch.zeros(labels.shape[0]).unsqueeze(0).type(torch.int64)), dim=1)

    combined_lengths = combined_lengths.squeeze()
    combined_labels = combined_labels.squeeze()

    return combined_items, combined_lengths, combined_labels

#Calculates mean variance of set (across all dimensions)
def set_variance(embeds, outfit_sizes):
    # Both embeds and outfit_sizes are tensors obtained from the train/valid loader

    outfit_splits = torch.split(embeds, split_size_or_sections=outfit_sizes.tolist(), dim=0)  # tuple containing splits
    set_rep = []

    for i in outfit_splits:
        std= torch.std(i, dim=0)
        abs_mean= torch.mean(torch.abs(i), dim=0)
        set_rep.append(torch.mean(torch.div(std, abs_mean)))
    set_rep = torch.stack(set_rep)

    return set_rep

def pairwise_loss(set_vars, set_labels, device):

    # Comparing variance for each pair of sets in the mini-batch:
    margin = torch.tensor(0.3)
    variance_comp = set_vars.unsqueeze(1) - set_vars.unsqueeze(0) + margin

    # Getting labels mask for identifying which of the pairwise comparisons are valid:
    labels_mask = set_labels.unsqueeze(1) - set_labels.unsqueeze(0)
    labels_mask = torch.clamp(labels_mask, min=0)

    # Getting final variance comparison:
    variance_comp = torch.mul(variance_comp, labels_mask.type(torch.FloatTensor).to(device))
    variance_comp = torch.clamp(variance_comp, min=0.0)  # Clamping loss to zero

    # Calculating final loss:
    pos_pairs = torch.gt(input=variance_comp, other=torch.tensor(1e-16).to(device))
    num_pos_pairs = torch.sum(pos_pairs)
    num_valid_pairs = torch.sum(labels_mask)
    pos_pairloss = torch.sum(variance_comp) / (num_pos_pairs + torch.tensor(1e-16))

    with torch.no_grad():
        overall_pairloss= torch.sum(variance_comp)/(num_valid_pairs + torch.tensor(1e-16))
        mean_consistent_var= torch.mean(torch.mul(set_vars, set_labels.type(torch.FloatTensor).to(device)))
        mean_inconsistent_var= torch.mean(torch.mul(set_vars, (torch.tensor(1)-set_labels).type(torch.FloatTensor).to(device)))

    return pos_pairloss, overall_pairloss, num_pos_pairs, num_valid_pairs, mean_consistent_var, mean_inconsistent_var

def coeffnet(model, dataloader, valid_loader, optimizer, scheduler, epochs, transforms, device, validating):

    for epoch in range(1, epochs+1):

        batch_losses=[]
        total_pos_pairs=0.
        total_valid_pairs=0.

        print('Current Learning rate is: ', optimizer.param_groups[0]['lr'], 'and', scheduler.get_lr())

        for batch, (outfit_ids, len_outfits, outfit_labels) in enumerate(dataloader):
            model.train()
            # print(outfit_ids, len_outfits, outfit_labels)
            items_x, _= outfit_to_tensor(outfit_ids, len_outfits, transforms=transforms) #torch.Size([15, 3, 224, 224])

            #Getting embeddings for all items present in the sets present in the current mini-batch:
            batch_embeds = model(items_x.to(device)) #torch.Size([15, 128])

            # Calculating mean length(norm) of embeddings:
            with torch.no_grad():
                embeds_norm = torch.norm(batch_embeds, p=2, dim=1)
                mean_embeds_norm = torch.mean(embeds_norm)

            #Creating embeddings for inconsistent sets:
            total_items, total_lengths, total_labels = adding_multi_negative_samples(items=batch_embeds, lengths=len_outfits, labels=outfit_labels)

            #Getting set representation for all the sets created/present in the current mini-batch:
            set_vars = set_variance(embeds=total_items, outfit_sizes=total_lengths)

            #Getting variance-based loss based on comparison between consistent and in-consistent sets:
            loss, overall_loss, pos_pairs, valid_pairs ,consis_var, inconsis_var= pairwise_loss(set_vars=set_vars, set_labels=total_labels, device=device)

            if math.isnan(loss):
                break

            with torch.no_grad():
                batch_train_loss = loss.item() * pos_pairs.item()
                batch_losses.append(batch_train_loss)

                total_pos_pairs = pos_pairs.item() + total_pos_pairs
                total_valid_pairs = valid_pairs.item() + total_valid_pairs

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Printing the batch loss after every ith iteration:
            if (batch+1)%20 ==0:
                print('Epoch {}, Step {}/{}, Pairs: {}/{}, Pos_Pair_Loss: {:.3f}, Overall_Loss: {:.3f}, Consis_coeff: {:.3f}, Inconsis_coeff: {:.3f}, Embeds Norm: {:.1f}'.format(epoch, (batch + 1), len(dataloader), pos_pairs, valid_pairs, loss, overall_loss, consis_var, inconsis_var, mean_embeds_norm))

            #Printing cumulative loss after every 100 iterations:
            if (((batch+1)%400 == 0) or ((batch+1)==len(dataloader))):
                cum_loss= sum(batch_losses)/total_pos_pairs
                cum_overall_loss= sum(batch_losses)/total_valid_pairs
                print('Epoch {}, After {} batches, Loss: {:.4f}, Overall Loss: {:.4f}'.format(epoch,(batch+1), cum_loss, cum_overall_loss))

            if (((batch+1)%800 == 0) or ((batch+1)==len(dataloader))) and validating:
                print("Entering Validation mode")

                with torch.no_grad():
                    val_batch_losses = []
                    val_total_pos_pairs = 0.
                    val_total_valid_pairs = 0.

                    for val_batch, (val_outfit_ids, val_len_outfits, val_outfit_labels) in enumerate(valid_loader):
                        model.eval()

                        val_items_x, _ = outfit_to_tensor(val_outfit_ids, val_len_outfits,transforms=transforms)  # torch.Size([15, 3, 224, 224])

                        # Getting embeddings for all items present in the sets present in the current mini-batch:
                        val_batch_embeds = model(val_items_x.to(device))  # torch.Size([15, 128])

                        # Creating embeddings for inconsistent sets:
                        val_total_items, val_total_lengths, val_total_labels = adding_multi_negative_samples(items=val_batch_embeds, lengths=val_len_outfits, labels=val_outfit_labels)

                        # Getting set representation for all the sets created/present in the current mini-batch:
                        val_set_vars = set_variance(embeds=val_total_items, outfit_sizes=val_total_lengths)

                        #Getting validation loss:
                        val_loss, val_overall_loss,val_pos_pairs, val_valid_pairs, _, _ = pairwise_loss(set_vars=val_set_vars, set_labels=val_total_labels, device=device)

                        val_batch_loss= val_loss.item()* val_pos_pairs.item()
                        val_batch_losses.append(val_batch_loss)

                        val_total_pos_pairs = val_pos_pairs.item() + val_total_pos_pairs
                        val_total_valid_pairs = val_valid_pairs.item() + val_total_valid_pairs

                val_loss= sum(val_batch_losses)/val_total_pos_pairs
                val_overall_loss= sum(val_batch_losses)/val_total_valid_pairs
                print('Validation Set, Loss:{:.4f}, Overall Loss:{:.4f}'.format(val_loss, val_overall_loss))

            # Saving a checkpoint to monitor progress:
            if ((batch + 1) % 100) == 0:
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
                bpath_remote = '/data/leuven/329/vsc32927/Thesis_varcoeff/redo/margin3/ckpt/' + dt_string + '_epoch' + str(epoch) + '_batch' + str(batch) + '.ckpt'
                torch.save({'batch_idx': batch}, bpath_remote)

            # if (batch+1)==101:
            #     break

        scheduler.step()
        save_model(epoch=epoch, model_state_params=model.state_dict(), optimizer_state_param=optimizer.state_dict(), scheduler_state_param=scheduler.state_dict(),loss=sum(batch_losses)/total_valid_pairs)
