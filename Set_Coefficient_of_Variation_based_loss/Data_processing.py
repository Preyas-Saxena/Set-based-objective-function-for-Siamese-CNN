

from torch.utils.data import Dataset
import json

class style_labels(Dataset):

    def __init__(self, datadir, filename):
        self.outfit_data = json.load(open(str(datadir + filename), 'r'))

        tot_outfits=0
        outfit_labels=[]
        for outfit in self.outfit_data:
            tot_outfits=tot_outfits+1
            outfit_id=outfit['set_id']
            items= outfit['items']
            outfit_labels.append([outfit_id, len(items)])

        self.tot_outfits= tot_outfits
        self.outfit_labels= outfit_labels

    def __len__(self):
        return self.tot_outfits

    def __getitem__(self, idx):
        outfit_id, num_items= self.outfit_labels[idx]
        label = 1
        return int(outfit_id), int(num_items), int(label)

