
from torch.utils.data import Dataset
import json
from PIL import Image

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class style_classes(Dataset):

    def __init__(self,outfit_direc,outfit_files,titles_direc,titles_file):

        self.outfit_titles = json.load(open(str(titles_direc + titles_file),'r'))

        outfit_data=[]
        for file in outfit_files:
            outfit_data = outfit_data + json.load(open(str(outfit_direc + file),'r'))

        outfits_list=[]
        for outfit in outfit_data:
            outfits_list.append(outfit['set_id'])

        casuals=[]
        formals=[]
        offices=[]
        bohos=[]
        commons = []

        for outfit in self.outfit_titles:
            url_title = self.outfit_titles[outfit]
            desc = url_title['url_name']
            title = url_title['title']
            complete = desc + title

            if 'casual' in complete.lower():
                casuals.append((outfit,1))

            if 'formal' in complete.lower():
                formals.append((outfit,2))

            if 'office' in complete.lower():
                offices.append((outfit,3))

            if 'boho' in complete.lower() or 'bohemian' in complete.lower():
                bohos.append((outfit,4))

            if 'casual' in complete.lower() and 'formal' in complete.lower():
                commons.append(outfit)
            if 'casual' in complete.lower() and 'office' in complete.lower():
                commons.append(outfit)
            if 'casual' in complete.lower() and 'boho' in complete.lower():
                commons.append(outfit)
            if 'casual' in complete.lower() and 'bohemian' in complete.lower():
                commons.append(outfit)

            if 'formal' in complete.lower() and 'office' in complete.lower():
                commons.append(outfit)
            if 'formal' in complete.lower() and 'boho' in complete.lower():
                commons.append(outfit)
            if 'formal' in complete.lower() and 'bohemian' in complete.lower():
                commons.append(outfit)

            if 'office' in complete.lower() and 'boho' in complete.lower():
                commons.append(outfit)
            if 'office' in complete.lower() and 'bohemian' in complete.lower():
                commons.append(outfit)

        complete_set = casuals + formals + offices + bohos
        unique_set=[(x,y) for (x,y) in complete_set if x not in commons ]

        final_outfits= [(x,y) for (x,y) in unique_set if x in outfits_list]
        self.final_outfits= final_outfits

    def __len__(self):
        return len(self.final_outfits)

    def __getitem__(self, index):
        (outfit_id, type) = self.final_outfits[index]
        return int(outfit_id), type
