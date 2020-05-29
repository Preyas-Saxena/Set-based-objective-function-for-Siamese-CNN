
import torch
import Model3
import json
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

torch.manual_seed(2)

#Actual Values from the training dataset:
mean_data=[0.7170, 0.6794, 0.6613]
std_data=[0.2358, 0.2511, 0.2574]

normalize = transforms.Normalize(mean=mean_data, std=std_data)
transform_list=transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])

# rootdir='/home/preyas/Desktop/Thesis/polyvore/'
rootdir='/data/leuven/329/vsc32927/polyvore/'

file='fill_in_blank_test.json'

# imgdir = '/home/preyas/Desktop/Thesis/images/'
imgdir='/data/leuven/329/vsc32927/images/'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'

PATH='/data/leuven/329/vsc32927/Thesis_varnet/redo/margin3/24-02-2020_01:35:09_base.ckpt'
checkpoint = torch.load(PATH, map_location= torch.device('cpu'))

#Initializing a vgg net model with an embedding of 20 size, using a pre-trained model but will update all the weights
vgg = Model3.initialize_model("vgg", 128, feature_extract=False, use_pretrained=True)
vgg.load_state_dict(checkpoint['model_state_dict'])
vgg.to(device)
vgg.eval()

# Parser function:
def parser_fn(fn):

    data = json.load(open(str(rootdir + fn), 'r'))  # 3076 questions
    qa_pair = []
    for idx, item in enumerate(data):
        questions = item['question']
        answers = item['answers']
        q_pairs = []
        with torch.no_grad():
            for ques in questions:
                set_id = ques.split('_')[0]
                img_index = ques.split('_')[1]
                img_path = imgdir + str(set_id) + '/' + str(img_index) + '.jpg'
                img=Model3.load_train_items(path=img_path, transform=transform_list)
                q_pairs.append(img)
                gt = set_id
            qpairs_tensor=vgg(torch.stack(q_pairs).to(device))

            a_pairs = []
            for ans_index, ans in enumerate(answers):
                ans_set_id = ans.split('_')[0]
                img_index = ans.split('_')[1]
                img_path = imgdir + str(ans_set_id) + '/' + str(img_index) + '.jpg'
                img=Model3.load_train_items(path=img_path, transform=transform_list)
                a_pairs.append(img)

                if ans_set_id==gt:
                    correct_index=ans_index

            apairs_tensor = vgg(torch.stack(a_pairs).to(device))
            qa_pair.append((qpairs_tensor, apairs_tensor, correct_index))

    return qa_pair

def fitb_accuracy(fitb_pairs):
    correct = 0.
    n_questions = 0.

    for q_index, (questions_list, answers_list, correct_idx) in enumerate(fitb_pairs):
        answer_score = np.zeros(len(answers_list), dtype=np.float32)
        for a_index, answer in enumerate(answers_list):
            set = torch.cat((questions_list, answer.unsqueeze(0)), dim=0)
            set_variance = torch.mean(torch.var(set, dim=0))
            answer_score[a_index] = set_variance.item()

        if correct_idx == np.argmin(answer_score):
            correct += 1
        n_questions += 1

    acc = correct / n_questions
    return acc

qa_pairs= parser_fn(fn=file)
acc=fitb_accuracy(fitb_pairs=qa_pairs)
print('Accuracy is {:.4f}'.format(acc))

