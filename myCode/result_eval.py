import sys
sys.path.insert(0,'/home/davide/workspace/tesi/cider')
from pydataformat.loadData import LoadData
from pyciderevalcap.eval import CIDErEvalCap as ciderEval
from nltk.translate.bleu_score import corpus_bleu
import os
import json

def bleus(references_dict ,to_test_dict):
    hyp = [el['caption'].split(" ") for el in to_test_dict]
    #corpus_bleu
    refs=[]
    for el in references_dict.items():
        current_img=[]
        for dict in el[1]:
            current_img.append(dict['caption'].split(" "))
        refs.append(current_img)
    print("bleu1 is ",corpus_bleu(refs,hyp,weights=(1.0,)))
    print("bleu2 is ",corpus_bleu(refs,hyp,weights=(0.5,0.5)))
    print("bleu3 is ",corpus_bleu(refs,hyp,weights=((1.0/3.0),(1.0/3.0),(1.0/3.0))))
    print("bleu4 is ",corpus_bleu(refs,hyp,weights=(0.25,0.25,0.25,0.25)))



def CIDEr (references_dict, toTest_dict):

    df_mode = 'corpus'
    scorer = ciderEval(references_dict, toTest_dict, df_mode)
    return scorer.evaluate()

def extract_reference(file):
    dict={}
    current_img_list=[]
    gts = open(file)
    current_img="1000268201_693b08cb0e"
    while True:
        line = gts.readline()
        if line=="":
            break
        imgId_caption = line.split("#")
        img_id = imgId_caption[0][:-4]
        caption=imgId_caption[1][2:-1]
        if current_img==img_id:
            current_img_list.append({'caption':caption})
        else:
            dict[img_id]=current_img_list
            current_img=img_id
            current_img_list =[{'caption':caption}]
    return dict



path = '/home/davide/workspace/tesi/tf_tree/pred/'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        with open(path+file, 'r') as f:
            predictions = json.load(f)
            predictions= sorted(predictions,key=lambda el:el['image_id'])
            current_imgs = [el['image_id'] for el in predictions]
            gts = extract_reference("/home/davide/Desktop/Flickr8k.token.txt")
            current_gts = {}
            for (k,v) in gts.items():
                if k in current_imgs:
                    current_gts[k]=v
            CIDEr(current_gts,predictions)
            bleus(current_gts,predictions)