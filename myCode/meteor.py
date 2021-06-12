import os.path

from pycocoevalcap.meteor.meteor import Meteor
import json
from os import listdir
mypath = "/home/davide/Desktop/flat"
files = listdir(mypath)

meteor = Meteor()

best_score = 0.0
best_file = ""
for file in files:
    file = open(os.path.join(mypath,file))
    lines = file.readlines()

    for el in lines:
        if el==" \n":
            lines.remove(el)

    with open("/home/davide/workspace/tesi/Flickr30k_captions.json") as f:
        dict = json.load(f)

    refs = {}
    preds = {}
    for i in range(0,len(lines),2):
        tmp = lines[i].replace("\n","").split(" : ")
        current_img = tmp[0]
        refs[current_img] = dict[current_img]
        preds[current_img]=[tmp[1]]

    score, scores = meteor.compute_score(refs,preds)
    if score>best_score:
        best_file = file
        best_score = score


print(best_score,best_file)