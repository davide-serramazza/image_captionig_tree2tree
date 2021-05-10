import os


def occ_only_in_sec(d1,d2):
    count = 0
    for k in d2:
        if not (d1.__contains__(k)):
            count += d2[k]
    print(count, " out occ of d2 are not in d1",)

def freq_count(file):
    global sen
    # open train, test, val split
    print(file)
    word_dict = {}
    n_word = 0
    tot_word = 0
    train = open(file)
    train = train.read().split("\n")
    for sen in train:
        if sen == "":
            break
        words = dict[sen[:-4]].split(" ")
        for w in words[:-1]:
            if w in word_dict.keys():
                word_dict[w] = word_dict[w] + 1
            else:
                word_dict[w] = 1
                tot_word += 1
            n_word += 1
    n_el= 0
    n_occ=0

    #select only word with more than 5 occurrency
    for k in word_dict.keys():
        if word_dict[k] < 6:
            n_el += 1
            n_occ += word_dict[k]
    print(n_el, " of ", tot_word, " have less then 5 occurancy, that is a percentage of", n_el/tot_word )
    print(n_occ, " of ",n_word , " have less then 5 occurancy, that is a percentage of ", n_occ/n_word)
    return word_dict

def read_parser_trees(what_extract):
    global r, d, f, files, dict, sen, file_path, line, tmp
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        files = f
    dict = {}
    for f in files:
        sen = ""
        file_path = open(path + f)
        line = "a"
        while line != "":
            line = file_path.readline()
            tmp = line.split(what_extract)
            if len(tmp) > 1:
                tmp = tmp[1].split('"/>')
                sen += tmp[0] + " "
        dict[f] = sen


def read_parser_trees_depth(what_extract, img_list=None):
    global r, d, f, files, dict, sen, file_path, line, tmp
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        files = f
    dict = {}
    tot_pos = 0
    for f in files:
        #if img_list is specified select just the images in this list
        if img_list!=None:
            if f not in test_content:
                continue
        sen = ""
        file_path = open(path + f)
        line = "a"
        while line != "":
            #extract pos tag and keep track of their occurancy depth
            line = file_path.readline()
            tmp = line.split(what_extract)
            if (len(tmp) > 1 and len(tmp[0])>1):
                pos = (tmp)[1][:-3]
                if pos in dict.keys():
                    current = dict[pos]
                    dict[pos] = ( current[0] + len(tmp[0])/2 , current[1] +1)
                else:
                    dict[pos] = (  len(tmp[0])/2 , 1 )
                tot_pos+=1

    #print distribution
    for k in dict.keys():
        print(k, dict[k][1]/tot_pos)

    #select only 20 pos tag more frquent
    dict = sorted(dict.items(), key= lambda  x : x[1])
    dict = dict[-20:]

    #create another dict and sort it by average occ
    pos_depth = []
    for el in dict:
        pos = el[0]
        occ = el[1]
        pos_depth.append( (pos , occ[0]/occ[1]) )
    pos_depth = sorted(pos_depth, key= lambda x: x[1])
    print(pos_depth)




path = '/home/davide/Desktop/parsed_s/'

files = []

test = open("/home/davide/Desktop/Flickr_8k.testImages.txt")
test_content = test.read()

pos_depth_test = read_parser_trees_depth('<node value="',test_content)
pos_depth_tot = read_parser_trees_depth('<node value="')

read_parser_trees('<leaf value="')

train_words = freq_count("/home/davide/Desktop/Flickr_8k.trainImages.txt")
val_words = freq_count("/home/davide/Desktop/Flickr_8k.devImages.txt")
test_words = freq_count("/home/davide/Desktop/Flickr_8k.testImages.txt")


dev_word = {**train_words, **val_words}
occ_only_in_sec(dev_word,test_words)
occ_only_in_sec(train_words,val_words)

read_parser_trees('<node value="')


freq_count("/home/davide/Desktop/Flickr_8k.trainImages.txt")
freq_count("/home/davide/Desktop/Flickr_8k.devImages.txt")
freq_count("/home/davide/Desktop/Flickr_8k.testImages.txt")