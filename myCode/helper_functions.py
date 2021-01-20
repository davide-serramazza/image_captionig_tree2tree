from myCode.read_images_file import read_images
from myCode.read_sentence import label_tree_with_sentenceTree,extraxt_topK_words
import tensorflow as tf
from random import randrange, uniform
from random import shuffle
import numpy as np
import json
import random
import myCode.shared_POS_words_lists as shared_list

######################
#functions to extract all the image trees and all the sentence trees

def get_image_batch(train_data,val_data,flat_encoder):
    def f(data,flat_encoder):
        to_return = []
        for el in data:
            if flat_encoder:
                to_return.append(el["img"])
            else:
                to_return.append(el["img_tree"])
        return to_return

    return f(train_data,flat_encoder), f(val_data,flat_encoder)

def read_sentences_from_file(arg2,train_data,val_data):
    def foo(arg2,data):
        f = open(arg2)
        f = f.read().splitlines()
        captions = []
        for el in data:
            for cap in f:
                tmp = cap.split(" : ")
                if tmp[0]==el["name"]:
                    captions.append('<start> '+tmp[1][:-1]+' <end>')
        return captions

    train_captions = foo(arg2,train_data)
    val_captions = foo(arg2,val_data)
    tokenizer,top_k = extraxt_topK_words(train_captions,filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    targets_train = tokenizer.texts_to_sequences(train_captions)
    target_val = tokenizer.texts_to_sequences(val_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(targets_train+target_val, padding='post')
    target_train = tf.one_hot(cap_vector[:len(targets_train)],depth=top_k)
    target_val = tf.one_hot(cap_vector[len(targets_train):],depth=top_k)
    shared_list.word_idx = tokenizer.word_index
    return target_train,target_val

def get_sentence_batch(train_data,val_data,tree_decoder,arg2):
    def f(data):
        to_return=[]
        for el in data:
            to_return.append(random.choice(el['sentence_trees']))
        return to_return
    if tree_decoder:
        return f(train_data), f(val_data)
    else:
        return read_sentences_from_file(arg2,train_data,val_data)

def istanciate_CNN(tree_encoder):
    if not tree_encoder:
        # instanciate inception cnn
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    else:
        image_features_extract_model = None
    return image_features_extract_model

def load_all_captions(json_file,val_data):
    with open(json_file) as json_file:
        data = json.load(json_file)
    val_all_captions = []
    for el in val_data:
        name = el["name"]
        val_all_captions.append([])
        current_img_captions = data[name]
        for sentence in current_img_captions:
            val_all_captions[-1].append( sentence.split(" ") )
    return val_all_captions

def load_data(args,tree_encoder,tree_decoder,tree_cnn_type):
    print('loading image trees....')
    image_features_extract_model = istanciate_CNN(tree_encoder)
    train_data = read_images(args.train,image_features_extract_model,tree_cnn_type)
    val_data = read_images(args.val,image_features_extract_model,tree_cnn_type)
    print('loading sentence trees...')
    if tree_decoder:
        label_tree_with_sentenceTree(train_data, val_data, args.targets)
    val_all_captions = load_all_captions(args.all_captions,val_data)
    return train_data,val_data,val_all_captions

#def load_all_data(args,):
    #print('loading image trees....')
    #train_data = read_images(args[0])
    #val_data = read_images(args[1])
    #test_data = read_images(args[3])
    #print('loading sentence trees...')
    #label_tree_with_sentenceTree(train_data+val_data, test_data, args[2])
#return train_data+val_data,test_data

#def laod_test_data(args,dictionary, embeddings):
    #print('loading image trees....')
    #test_data = read_images(args[0])
    #print('loading sentence trees...')
    #label_tree_with_sentenceTree(test_data,args[3],embeddings,dictionary)
    #return test_data

#######################

def help():
    """
    function exlaning argumen to be passed
    :return:
    """
    print("1 -> train set file\n2 -> validation file\n3 -> parsed sentence dir")


def define_flags():
    """
    function that define flags used later in the traning
    :return:
    """
    tf.flags.DEFINE_string(
        "activation",
        default='tanh',
        help="activation used where there are no particular constraints")

    tf.flags.DEFINE_integer(
        "max_iter",
        default=61,
        help="Maximum number of iteration to train")

    tf.flags.DEFINE_integer(
        "check_every",
        default=10,
        help="How often (iterations) to check performances")

    tf.flags.DEFINE_integer(
        "save_model_every",
        default=20,
        help="How often (iterations) to save model")

    tf.flags.DEFINE_string(
        "model_dir",
        default="tensorboard/",
        help="Directory to put the model summaries, parameters and checkpoint.")


def select_one_random(list):
    """
    function to select one random item within the given list (used in parameter selection for random search)
    :param list:
    :return:
    """
    return list [randrange(len(list))]

def select_one_in_range(list, integer):
    """
    function to select a random value in the given range
    :param list:
    :param integer:
    :return:
    """

    rand = uniform(list[0], list[1])

    if integer:
        return round(rand)
    else:
        return rand

def shuffle_dev_set (train, validation):
    """
    function to shuffle train set and validation set
    :param train:
    :param validation:
    :return: new train and validation with shuffled item and keeping the same proportion of the orginal
    ones
    """
    dev_set = train + validation
    tot_len = len(dev_set)
    prop = float( len(train)  ) / float( len(train) + len(validation) )
    train_end = int (float(tot_len)*prop)
    shuffle(dev_set)
    return dev_set[:train_end] , dev_set[train_end:]

def shuffle_data(input,target,len_input):
    assert input.shape[0] == len(target)
    perm = np.random.permutation([i for i in range(0,len_input)])
    input_shuffled =  tf.gather(input,[i for i in perm])
    target_shuffled = [target[i] for i in perm]
    input = None
    target = None
    return input_shuffled, target_shuffled

def max_arity (list):
    """
    funtion to get the msx_arity in data set
    :param list: list of tree(dataset)
    :return:
    """
    max_arity = 0
    for el in list:
        actual_arity = get_tree_arity(el)
        if actual_arity > max_arity:
            max_arity = actual_arity
    return max_arity


def get_tree_arity(t ):
    max_arity = len(t.children)
    for child in t.children:
        actual_arity = get_tree_arity(child)
        if actual_arity > max_arity:
            max_arity = actual_arity
    return max_arity


def get_max_arity(input_train, input_val, target_train, target_val):
    # compute max_arity
    train_image_max_arity = max_arity(input_train)
    val_image_max_arity = max_arity(input_val)
    image_max_arity = max(train_image_max_arity, val_image_max_arity)
    train_sen_max_arity = max_arity(target_train)
    val_sen_max_arity = max_arity(target_val)
    sen_max_arity = max(train_sen_max_arity, val_sen_max_arity)
    return image_max_arity, sen_max_arity

def take_word_vectors(t ,l:list):
    if t.node_type_id=="word":
        l.append(t.value.abstract_value)
    for c in t.children:
        take_word_vectors(c,l)


def extract_words_from_tree(trees):
    to_return = []
    for tree in trees:
        current_pred = []
        take_word_vectors(tree,current_pred)
        to_return.append(current_pred)
    return to_return


def compute_max_arity(input_train, input_tree, target_train, target_tree):
    if input_tree != None:
        train_image_max_arity = max_arity(input_train)
        val_image_max_arity = 0
        image_max_arity = max(train_image_max_arity, val_image_max_arity)
    else:
        image_max_arity = 0
        input_train = tf.Variable(input_train)
        input_train = tf.squeeze(input_train)
    if target_tree != None:
        train_sen_max_arity = max_arity(target_train)
        val_sen_max_arity = 10
        sen_max_arity = max(train_sen_max_arity, val_sen_max_arity)
    else:
        sen_max_arity = 0
    return image_max_arity, input_train, sen_max_arity
