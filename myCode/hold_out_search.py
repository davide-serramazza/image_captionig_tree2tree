import sys
sys.path.insert(0, '.')

from myCode.tree_defintions import *
from myCode.helper_functions import *
from myCode.validation import validation
from tensorflow_trees.definition import Tree
import numpy as np
from  myCode.tree_defintions import TagValue
import os
import argparse
import xml.etree.cElementTree as ET

def main():
    global embeddings
    global tag_dictionary

    # check arguments
    args = sys.argv[1:]
    #TODO sistemare sta cosa
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('train')
    parser.add_argument('val')
    parser.add_argument('targets')
    parser.add_argument('all_captions')
    parser.add_argument('test')
    parser.add_argument('CNN_type')
    args = parser.parse_args()

    print("begin")

    ##################
    # FLAGS
    ##################
    FLAGS = tf.flags.FLAGS
    define_flags()
    tf.enable_eager_execution()

    ###################
    # tree definition
    ###################
    tree_enoder = os.path.isfile(args.train)
    cnn_type = args.CNN_type if tree_enoder else None
    image_tree = ImageTree(cnn_type) if tree_enoder else None
    tree_decoder = os.path.isdir(args.targets)
    sentence_tree = SentenceTree() if tree_decoder else None

    # load tree
    train_data, val_data, val_all_captions = load_data(args,tree_enoder,tree_decoder,cnn_type)
    print(len(train_data),len(val_data),len(val_all_captions)   )
    # get batch for traning
    input_train,input_val = get_image_batch(train_data,val_data,image_tree==None)
    target_train, target_val = get_sentence_batch(train_data,val_data,tree_decoder,args.targets)

    # define parameters to search:
    parameters = []
    parameters.append([300])  # embedding_size
    parameters.append([100])  # max node count
    parameters.append([10])  # max_depth
    parameters.append([4])  # cut_arity
    parameters.append([0.000])  # lambda tree2tree 0.0005,0.001 tree_decoder 0.0,0.0005
    parameters.append([0.000])  # beta tree2tree 0.01 tree_decoder 0.005
    parameters.append([0.3])  # hidden_coefficient
    parameters.append([0.001])  # learning
    parameters.append([0.02])  # clip gradient
    parameters.append([6,6])  # batch size
    parameters.append([200])  # word embedding #300
    parameters.append([1.0])  # hidden word size1.3
    parameters.append([0.4,0.5])   #dropout (keep probability)

    #tree_decoder
    print("begin experiments")
    validation(input_train, target_train,input_val, target_val, parameters, FLAGS, image_tree,
               sentence_tree,name="to_report/flat",val_all_captions=val_all_captions)




if __name__ == "__main__":
    main()
