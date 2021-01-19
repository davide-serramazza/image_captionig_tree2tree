import sys
sys.path.insert(0, '/home/davide/workspace/tesi/')

from myCode.tree_defintions import *
from myCode.helper_functions import *
from myCode.validation import validation
import argparse
import os

def main():
    # check arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('train', type=str, help="Training set \
    (if provided a list it is assumed it contains the tree encoding, otherwise if dir it is assumed it contains image files ")
    parser.add_argument('val', type=str, help="Validation set. Same assumptions of train set")
    parser.add_argument('targets', type=str, help="dir containing a xml file for each images in which the parse tree\
     of the caption is described.This targets are used only for training")
    parser.add_argument('all_captions', type=str, help="file containing each image captions not just one.\
    This is used only for validation purposes as for compute the blue scores against 5 caption")
    parser.add_argument('CNN_to_use', type=str, help="CNN used to encoding the images (in case of image input type None).\
                        The two valid options are inception and alexnet")
    args = parser.parse_args()

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
    cnn_type = args.CNN_to_use if tree_enoder else None
    image_tree = ImageTree(cnn_type) if tree_enoder else None
    tree_decoder = os.path.isdir(args.targets)
    sentence_tree = SentenceTree() if tree_decoder else None

    # load tree
    train_data, val_data, val_all_captions = load_data(args,tree_enoder,tree_decoder,cnn_type)
    # get batch for training
    #TODO
    input_train,input_val = get_image_batch(train_data,val_data,image_tree==None)
    target_train, target_val = get_sentence_batch(train_data,val_data,tree_decoder,args.targets)

    # define parameters to search:
    parameters = []
    parameters.append([300])  # embedding_size
    parameters.append([100])  # max node count
    parameters.append([10])  # max_depth
    parameters.append([4])  # cut_arity
    parameters.append([0.001])  # lambda #0.05,0.005,0.0005
    parameters.append([0.007])  # beta
    parameters.append([0.3])  # hidden_coefficient
    parameters.append([0.001])  # learning
    parameters.append([0.02])  # clip gradient
    parameters.append([6,6])  # batch size
    parameters.append([200])  # word embedding #300
    parameters.append([1.0])  # hidden word size1.3
    parameters.append([0.35,0.4,0.45])   #dropout (keep probability)

    #tree_decoder
    print("begin experiments")
    validation(input_train, target_train,input_val, target_val, parameters, FLAGS, image_tree,
               sentence_tree,name="tree_decoder_tutorial_dropoutInLSTM",val_all_captions=val_all_captions)




if __name__ == "__main__":
    main()
