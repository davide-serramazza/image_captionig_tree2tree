import sys
sys.path.insert(0, '.')

import argparse
from myCode.helper_functions import compute_max_arity,define_flags,load_data
from myCode.models import *
from myCode.train_model import train_model
from myCode.tree_defintions import *
import os


def main():
    parser = argparse.ArgumentParser(description='data to use, modality and iper-Parameter ')
    parser.add_argument('train', type=str, help="Training set \
    (if provided a list it is assumed it contains the tree encoding, otherwise if dir it is assumed it contains image files ")
    parser.add_argument('val', type=str, help="Validation set. Same assumptions of train set")
    parser.add_argument('targets', type=str, help="dir containing a xml file for each images in which the parse tree\
     of the caption is described.This targets are used only for training")
    parser.add_argument('all_captions', type=str, help="file containing each image captions.\
    These are used for validation purposes e.g. for compute the blue scores against 5 or more caption")
    parser.add_argument('CNN_to_use', type=str, help="CNN used to encoding the images (in case of image input type None).\
                        The two valid options are inception and alexnet")
    #TODO fare argomenti opzionali
    parser.add_argument('emb_tree_word_size', type=int, help="dimension of tree embedding and word embedding (must be equal in NIC)")
    max_node_count = 200
    max_depth =  20
    cut_arity = 5
    parser.add_argument('lambd', type=float, help="coefficient of penalization on encoder representation")
    parser.add_argument('beta', type=float, help="coefficient for L2 reg.")
    parser.add_argument('hidden_coeff', type=float, help="coefficient for hidden dimension of encoder and decoder layer")
    learning_rate = 0.0001
    clipping = 0.02
    batch_size = 32
    parser.add_argument('rnn_unit_size', type=int, help="size of rnn expressed as coefficient relative to embedding size")
    parser.add_argument('drop_rate', type=float, help="drop rate for hidden layers")
    parser.add_argument('drop_rate_input', type=float, help="drop rate for input layers")

    ####################
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
    train_data, val_data, flat_val_captions = load_data(args, tree_enoder, tree_decoder, cnn_type, batch_size)

    image_max_arity, sen_max_arity = compute_max_arity(train_data, val_data,tree_enoder)
    print(image_max_arity,sen_max_arity)

    activation = getattr(tf.nn, FLAGS.activation)
    decoder, encoder = get_encoder_decoder(emb_tree_size=args.emb_tree_word_size,cut_arity=cut_arity,max_arity=
    max(image_max_arity,sen_max_arity),max_node_count=max_node_count,max_depth=max_depth,hidden_coeff=args.hidden_coeff,
                                           activation=activation,image_tree=image_tree,sentence_tree=sentence_tree,emb_word_size=args.emb_tree_word_size,
                                           hidden_word=args.rnn_unit_size,drop_rate=args.drop_rate,drop_rate_input=args.drop_rate_input)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    print("begin train")
    name = "emb_dim_"+str(args.emb_tree_word_size)+"_rnn_units_"+str(args.rnn_unit_size)+"_lambda_"+str(args.lambd)+ \
           " _beta_"+str(args.beta)+"_hidden_coeff_"+str(args.hidden_coeff)[2:]+"_drop_rate_" + str(args.drop_rate)
    train_model(FLAGS=FLAGS, decoder=decoder,
                encoder=encoder, train_data=train_data, val_data=val_data, optimizer=optimizer,
                beta=args.beta, lamb=args.lambd, clipping=clipping, batch_size=batch_size,
                tree_encoder =not(image_tree==None), tree_decoder = not(sentence_tree==None),
                flat_val_captions=flat_val_captions, tensorboard_name=name)


if __name__ == "__main__":
    main()
