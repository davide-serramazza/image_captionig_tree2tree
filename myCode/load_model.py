import sys
sys.path.insert(0,'/home/serramazza/tf_tree_new')

from myCode.tree_defintions import *
from myCode.helper_functions import *
from myCode.models import get_encoder_decoder
import tensorflow.contrib.eager as tfe
import myCode.shared_POS_words_lists as dict
import json
from tensorflow_trees.definition import Tree

tag_dictionary = None
embeddings = None
s=""
pos_dict={}
total_pos_number=0

def get_predict_embeddings(tree):
    global s
    if tree.node_type_id=="word":
        s += " " +  tree.value.abstract_value
    else:
        for child in tree.children:
            get_predict_embeddings(child)


def predict_test_dataSet(decoder, encoder, input_test, names_data, file_name):
    global s
    predicted = []
    batch_enc = encoder(input_test)
    batch_dec = decoder(encodings=batch_enc.get_root_embeddings())
    i=0
    for tree_dec, name in zip(batch_dec.decoded_trees, names_data):
        get_predict_embeddings(tree_dec)
        predicted.append({'image_id':name,
                          'caption': s})
        s = ""
        i+=1
    with open(file_name, 'w') as file:
        json.dump(predicted, file, sort_keys=True, indent=4)


def restore_model(decoder, encoder,path):
    file = tf.train.latest_checkpoint(path)
    tfe.Saver((encoder.weights)).restore(file)
    tfe.Saver((encoder.variables)).restore(file)
    tfe.Saver((decoder.weights)).restore(file)
    tfe.Saver((decoder.variables)).restore(file)

def pos_depth(tree,depth):
    global pos_dict
    global total_pos_number
    if tree.node_type_id == "POS_tag":
        pos = tree.value.abstract_value
        if pos in pos_dict:
            (dic_depth, occ) = pos_dict[pos]
            pos_dict[pos] = ((dic_depth+depth), occ +1)
        else:
            pos_dict[pos] = (depth,1)
        total_pos_number+=1
    #recursion
    for child in tree.children:
        pos_depth(child,depth+1)


def main():
    global embeddings
    global tag_dictionary

    args = sys.argv[1:]

    if len(args)==0:
        print("1 -> train set file(maybe smaller)\n2 -> validation file(maybe smaller)\n"
              "3 -> embedding dictionaty\n4 -> tag dictionary\n"
              "5 -> parsed sentence dir\n5 -> test data file")
        exit()

    ##################
    # FLAGS
    ##################
    FLAGS = tf.flags.FLAGS
    define_flags()
    tf.enable_eager_execution()

    ###################
    # tree definition
    ###################
    image_tree = ImageTree()
    sentence_tree = SentenceTree()


    #load input and val tree for fake forward
    dev_data, test_data = load_all_data(args, dict.tags, dict.embeddings)
    input_dev = get_image_batch(dev_data)
    target_dev = get_sentence_batch(dev_data)
    input_test = get_image_batch(test_data)
    target_test = get_sentence_batch(test_data)

    #instanciate model
    activation = getattr(tf.nn, FLAGS.activation)

    #parameters

    embedding_size = 100
    cut_arity = 4
    hidden_coeff = 0.5
    max_node_count = 100
    max_depth = 5
    image_max_arity = 12
    sen_max_arity = 12

    decoder, encoder = get_encoder_decoder(emb_size=embedding_size,cut_arity=cut_arity,max_arity=
    max(image_max_arity,sen_max_arity),max_node_count=max_node_count,max_depth=max_depth,
                                           hidden_coeff=hidden_coeff,activation=activation,
                                           image_tree=image_tree.tree_def,sentence_tree=sentence_tree.tree_def, var_arity_decoder="FLAT")

    #forward with train data (otherwise Savr.resoterdoen't work)
    batch_enc = encoder(input_dev)
    decoder(encodings=batch_enc.get_root_embeddings(), targets=target_dev)

    #restoring model
    files = ["save.ckpt-9996", "save.ckpt-29596", "save.ckpt-49196", "save.ckpt-68796","save.ckpt-88396","save.ckpt-107996"]
    i=0
    for file in files:
        print("restoring model", i*100+50)
        restore_model(decoder, encoder,"/home/serramazza/tf_tree/saved_model_lungo_shuffled2/"+file)

        #performe analysis
        batch_val_enc = encoder(input_test)
        batch_val_dec = decoder(encodings=batch_val_enc.get_root_embeddings())

        for tree in batch_val_dec.decoded_trees:
            pos_depth(tree,1)

        for k in pos_dict:
            print( "average depth of ",k ," is ", pos_dict[k][0]/pos_dict[k][1])

        for k in pos_dict:
            print( "amount of post being ",k ," is ", pos_dict[k][1]/total_pos_number)

        s_avg, v_avg,tot_pos,matched_pos, tot_word, matched_word = Tree.compare_trees(target_test, batch_val_dec.decoded_trees)
        print("strcture", s_avg," ,value", v_avg)
        print(matched_pos, " out of ", tot_pos, " POS tag that is a pecentage of " ,matched_pos/tot_pos)
        print(matched_word, " out of ", tot_word, " word that is a pecentage of ",matched_word/tot_word)
        print("\n\n\n\n")
        i+=1

if __name__ == "__main__":
    main()
