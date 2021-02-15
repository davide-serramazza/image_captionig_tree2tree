from myCode.read_images_file import read_images
from myCode.read_sentence import label_tree_with_sentenceTree
import tensorflow as tf
import json
from random import  choice
from tensorflow_trees.definition import Tree
from myCode.read_sentence import label_tree_with_real_data
import myCode.shared_POS_words_lists as shared_list
import xml.etree.ElementTree as ET


######################

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

def load_flat_captions(json_file,val_data):
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
        label_tree_with_sentenceTree(train_data,val_data, args.targets)
    flat_val_caption = load_flat_captions(args.all_captions,val_data)
    return train_data,val_data,flat_val_caption


#######################

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
        default=200,
        help="Maximum number of iteration to train")

    tf.flags.DEFINE_integer(
        "check_every",
        default=5,
        help="How often (iterations) to check performances")

    tf.flags.DEFINE_integer(
        "save_model_every",
        default=20,
        help="How often (iterations) to save model")

    tf.flags.DEFINE_string(
        "model_dir",
        default="tensorboard/",
        help="Directory to put the model summaries, parameters and checkpoint.")


def get_image_arity(t ):
    max_arity = len(t.children)
    for child in t.children:
        actual_arity = get_image_arity(child)
        if actual_arity > max_arity:
            max_arity = actual_arity
    return max_arity

def get_sen_arity(tree : ET.Element):
    max_arity = len(tree.getchildren())
    for el in tree.getchildren():
        current_arity = get_sen_arity(el)
        if current_arity > max_arity:
            max_arity = current_arity
    return max_arity

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


def compute_max_arity(train_data,val_data):
    image_max_arity=0
    sen_max_arity=0
    for el in train_data+val_data:
        current_img_arity=get_image_arity(el['img_tree'])
        if current_img_arity>image_max_arity:
            image_max_arity=current_img_arity

        for caption in el['sentences']:
            current_sen_arity= get_sen_arity(caption)
            if current_sen_arity>sen_max_arity:
                sen_max_arity=current_sen_arity
    return image_max_arity, sen_max_arity

def get_input_target_minibatch(data,j,batch_size):
    images=[]
    captions=[]
    for i in range( j,min( j+batch_size,len(data) ) ):
        el = data[i]
        images.append(el['img_tree'])
        # choose a random caption
        caption = choice(el['sentences'])

        #build the tree now
        sentence_tree = Tree(node_type_id="dummy root", children=[],value="dummy")
        label_tree_with_real_data(caption, sentence_tree,shared_list.tokenizer)
        captions.append( sentence_tree.children[0] )

    return images,captions