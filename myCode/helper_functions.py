import os

from myCode.read_images_file import read_images
from myCode.read_sentence import label_tree_with_sentenceTree
import tensorflow as tf
import json
from random import  choice
from tensorflow_trees.definition import Tree
from myCode.read_sentence import label_tree_with_real_data, get_flat_captions
import myCode.shared_POS_words_lists as shared_list
import xml.etree.ElementTree as ET
import tensorflow.contrib.summary as tfs
from xml.dom import minidom

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

def load_flat_captions(data,val_data):

    val_all_captions = []
    for el in val_data:
        name = el["name"]
        val_all_captions.append([])
        current_img_captions = data[name]
        for sentence in current_img_captions:
            val_all_captions[-1].append( sentence )
    return val_all_captions

def load_data(args,tree_encoder,tree_decoder,tree_cnn_type,batch_size):
    print('loading images....')
    image_features_extract_model = istanciate_CNN(tree_encoder)
    train_data = read_images(args.train,image_features_extract_model,tree_cnn_type,batch_size)
    val_data = read_images(args.val,image_features_extract_model,tree_cnn_type,batch_size)
    if args.test!='None':
        test_data = read_images(args.test,image_features_extract_model,tree_cnn_type,batch_size)
        train_data = train_data+val_data
        val_data = test_data
    print('loading sentences...')
    with open(args.all_captions) as json_file:
        flat_captions = json.load(json_file)
    if tree_decoder:
        sen_max_len = label_tree_with_sentenceTree(train_data,val_data, args.targets)
    else:
        sen_max_len = get_flat_captions(train_data,val_data, flat_captions)

    flat_val_caption = load_flat_captions(flat_captions,val_data)

    return train_data,val_data,flat_val_caption, sen_max_len


def extract_words(predictions, beam, val_data ,it_n ,name):
    predicted_words=tf.unstack( tf.argmax(predictions,axis=-1) ) if not beam else [tf.argmax(el,axis=-1) for el in predictions]
    sentences = []
    with open("pred_sens/"+name+"_it="+str(it_n)+"_beam="+str(beam)+".txt", "w+") as file:
        for (el,sen) in zip(val_data,predicted_words):
            s=""
            for word in sen:
                word = shared_list.tokenizer.index_word[ word.numpy()]
                if word=='<end>':
                    break
                else:
                    s+=( word +" ")
            sentences.append(s)
            file.write(el["name"]+" : "+s+"\n")
    return sentences

def extract_words_from_tree(trees, beam, val_data ,it_n ,name):
    to_return = []

    with open("pred_sens/"+name+"_it="+str(it_n)+"_beam="+str(beam)+".txt", "w+") as file:
        dir_name="pred_tree/"+name+"_it="+str(it_n)+"_beam="+str(beam)+"/"
        os.mkdir(dir_name)
        for (tree,img) in zip (trees,val_data):
            current_pred = []

            root = ET.Element('root')
            take_word_vectors(tree,current_pred,root)
            xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")

            with open(dir_name+img['name'],'w+') as f:
                f.write(xmlstr)

            s=""
            for el in current_pred:
                s+=(el+" ")
            to_return.append(s)

            file.write(img["name"]+" : "+s+"\n")

    return to_return

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
        default=1001,
        help="Maximum number of iteration to train")

    tf.flags.DEFINE_integer(
        "check_every",
        default=30,
        help="How often (iterations) to check performances")

    tf.flags.DEFINE_integer(
        "save_model_every",
        default=20,
        help="How often (iterations) to save model")

    tf.flags.DEFINE_string(
        "model_dir",
        default="tensorboard/",
        help="Directory to put the model summaries, parameters and checkpoint.")


def get_image_arity(t):
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

def take_word_vectors(t ,l:list,node):
    if t.node_type_id=="word":
        l.append(t.value.abstract_value)
        node.tag = 'word'
        node.attrib['value'] = t.value.abstract_value
    else:
        node.tag = 'POS'
        try:
            node.attrib['value'] = t.value.abstract_value
        except AttributeError:
            node.attrib['value'] = 'TRUNCATED'
    for c in t.children:
        child = ET.SubElement(node,'')
        take_word_vectors(c,l,child)


def compute_max_arity(train_data,val_data,tree_encoder,tree_decoder):
    image_max_arity=0
    sen_max_arity=0
    for el in train_data+val_data:
        if tree_encoder:
            current_img_arity=get_image_arity(el['img_tree'])
            if current_img_arity>image_max_arity:
                image_max_arity=current_img_arity

        if tree_decoder:
            for caption in el['sentences']:
                current_sen_arity= get_sen_arity(caption)
                if current_sen_arity>sen_max_arity:
                    sen_max_arity=current_sen_arity
    return image_max_arity, sen_max_arity


def get_input_target_minibatch(data,j,batch_size,tree_encoder,tree_decoder):
    images=[]
    captions=[]
    for i in range( j,min( j+batch_size,len(data) ) ):
        el = data[i]
        if tree_encoder:
            images.append(el['img_tree'])
        else:
            images.append(el['img'])
        # choose a random caption
        if tree_decoder:
            caption = choice(el['sentences'])
            #build the tree now
            sentence_tree = Tree(node_type_id="dummy root", children=[],value="dummy")
            label_tree_with_real_data(caption, sentence_tree,shared_list.tokenizer)
            captions.append( sentence_tree.children[0] )
        else:
            captions.append('<start> ' + choice(el['sentences']) + ' <end>' )

    if not tree_encoder:
        images  = tf.convert_to_tensor(images)
    if not tree_decoder:
        all_captions = shared_list.tokenizer.texts_to_sequences(captions)
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(all_captions, padding='post')
        captions = tf.convert_to_tensor(cap_vector)
    return images,captions

def dump_in_tensorboard(losses_current_it,loss_validation=None,res_sampling=None,res_beam=None,tree_comparison_ris=None):
    if loss_validation==None:
        tfs.scalar("loss/struct", losses_current_it["struct"])
        tfs.scalar("loss/pos", losses_current_it["pos"])
        tfs.scalar("loss/word", losses_current_it["word"])
    elif losses_current_it==None:
        tfs.scalar("loss/validation_struct", loss_validation["struct"])
        tfs.scalar("loss/validation_pos", loss_validation["pos"])
        tfs.scalar("loss/validation_word", loss_validation["word"])
        tfs.scalar("metric/sampling/CIDEr", res_sampling["CIDEr"])
        tfs.scalar("metric/sampling/BLeu_1", res_sampling["Bleu"][0])
        tfs.scalar("metric/sampling/BLeu_2", res_sampling["Bleu"][1])
        tfs.scalar("metric/sampling/BLeu_3", res_sampling["Bleu"][2])
        tfs.scalar("metric/sampling/BLeu_4", res_sampling["Bleu"][3])
        tfs.scalar("metric/sampling/Rouge", res_sampling["Rouge"])
        tfs.scalar("metric/beam/CIDEr", res_beam["CIDEr"])
        tfs.scalar("metric/beam/BLeu_1", res_beam["Bleu"][0])
        tfs.scalar("metric/beam/BLeu_2", res_beam["Bleu"][1])
        tfs.scalar("metric/beam/BLeu_3", res_beam["Bleu"][2])
        tfs.scalar("metric/beam/BLeu_4", res_beam["Bleu"][3])
        tfs.scalar("metric/beam/Rouge", res_beam["Rouge"])

        tfs.scalar("overlaps/struct_avg", tree_comparison_ris["overlaps_s"])
        tfs.scalar("overlaps/struct_acc", tree_comparison_ris["overlaps_v"])
        tfs.scalar("overlaps/tot_pos", tree_comparison_ris["tot_pos"])
        tfs.scalar("overlaps/matched_pos", tree_comparison_ris["matched_pos"])
        tfs.scalar("overlaps/tot_word", tree_comparison_ris["tot_word"])
        tfs.scalar("overlaps/matched_word", tree_comparison_ris["matched_word"])

