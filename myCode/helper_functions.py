from myCode.read_images_file import read_images
from myCode.read_sentence import label_tree_with_sentenceTree
import tensorflow as tf
import json
from random import  choice
from tensorflow_trees.definition import Tree
from myCode.read_sentence import label_tree_with_real_data
import myCode.shared_POS_words_lists as shared_list
import xml.etree.ElementTree as ET
import tensorflow.contrib.summary as tfs


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

def load_data(args,tree_encoder,tree_decoder,tree_cnn_type,batch_size):
    print('loading image trees....')
    image_features_extract_model = istanciate_CNN(tree_encoder)
    train_data = read_images(args.train,image_features_extract_model,tree_cnn_type,batch_size)
    val_data = read_images(args.val,image_features_extract_model,tree_cnn_type,batch_size)
    if args.test!=None:
        a = 2
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


def compute_max_arity(train_data,val_data,tree_encoder):
    image_max_arity=0
    sen_max_arity=0
    for el in train_data+val_data:
        if tree_encoder:
            current_img_arity=get_image_arity(el['img_tree'])
            if current_img_arity>image_max_arity:
                image_max_arity=current_img_arity

        for caption in el['sentences']:
            current_sen_arity= get_sen_arity(caption)
            if current_sen_arity>sen_max_arity:
                sen_max_arity=current_sen_arity
    return image_max_arity, sen_max_arity


def get_input_target_minibatch(data,j,batch_size,tree_encoder):
    images=[]
    captions=[]
    for i in range( j,min( j+batch_size,len(data) ) ):
        el = data[i]
        if tree_encoder:
            images.append(el['img_tree'])
        else:
            images.append(el['img'])
        # choose a random caption
        caption = choice(el['sentences'])

        #build the tree now
        sentence_tree = Tree(node_type_id="dummy root", children=[],value="dummy")
        label_tree_with_real_data(caption, sentence_tree,shared_list.tokenizer)
        captions.append( sentence_tree.children[0] )

    if not tree_encoder:
        images  = tf.convert_to_tensor(images)
    return images,captions

class Summary ():
    def __init__(self,train):
        self.loss_struct = 0
        self.loss_value = 0
        self.loss_POS = 0
        self.loss_word = 0
        self.h_norm_it= 0
        self.w_norm_it= 0
        #grads
        self.gnorm_tot_it= 0
        self.gnorm_enc_it=0
        self.gnorm_dec_it=0
        self.gnorm_RNN_it=0
        self.gnorm_wordModule_it=0
        #grads clipped
        self.gnorm_tot_it_clipped= 0
        self.gnorm_enc_it_clipped=0
        self.gnorm_dec_it_clipped=0
        self.gnorm_RNN_it_clipped=0
        self.gnorm_wordModule_it_clipped=0

        self.n_miniBatch = 0
        self.train = train


    def add_miniBatch_summary(self,loss_struct_miniBatch,loss_value_miniBatch,loss_value_miniBatch_pos,loss_value_miniBatch_word,
                              h_norm,w_norm,gnorm,grad,grad_clipped,variables):
        self.loss_struct += loss_struct_miniBatch
        self.loss_value += loss_value_miniBatch
        self.loss_POS += loss_value_miniBatch_pos
        self.loss_word +=  loss_value_miniBatch_word
        self.h_norm_it += h_norm
        self.w_norm_it += w_norm

        enc_v=[]
        dec_v=[]
        rnn_v=[]
        word_module_v=[]

        enc_v_clipped=[]
        dec_v_clipped=[]
        rnn_v_clipped=[]
        word_module_v_clipped=[]

        for (g,gc,v) in zip (grad,grad_clipped,variables):
            if "cnn__encoder" in v.name:
                enc_v.append(g)
                enc_v_clipped.append(gc)
            elif 'final_word_pred' in v.name or 'word_embedding' in v.name:
                word_module_v.append(g)
                word_module_v_clipped.append(gc)
            elif 'LSTM' in v.name:
                word_module_v.append(g)
                rnn_v.append(g)
                word_module_v_clipped.append(gc)
                rnn_v_clipped.append(gc)
            else:
                dec_v.append(g)
                dec_v_clipped.append(gc)

        self.gnorm_tot_it += gnorm
        self.gnorm_enc_it += tf.global_norm(enc_v)
        self.gnorm_dec_it += tf.global_norm(dec_v)
        self.gnorm_RNN_it += tf.global_norm(rnn_v)
        self.gnorm_wordModule_it += tf.global_norm(word_module_v)

        self.gnorm_tot_it_clipped += tf.global_norm(grad_clipped)
        self.gnorm_enc_it_clipped += tf.global_norm(enc_v_clipped)
        self.gnorm_dec_it_clipped += tf.global_norm(dec_v_clipped)
        self.gnorm_RNN_it_clipped += tf.global_norm(rnn_v_clipped)
        self.gnorm_wordModule_it_clipped += tf.global_norm(word_module_v_clipped)

        self.n_miniBatch+=1

    def print_summary(self,it):
        name = "" if self.train else "val"
        tfs.scalar("loss/loss_struc"+name, self.loss_struct /self.n_miniBatch)
        tfs.scalar("loss/loss_value"+name, self.loss_value /self.n_miniBatch)
        tfs.scalar("loss/loss_value_POS"+name, self.loss_POS /self.n_miniBatch)
        tfs.scalar("loss/loss_value_word"+name, self.loss_word /self.n_miniBatch)
        tfs.scalar("norms/hidden representation norm"+name, self.h_norm_it/self.n_miniBatch)
        tfs.scalar("norms/square of weights norm"+name, self.w_norm_it/self.n_miniBatch)

        tfs.scalar("norms/grad"+name, self.gnorm_tot_it /self.n_miniBatch)
        tfs.scalar("norms/dec_grad"+name, self.gnorm_dec_it/self.n_miniBatch)
        tfs.scalar("norms/RNN_grad"+name, self.gnorm_RNN_it/self.n_miniBatch)
        tfs.scalar("norms/word_module_grad"+name, self.gnorm_wordModule_it/self.n_miniBatch)
        tfs.scalar("norms/enc_grad"+name, self.gnorm_enc_it/self.n_miniBatch)

        tfs.scalar("norms/grad_clipped"+name, self.gnorm_tot_it_clipped /self.n_miniBatch)
        tfs.scalar("norms/dec_grad_clipped"+name, self.gnorm_dec_it_clipped/self.n_miniBatch)
        tfs.scalar("norms/RNN_grad_clipped"+name, self.gnorm_RNN_it_clipped/self.n_miniBatch)
        tfs.scalar("norms/word_module_grad_clipped"+name, self.gnorm_wordModule_it_clipped/self.n_miniBatch)
        tfs.scalar("norms/enc_grad_clipped"+name, self.gnorm_enc_it_clipped/self.n_miniBatch)

        print("iterartion",it,self.loss_struct /self.n_miniBatch,self.loss_word /self.n_miniBatch,
              self.loss_POS /self.n_miniBatch)