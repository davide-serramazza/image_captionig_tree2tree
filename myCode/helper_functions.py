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
    print('loading image trees....')
    image_features_extract_model = istanciate_CNN(tree_encoder)
    train_data = read_images(args.train,image_features_extract_model,tree_cnn_type,batch_size)
    val_data = read_images(args.val,image_features_extract_model,tree_cnn_type,batch_size)
    if args.test!='None':
        test_data = read_images(args.test,image_features_extract_model,tree_cnn_type,batch_size)
        train_data = train_data+val_data
        val_data = test_data
    print('loading sentence trees...')
    with open(args.all_captions) as json_file:
        flat_captions = json.load(json_file)
    if tree_decoder:
        sen_max_len = label_tree_with_sentenceTree(train_data,val_data, args.targets)
    else:
        sen_max_len = get_flat_captions(train_data,val_data, flat_captions)

    flat_val_caption = load_flat_captions(flat_captions,val_data)

    return train_data,val_data,flat_val_caption, sen_max_len


def extract_words(predictions, beam):
    predicted_words=tf.unstack( tf.argmax(predictions,axis=-1) ) if beam else tf.unstack( tf.argmax(predictions[:,1:,:],axis=-1) )
    sentences = []
    for sen in predicted_words:
        s=""
        for word in sen:
            word = shared_list.tokenizer.index_word[ word.numpy()]
            if word=='<end>':
                break
            else:
                s+=( word +" ")
        sentences.append(s)
    return sentences

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
        default=100,
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
        s=""
        for el in current_pred:
            s+=(el+" ")
        to_return.append(s)
    return to_return


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
                              w_norm,gnorm,grad,grad_clipped,variables):

        if loss_struct_miniBatch is not None:
            self.loss_struct += loss_struct_miniBatch
            self.loss_value += loss_value_miniBatch
            self.loss_POS += loss_value_miniBatch_pos
        self.loss_word +=  loss_value_miniBatch_word
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

        return self.loss_word /self.n_miniBatch, self.loss_POS /self.n_miniBatch

    def print_supervised_validation_summary(self,loss_struct_val, loss_validation,loss_values_validation_pos,
            loss_values_validation_word,loss_word, loss_POS,it):

        tfs.scalar("loss/validation/loss_struc", loss_struct_val)
        tfs.scalar("loss/validation/loss_value", loss_validation)
        tfs.scalar("loss/validation/loss_value_POS", loss_values_validation_pos)
        tfs.scalar("loss/validation/loss_value_word", loss_values_validation_word)

        print("iteration ", it, " supervised:\nloss train word is ", loss_word, " loss train POS is ", loss_POS , "\n",
              " loss validation word is ", loss_values_validation_word, " loss validation POS is ", loss_values_validation_pos)


    def print_unsupervised_validation_summary(self,res,res_b,it,tree_decoder, s_avg=None, v_avg=None ,tot_pos_uns=None ,
        matched_pos_uns=None,total_word_uns=None,matched_word_uns=None):

            tfs.scalar("metrics/bleu/blue-1", res['Bleu'][0])
            tfs.scalar("metrics/bleu/blue-2",  res['Bleu'][1])
            tfs.scalar("metrics/bleu/blue-3",  res['Bleu'][2])
            tfs.scalar("metrics/bleu/blue-4",  res['Bleu'][3])
            tfs.scalar("metrics/CIDEr", res['CIDEr'])
            tfs.scalar("metrics/Rouge", res['Rouge'])
            tfs.scalar("metrics/METEOR",res['METEOR'])
            print("sampling" , res)

            tfs.scalar("metrics/bleu/blue-1_b", res_b['Bleu'][0])
            tfs.scalar("metrics/bleu/blue-2_b",  res_b['Bleu'][1])
            tfs.scalar("metrics/bleu/blue-3_b",  res_b['Bleu'][2])
            tfs.scalar("metrics/bleu/blue-4_b",  res_b['Bleu'][3])
            tfs.scalar("metrics/CIDEr_b", res_b['CIDEr'])
            tfs.scalar("metrics/Rouge_b", res_b['Rouge'])
            tfs.scalar("metrics/METEOR_b",res_b['METEOR'])
            print("beam    " , res_b, "\n")

            if tree_decoder:
                tfs.scalar("overlaps/unsupervised/struct_avg", s_avg)
                tfs.scalar("overlaps/unsupervised/value_avg", v_avg)
                tfs.scalar("overlaps/unsupervised/total_POS", tot_pos_uns)
                tfs.scalar("overlaps/unsupervised/matched_POS", matched_pos_uns)
                tfs.scalar("overlaps/unsupervised/total_words", total_word_uns)
                tfs.scalar("overlaps/unsupervised/matched_words", matched_word_uns)

                print("iteration ", it, " unsupervised:\n", matched_pos_uns," out of ", tot_pos_uns, " POS match",
                      "that is a perc of", (matched_pos_uns/tot_pos_uns)*100, " " ,matched_word_uns, " out of ",total_word_uns,
                      "word match that is a percentage of ", (matched_word_uns/total_word_uns)*100, " struct val ", s_avg)