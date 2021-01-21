from myCode.helper_functions import compute_max_arity
from myCode.models import *
import tensorflow as tf
from myCode.train_model import train_model

def validation(train_data,val_data,parameters, FLAGS,input_tree, target_tree, name: str,val_all_captions,test=None) :

    #open file
    #f= open(name+".txt","ab", buffering=0)

    image_max_arity, sen_max_arity = compute_max_arity(train_data, val_data)

    #selected actual parameter to try
    i=0
    emb_tree_size = parameters[0][0]
    max_node_count = parameters[1][0]
    max_depth =  parameters[2][0]
    cut_arity = parameters[3][0]
    for lamb in parameters[4]:
        for b in parameters[5]:
            beta = b
            hidden_coeff = parameters[6][0]
            learning_rate = parameters[7][0]
            clipping = parameters[8][0]
            batch_size = parameters[9][0]
            batch_size = pow(2,batch_size)
            emb_word_size = emb_tree_size
            hid = parameters[11][0]
            for keep_rate in parameters[12]:
                #hidden_word = int(WordValue.representation_shape*hid)
                hidden_word= int(emb_word_size*hid)
                print(hidden_word)

                activation = getattr(tf.nn, FLAGS.activation)

                decoder, encoder = get_encoder_decoder(emb_tree_size=emb_tree_size,cut_arity=cut_arity,max_arity=max(image_max_arity,
                    sen_max_arity),max_node_count=max_node_count,max_depth=max_depth,hidden_coeff=hidden_coeff,
                    activation=activation,image_tree=input_tree,sentence_tree=target_tree,emb_word_size=emb_word_size,
                    hidden_word=hidden_word,keep_rate=keep_rate)

                #train
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

                matched_word_uns,matched_pos_uns, s_avg, bleu, best_n_it= train_model(FLAGS=FLAGS,decoder=decoder,
                    encoder=encoder,train_data=train_data, val_data=val_data ,optimizer=optimizer,
                    beta=beta,lamb=lamb,clipping=clipping,batch_size=batch_size,n_exp=i,name=name,
                    tree_encoder =not(input_tree==None), tree_decoder = not(target_tree==None),final=False,
                    val_all_captions=val_all_captions)

                string = "\n" +str(i) +")models with parameters emb_tree_size " + str (emb_tree_size) + " max node count " + str(max_node_count) + \
                         " max_depth " + str(max_depth) + " cut arity " + str(cut_arity) + \
                         " emb_word_size " + str(emb_word_size) + " hidden_word_dim " + str(hidden_word) +\
                         " lamdda " + str(lamb) + " beta " + str(beta) + \
                         " hidden coeff " + str(hidden_coeff) +" learn rate " + str(learning_rate) + " clipping "+ str(clipping) + \
                         " batch size " + str(batch_size) + " ,matched word unsupervised " + str(matched_word_uns)  +\
                         " ,matched POS unsupervised " + str(matched_pos_uns) +  " and struct accuracy " + str(s_avg) + \
                         " bleu-1 "+str(bleu)+" in "+ str(best_n_it) + " itertions\n"

                #f.write(str.encode(string))
                print ("experiment " + str(i) + " out of 27 finished\n")
                i+=1
