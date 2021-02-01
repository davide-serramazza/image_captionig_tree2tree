import tensorflow.contrib.eager as tfe
import tensorflow.contrib.summary as tfs
from myCode.helper_functions import max_arity,shuffle_data,extract_words_from_tree, compute_max_arity
from myCode.models import *
from  myCode.helper_functions import select_one_in_range
from tensorflow_trees.definition import Tree
from myCode.load_model import restore_model,predict_test_dataSet
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
import myCode.shared_POS_words_lists as shared_list
from tensorflow.python.keras import backend as K

def train_model(FLAGS, decoder, encoder, input_train, target_train,input_val, target_val,
                optimizer, beta,lamb,clipping,batch_size,n_exp, name,val_all_captions,
                tree_encoder, tree_decoder, final=False, keep_rate=1.0, test=None):

    best_n_it = 0
    best_loss = 100
    best_matched_word=0
    best_matched_pos=0
    best_struct=0
    best_bleu=0
    if final:
        FLAGS.max_iter = 2000
        FLAGS.check_every = 10

    #tensorboard
    summary_writer = tfs.create_file_writer(FLAGS.model_dir+name+"/" +str(n_exp), flush_millis=1000)
    summary_writer.set_as_default()

    with tfs.always_record_summaries():
        for i in range(FLAGS.max_iter):
            loss_struct=0
            loss_value=0
            loss_POS = 0
            loss_word = 0

            #shuffle dataset at beginning of each iteration
            len_input = len(input_train) if type(input_train)==list else input_train.shape[0]
            #input_train,target_train = shuffle_data(input_train,target_train,len_input)
            for j in range(0,len_input,batch_size):
                with tfe.GradientTape() as tape:

                    current_batch_input=input_train[j:j+batch_size]
                    current_batch_target = target_train[j:j+batch_size]

                    # encode and decode datas
                    batch_enc = encoder(current_batch_input)
                    root_emb = batch_enc.get_root_embeddings() if tree_encoder else batch_enc
                    if tree_decoder:
                        batch_dec = decoder(encodings=root_emb, targets=current_batch_target,keep_rate=keep_rate,n_it=i)
                        # compute global loss
                        loss_struct_miniBatch, loss_values_miniBatch = batch_dec.reconstruction_loss()
                        loss_value__miniBatch = loss_values_miniBatch["POS_tag"] + loss_values_miniBatch["word"]
                        loss_miniBatch = loss_value__miniBatch+loss_struct_miniBatch

                        #compute minibatch loss
                        loss_struct += loss_struct_miniBatch
                        loss_value += loss_value__miniBatch
                        loss_POS += loss_values_miniBatch["POS_tag"]
                        loss_word +=  loss_values_miniBatch["word"]
                    else:
                        loss_single_word=0
                        hidden = decoder.reset_state(batch_size=current_batch_target.shape[0])
                        dec_input = tf.expand_dims([shared_list.word_idx['<start>']] * current_batch_target.shape[0], 1)
                        for h in range(1, current_batch_target.shape[1]):
                            predictions, hidden = decoder(dec_input, root_emb, hidden,keep_rate=keep_rate)
                            loss_single_word +=loss_function (current_batch_target[:, h],predictions)
                            dec_input = tf.expand_dims(tf.argmax(current_batch_target[:, h],axis=-1), 1)
                        loss_miniBatch = (loss_single_word/int(current_batch_target.shape[1]))
                        loss_word += loss_miniBatch

                    variables = encoder.variables + decoder.variables

                    #compute h and w norm for regularization
                    h_norm= tf.norm(root_emb)
                    w_norm=0
                    for w in variables:
                        norm = tf.norm(w)
                        if norm >= 0.001:
                            w_norm += norm

                    # compute gradient
                    grad = tape.gradient(loss_miniBatch+ beta*w_norm +lamb*h_norm, variables)
                    gnorm = tf.global_norm(grad)
                    grad, _ = tf.clip_by_global_norm(grad, clipping, gnorm)
                    tfs.scalar("norms/grad", gnorm)
                    tfs.scalar("norms/h_norm", h_norm)
                    tfs.scalar("norms/w_norm", w_norm)

                    # apply optimizer on gradient
                    optimizer.apply_gradients(zip(grad, variables), global_step=tf.train.get_or_create_global_step())


            loss_struct /= (int(int(len_input)/batch_size)+1)
            loss_value /= (int(int(len_input)/batch_size)+1)
            loss_POS  /= (int(int(len_input)/batch_size)+1)
            loss_word /= (int(int(len_input)/batch_size)+1)
            loss = loss_struct+loss_value
            print(name,":iterartion",i,loss,loss_word,loss_POS)

            tfs.scalar("loss/loss_struc", loss_struct)
            tfs.scalar("loss/loss_value", loss_value)
            tfs.scalar("loss/loss_value_POS", loss_POS)
            tfs.scalar("loss/loss_value_word", loss_word)


            # print stats
            if i % FLAGS.check_every == 0:
                #var_to_save = encoder.variables+encoder.weights + decoder.variables+decoder.weights + optimizer.variables()
                #tfe.Saver(var_to_save).save(checkpoint_prefix,global_step=tf.train.get_or_create_global_step())

                if not tree_encoder:
                    input_val = tf.Variable(input_val)
                    input_val = tf.squeeze(input_val)

                batch_val_enc = encoder(input_val)
                if tree_encoder:
                    batch_val_enc = batch_val_enc.get_root_embeddings()

                if tree_decoder:
                    batch_val_dec = decoder(encodings=batch_val_enc,targets=target_val,n_it=i)
                    loss_struct_val, loss_values_validation = batch_val_dec.reconstruction_loss()
                    loss_validation = loss_struct_val + loss_values_validation["POS_tag"]+loss_values_validation["word"]
                    tfs.scalar("loss/validation/loss_struc", loss_struct_val)
                    tfs.scalar("loss/validation/loss_value", loss_validation)
                    tfs.scalar("loss/validation/loss_value_POS", loss_values_validation["POS_tag"])
                    tfs.scalar("loss/validation/loss_value_word", loss_values_validation["word"])

                    print("iteration ", i, " supervised:\nloss train word is ", loss_word, " loss train POS is ", loss_POS , "\n",
                      " loss validation word is ", loss_values_validation["word"], " loss validation POS is ", loss_values_validation["POS_tag"])

                    #get unsupervised validation loss
                    batch_unsuperv = decoder(encodings=batch_val_enc)
                    s_avg, v_avg, tot_pos_uns, matched_pos_uns, total_word_uns ,matched_word_uns= \
                        Tree.compare_trees(target_val, batch_unsuperv.decoded_trees)
                    pred_sentences = extract_words_from_tree(batch_unsuperv.decoded_trees)
                    tfs.scalar("overlaps/unsupervised/struct_avg", s_avg)
                    tfs.scalar("overlaps/unsupervised/value_avg", v_avg)
                    tfs.scalar("overlaps/unsupervised/total_POS", tot_pos_uns)
                    tfs.scalar("overlaps/unsupervised/matched_POS", matched_pos_uns)
                    tfs.scalar("overlaps/unsupervised/total_words", total_word_uns)
                    tfs.scalar("overlaps/unsupervised/matched_words", matched_word_uns)
                else:
                    pred_sentences= decoder.sampling(batch_val_enc,wi=shared_list.word_idx,
                        iw=shared_list.idx_word,max_length=int(target_val.shape[1]))

                bleu_1 = corpus_bleu(val_all_captions,pred_sentences,weights=(1.0,))
                bleu_2 = corpus_bleu(val_all_captions,pred_sentences,weights=(0.5,0.5))
                bleu_3 = corpus_bleu(val_all_captions,pred_sentences,weights=(1/3,1/3,1/3))
                bleu_4 = corpus_bleu(val_all_captions,pred_sentences,weights=(0.25,0.25,0.25,0.25))
                tfs.scalar("bleu/blue-1", bleu_1)
                tfs.scalar("bleu/blue-2", bleu_2)
                tfs.scalar("bleu/blue-3", bleu_3)
                tfs.scalar("bleu/blue-4", bleu_4)

                if tree_decoder:
                    print("iteration ", i, " unsupervised:\n", matched_pos_uns," out of ", tot_pos_uns, " POS match",
                    "that is a perc of", (matched_pos_uns/tot_pos_uns)*100, " " ,matched_word_uns, " out of ",total_word_uns,
                    "word match that is a percentage of ", (matched_word_uns/total_word_uns)*100, " struct val ", s_avg,
                      " bleu-1 ", bleu_1," bleu-2 ", bleu_2," bleu-3 ", bleu_3," bleu-4 ", bleu_4)
                else:
                    print(name,":iteration ", i," bleu-1 ", bleu_1," bleu-2 ", bleu_2," bleu-3 ", bleu_3," bleu-4 ", bleu_4)
                    loss_validation=0
                    matched_word_uns=0
                    matched_pos_uns=0
                    s_avg=0

                if best_bleu < bleu_1:
                    #update best results
                    best_bleu = bleu_1
                    best_loss = loss_validation
                    best_matched_word=matched_word_uns
                    best_matched_pos = matched_pos_uns
                    best_n_it = i
                    best_struct = s_avg
                    #predictions
                elif best_loss > loss_validation:
                    best_loss=loss_validation
                #else:
                #    break

    return best_matched_word,best_matched_pos, best_struct,best_bleu, best_n_it


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(tf.argmax(real,axis=-1), 0))
    loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=real, logits=pred)
    #loss_ = K.categorical_crossentropy(real, pred, from_logits=True)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def validation(input_train, target_train ,input_val, target_val,parameters, FLAGS,input_tree, target_tree, name: str,val_all_captions,test=None) :

    #open file
#    f= open(name+".txt","ab", buffering=0)

    #compute max_arity
    image_max_arity, input_train, sen_max_arity = compute_max_arity(input_train, input_tree, target_train, target_tree)

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
                    hidden_word=hidden_word)

                #train
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

                matched_word_uns,matched_pos_uns, s_avg, bleu, best_n_it= train_model(FLAGS=FLAGS,decoder=decoder,
                    encoder=encoder,
                    input_train=input_train,target_train=target_train,
                    input_val=input_val, target_val=target_val,optimizer=optimizer,
                    beta=beta,lamb=lamb,clipping=clipping,batch_size=batch_size,n_exp=i,name=name,
                    tree_encoder =not(input_tree==None), tree_decoder = not(target_tree==None),final=False,
                    val_all_captions=val_all_captions,keep_rate=keep_rate)

                string = "\n" +str(i) +")models with parameters emb_tree_size " + str (emb_tree_size) + " max node count " + str(max_node_count) + \
                         " max_depth " + str(max_depth) + " cut arity " + str(cut_arity) + \
                         " emb_word_size " + str(emb_word_size) + " hidden_word_dim " + str(hidden_word) +\
                         " lamdda " + str(lamb) + " beta " + str(beta) + \
                         " hidden coeff " + str(hidden_coeff) +" learn rate " + str(learning_rate) + " clipping "+ str(clipping) + \
                         " batch size " + str(batch_size) + " ,matched word unsupervised " + str(matched_word_uns)  +\
                         " ,matched POS unsupervised " + str(matched_pos_uns) +  " and struct accuracy " + str(s_avg) + \
                         " bleu-1 "+str(bleu)+" in "+ str(best_n_it) + " itertions\n"

                f.write(str.encode(string))
                print ("experiment " + str(i) + " out of 27 finished\n")
                i+=1
