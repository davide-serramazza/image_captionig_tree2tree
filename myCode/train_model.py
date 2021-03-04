import tensorflow.contrib.eager as tfe
import tensorflow.contrib.summary as tfs
from myCode.helper_functions import extract_words_from_tree, get_input_target_minibatch
from tensorflow_trees.definition import Tree
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
import myCode.shared_POS_words_lists as shared_list
from random import shuffle

def train_model(FLAGS, decoder, encoder, train_data,val_data,
                optimizer, beta,lamb,clipping,batch_size, flat_val_captions, tensorboard_name,
                tree_encoder, tree_decoder, final=False, keep_rate=1.0):

    best_loss = 100
    best_bleu=0
    if final:
        FLAGS.max_iter = 2000
        FLAGS.check_every = 10

    #tensorboard
    summary_writer = tfs.create_file_writer(FLAGS.model_dir+tensorboard_name, flush_millis=1000)
    summary_writer.set_as_default()

    with tfs.always_record_summaries():
        for i in range(FLAGS.max_iter):
            loss_struct=0
            loss_value=0
            loss_POS = 0
            loss_word = 0

            #shuffle dataset at beginning of each iteration
            shuffle(train_data)
            len_input = len(train_data) if type(train_data)==list else train_data.shape[0]
            #input_train,target_train = shuffle_data(input_train,target_train,len_input)
            for j in range(0,len_input,batch_size):
                with tfe.GradientTape() as tape:

                    current_batch_input, current_batch_target = get_input_target_minibatch(train_data,j,batch_size,tree_encoder)

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
                    h_norm= tf.norm(root_emb,ord=1)
                    w_norm= tf.add_n([tf.nn.l2_loss(v) for v in variables])

                    # compute gradient
                    grad = tape.gradient(loss_miniBatch+ beta*w_norm +lamb*h_norm, variables)
                    gnorm = tf.global_norm(grad)
                    grad, _ = tf.clip_by_global_norm(grad, clipping, gnorm)
                    tfs.scalar("norms/grad", gnorm)
                    tfs.scalar("norms/hidden representation norm", h_norm)
                    tfs.scalar("norms/square of weights norm", w_norm)
                    del current_batch_target
                    del current_batch_input

                    # apply optimizer on gradient
                    optimizer.apply_gradients(zip(grad, variables), global_step=tf.train.get_or_create_global_step())


            loss_struct /= (int(int(len_input)/batch_size)+1)
            loss_value /= (int(int(len_input)/batch_size)+1)
            loss_POS  /= (int(int(len_input)/batch_size)+1)
            loss_word /= (int(int(len_input)/batch_size)+1)
            loss = loss_struct+loss_value
            print("iterartion",i,loss,loss_word,loss_POS)

            tfs.scalar("loss/loss_struc", loss_struct)
            tfs.scalar("loss/loss_value", loss_value)
            tfs.scalar("loss/loss_value_POS", loss_POS)
            tfs.scalar("loss/loss_value_word", loss_word)


            # print stats
            if i % FLAGS.check_every == 0:
                #var_to_save = encoder.variables+encoder.weights + decoder.variables+decoder.weights + optimizer.variables()
                #tfe.Saver(var_to_save).save(checkpoint_prefix,global_step=tf.train.get_or_create_global_step())
                input_val,target_val =  get_input_target_minibatch(val_data,0,len(val_data),tree_encoder)

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

                bleu_1 = corpus_bleu(flat_val_captions,pred_sentences,weights=(1.0,))
                bleu_2 = corpus_bleu(flat_val_captions,pred_sentences,weights=(0.5,0.5))
                bleu_3 = corpus_bleu(flat_val_captions,pred_sentences,weights=(1/3,1/3,1/3))
                bleu_4 = corpus_bleu(flat_val_captions,pred_sentences,weights=(0.25,0.25,0.25,0.25))
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
                    print("iteration ", i," bleu-1 ", bleu_1," bleu-2 ", bleu_2," bleu-3 ", bleu_3," bleu-4 ", bleu_4)
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
                del target_val

# TODO spostare in RNN_Decoder ed importarla in validation
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(tf.argmax(real,axis=-1), 0))
    loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=real, logits=pred)
    #loss_ = K.categorical_crossentropy(real, pred, from_logits=True)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)