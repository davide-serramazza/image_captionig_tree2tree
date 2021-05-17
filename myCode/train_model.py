import tensorflow.contrib.eager as tfe
import tensorflow.contrib.summary as tfs
from myCode.helper_functions import extract_words_from_tree, get_input_target_minibatch
from tensorflow_trees.definition import Tree
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
import myCode.shared_POS_words_lists as shared_list
from random import shuffle
from helper_functions import Summary

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


def train_model(FLAGS, decoder, encoder, train_data,val_data,
                optimizer, beta,lamb,clipping,batch_size, flat_val_captions, tensorboard_name,
                tree_encoder, tree_decoder, final=False, keep_rate=1.0):

    #tensorboard
    summary_writer = tfs.create_file_writer(FLAGS.model_dir+tensorboard_name, flush_millis=1000)
    summary_writer.set_as_default()

    with tfs.always_record_summaries():
        for i in range(FLAGS.max_iter):
            loss_POS = 0
            loss_word = 0

            #shuffle dataset at beginning of each iteration
            shuffle(train_data)
            len_input = len(train_data) if type(train_data)==list else train_data.shape[0]
            #input_train,target_train = shuffle_data(input_train,target_train,len_input)
            for j in range(0,len_input,batch_size):
                with tfe.GradientTape() as tape:

                    summary = Summary(train=True)
                    current_batch_input, current_batch_target = get_input_target_minibatch(train_data,j,batch_size,tree_encoder)

                    # encode and decode datas
                    batch_enc = encoder(current_batch_input,training=True)
                    root_emb = batch_enc.get_root_embeddings() if tree_encoder else batch_enc
                    if tree_decoder:
                        batch_dec = decoder(encodings=root_emb, targets=current_batch_target,training=True,samp=True)
                        # compute global loss
                        loss_struct_miniBatch, loss_values_miniBatch = batch_dec.reconstruction_loss()
                        loss_value__miniBatch = loss_values_miniBatch["POS_tag"] + loss_values_miniBatch["word"]
                        loss_miniBatch = loss_value__miniBatch+loss_struct_miniBatch
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
                    grad_clipped,_= tf.clip_by_global_norm(grad,clipping,gnorm)
                    del current_batch_target
                    del current_batch_input
                    # apply optimizer on gradient
                    optimizer.apply_gradients(zip(grad_clipped, variables), global_step=tf.train.get_or_create_global_step())
                    summary.add_miniBatch_summary(loss_struct_miniBatch,loss_value__miniBatch,loss_values_miniBatch["POS_tag"],
                                                  loss_values_miniBatch["word"],h_norm,w_norm,gnorm,grad,grad_clipped,variables)

            summary.print_summary(i)
            # print stats
            if i % FLAGS.check_every == 0:
                input_val,target_val =  get_input_target_minibatch(val_data,0,len(val_data),tree_encoder)

                if not tree_encoder:
                    input_val = tf.Variable(input_val)
                    input_val = tf.squeeze(input_val)

                batch_val_enc = encoder(input_val,training=False)
                if tree_encoder:
                    batch_val_enc = batch_val_enc.get_root_embeddings()

                batch_val_dec = decoder(encodings=batch_val_enc,targets=target_val,training=False,samp=True)
                loss_struct_val, loss_values_validation = batch_val_dec.reconstruction_loss()
                loss_validation = loss_struct_val + loss_values_validation["POS_tag"]+loss_values_validation["word"]
                tfs.scalar("loss/validation/loss_struc", loss_struct_val)
                tfs.scalar("loss/validation/loss_value", loss_validation)
                tfs.scalar("loss/validation/loss_value_POS", loss_values_validation["POS_tag"])
                tfs.scalar("loss/validation/loss_value_word", loss_values_validation["word"])

                print("iteration ", i, " supervised:\nloss train word is ", loss_word, " loss train POS is ", loss_POS , "\n",
                      " loss validation word is ", loss_values_validation["word"], " loss validation POS is ", loss_values_validation["POS_tag"])

                #get unsupervised validation loss
                # sampling
                batch_unsuperv = decoder(encodings=batch_val_enc,training=False,samp=True)
                pred_sentences = extract_words_from_tree(batch_unsuperv.decoded_trees)

                refs = dict()
                preds = dict()
                j=0
                for pred,ref in zip(pred_sentences,flat_val_captions):
                    p = ""
                    r = []
                    for el in pred:
                        p += el+" "
                    preds[j]=[p]

                    for single_ref in ref:
                        tmp= ""
                        for el in single_ref[:-1]:
                            tmp+=el+" "
                        r.append(tmp)
                    refs[j]=r

                    j+=1

                scores = [Bleu(4) , Meteor() , Cider() ]
                ris = dict()
                for scorer in scores:
                    score, scores = scorer.compute_score(refs,preds)
                    ris[scorer.method()] =  score

                print("sampling" , ris)
                tfs.scalar("metrics/bleu/blue-1", ris['Bleu'][0])
                tfs.scalar("metrics/bleu/blue-2",  ris['Bleu'][1])
                tfs.scalar("metrics/bleu/blue-3",  ris['Bleu'][2])
                tfs.scalar("metrics/bleu/blue-4",  ris['Bleu'][3])
                tfs.scalar("metrics/CIDEr", ris['CIDEr'])
                tfs.scalar("metrics/METEOR", ris['METEOR'])


                # beam
                batch_unsuperv_b = decoder(encodings=batch_val_enc,training=False,samp=False)
                pred_sentences_b = extract_words_from_tree(batch_unsuperv_b.decoded_trees)

                refs = dict()
                preds = dict()
                j=0
                for pred,ref in zip(pred_sentences_b,flat_val_captions):
                    p = ""
                    r = []
                    for el in pred:
                        p += el+" "
                    preds[j]=[p]

                    for single_ref in ref:
                        tmp= ""
                        for el in single_ref[:-1]:
                            tmp+=el+" "
                        r.append(tmp)
                    refs[j]=r

                    j+=1

                scores = [Bleu(4) , Meteor() , Cider() ]

                ris_b = dict()
                for scorer in scores:
                    score, scores = scorer.compute_score(refs,preds)
                    ris_b[scorer.method()] =  score


                tfs.scalar("metrics/bleu/blue-1_b", ris_b['Bleu'][0])
                tfs.scalar("metrics/bleu/blue-2_b",  ris_b['Bleu'][1])
                tfs.scalar("metrics/bleu/blue-3_b",  ris_b['Bleu'][2])
                tfs.scalar("metrics/bleu/blue-4_b",  ris_b['Bleu'][3])
                tfs.scalar("metrics/CIDEr_b", ris_b['CIDEr'])
                tfs.scalar("metrics/METEOR_b", ris_b['METEOR'])

                print("beam    " , ris_b, "\n")


                s_avg, v_avg, tot_pos_uns, matched_pos_uns, total_word_uns ,matched_word_uns= \
                    Tree.compare_trees(target_val, batch_unsuperv.decoded_trees)
                tfs.scalar("overlaps/unsupervised/struct_avg", s_avg)
                tfs.scalar("overlaps/unsupervised/value_avg", v_avg)
                tfs.scalar("overlaps/unsupervised/total_POS", tot_pos_uns)
                tfs.scalar("overlaps/unsupervised/matched_POS", matched_pos_uns)
                tfs.scalar("overlaps/unsupervised/total_words", total_word_uns)
                tfs.scalar("overlaps/unsupervised/matched_words", matched_word_uns)

                print("iteration ", i, " unsupervised:\n", matched_pos_uns," out of ", tot_pos_uns, " POS match",
                    "that is a perc of", (matched_pos_uns/tot_pos_uns)*100, " " ,matched_word_uns, " out of ",total_word_uns,
                      "word match that is a percentage of ", (matched_word_uns/total_word_uns)*100, " struct val ", s_avg)

                del target_val