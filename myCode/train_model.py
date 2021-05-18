import tensorflow.contrib.eager as tfe
import tensorflow.contrib.summary as tfs
from myCode.helper_functions import extract_words_from_tree, get_input_target_minibatch
from tensorflow_trees.definition import Tree
import tensorflow as tf
from random import shuffle
from helper_functions import Summary
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider


def train_model(FLAGS, decoder, encoder, train_data,val_data,
                optimizer, beta,clipping,batch_size, flat_val_captions, tensorboard_name,
                tree_encoder, tree_decoder):

    #tensorboard
    summary_writer = tfs.create_file_writer(FLAGS.model_dir+tensorboard_name, flush_millis=1000)
    summary_writer.set_as_default()

    with tfs.always_record_summaries():
        for i in range(FLAGS.max_iter):

            #shuffle dataset at beginning of each iteration
            shuffle(train_data)
            len_input = len(train_data) if type(train_data)==list else train_data.shape[0]

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
                        pass

                    variables = encoder.variables + decoder.variables

                    #compute h and w norm for regularization
                    w_norm= tf.add_n([tf.nn.l2_loss(v) for v in variables])

                    # compute gradient
                    grad = tape.gradient(loss_miniBatch+ beta*w_norm , variables)
                    gnorm = tf.global_norm(grad)
                    grad_clipped,_= tf.clip_by_global_norm(grad,clipping,gnorm)
                    # apply optimizer on gradient
                    optimizer.apply_gradients(zip(grad_clipped, variables), global_step=tf.train.get_or_create_global_step())
                    summary.add_miniBatch_summary(loss_struct_miniBatch,loss_value__miniBatch,loss_values_miniBatch["POS_tag"],
                        loss_values_miniBatch["word"],w_norm,gnorm,grad,grad_clipped,variables)


            loss_word, loss_POS = summary.print_summary(i)
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

                summary.print_supervised_validation_summary(loss_struct_val, loss_validation,loss_values_validation["POS_tag"],
                        loss_values_validation["word"],loss_word, loss_POS,i)

                #get unsupervised validation loss
                # first sampling
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
                res = dict()
                for scorer in scores:
                    score, scores = scorer.compute_score(refs,preds)
                    res[scorer.method()] =  score







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

                res_b = dict()
                for scorer in scores:
                    score, scores = scorer.compute_score(refs,preds)
                    res_b[scorer.method()] =  score




                s_avg, v_avg, tot_pos_uns, matched_pos_uns, total_word_uns ,matched_word_uns= \
                    Tree.compare_trees(target_val, batch_unsuperv.decoded_trees)

                summary.print_unsupervised_validation_summary(res,res_b, s_avg, v_avg,tot_pos_uns,matched_pos_uns,total_word_uns,
                                                      matched_word_uns,i)