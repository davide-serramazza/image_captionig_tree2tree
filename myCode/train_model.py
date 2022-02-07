import tensorflow.contrib.eager as tfe
import tensorflow.contrib.summary as tfs
from myCode.helper_functions import get_input_target_minibatch,extract_words_from_tree,extract_words,dump_in_tensorboard
from tensorflow_trees.definition import Tree
import tensorflow as tf
from random import shuffle
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def train_model(FLAGS, decoder, encoder, train_data,val_data,
                optimizer, beta,clipping,batch_size, sen_max_len,flat_val_captions, tensorboard_name,
                tree_encoder, tree_decoder):

    #tensorboard
    summary_writer = tfs.create_file_writer(FLAGS.model_dir+tensorboard_name, flush_millis=1000)
    summary_writer.set_as_default()

    with tfs.always_record_summaries():
        for i in range(FLAGS.max_iter):

            #shuffle dataset at beginning of each iteration
            shuffle(train_data)
            len_input = len(train_data) if type(train_data)==list else train_data.shape[0]

            losses_current_it={"struct" : 0, "pos" : 0, "word" : 0, "count":0}
            for j in range(0,len_input,batch_size):
                with tfe.GradientTape() as tape:
                    current_batch_input, current_batch_target = get_input_target_minibatch(train_data,j,batch_size,tree_encoder,tree_decoder)

                    # encode and decode data
                    batch_enc = encoder(current_batch_input,training=True)
                    root_emb = batch_enc.get_root_embeddings() if tree_encoder else batch_enc
                    loss_miniBatch, loss_struct_miniBatch, loss_value__miniBatch, loss_values_miniBatch = train_tree_decoder(decoder,
                        root_emb,current_batch_target) if tree_decoder else train_flat_decoder(decoder, root_emb, current_batch_target)
                    losses_current_it["struct"]+=loss_struct_miniBatch
                    losses_current_it["pos"]+=loss_values_miniBatch["POS_tag"]
                    losses_current_it["word"]+=loss_values_miniBatch["word"]
                    losses_current_it["count"]+=1
                    variables = encoder.variables + decoder.variables

                    # compute w norm for regularization
                    w_norm= tf.add_n([tf.nn.l2_loss(v) for v in variables])

                    # compute gradient
                    grad = tape.gradient(loss_miniBatch+ beta*w_norm , variables)
                    gnorm = tf.global_norm(grad)
                    grad_clipped,_= tf.clip_by_global_norm(grad,clipping,gnorm)

                    # apply gradient
                    optimizer.apply_gradients(zip(grad_clipped, variables), global_step=tf.train.get_or_create_global_step())

            losses_current_it["struct"]/=losses_current_it["count"]
            losses_current_it["pos"]/=losses_current_it["count"]
            losses_current_it["word"]/=losses_current_it["count"]
            dump_in_tensorboard(losses_current_it)
            print("iteration ",i, " loss struct is ",losses_current_it["struct"] , \
            " loss pos is ",losses_current_it["pos"] , " loss word is ",losses_current_it["word"])

            if i % FLAGS.check_every == 0:
                input_val,target_val =  get_input_target_minibatch(val_data,0,len(val_data),tree_encoder,tree_decoder)
                batch_val_enc = encoder(input_val,training=False)
                if tree_encoder:
                    batch_val_enc = batch_val_enc.get_root_embeddings()

                if tree_decoder:
                    batch_val_dec = decoder(encodings=batch_val_enc,targets=target_val,training=False,samp=True)
                    loss_struct_val, loss_values_validation = batch_val_dec.reconstruction_loss()

                # get unsupervised validation loss; first sampling
                batch_unsuperv = decoder(encodings=batch_val_enc,training=False,samp=True) if tree_decoder else   \
                    decoder.sampling(features=batch_val_enc,max_length=sen_max_len)
                pred_sentences = extract_words_from_tree(batch_unsuperv.decoded_trees,beam=False,val_data=val_data,it_n=i,name=tensorboard_name)\
                    if tree_decoder else extract_words(batch_unsuperv,beam=False , val_data=val_data,it_n=i,name=tensorboard_name)

                res = compute_scores(flat_val_captions, pred_sentences)
                print(res)

                # then beam
                batch_unsuperv_b = decoder(encodings=batch_val_enc,training=False,samp=False)  if tree_decoder else \
                    decoder.beam_search(features=batch_val_enc,pos_embs=None, sentences_len=sen_max_len,flat_decoder=True)
                pred_sentences_b = extract_words_from_tree(batch_unsuperv_b.decoded_trees,beam=True,val_data=val_data,it_n=i,name=tensorboard_name)\
                    if tree_decoder else extract_words(batch_unsuperv_b,beam=True , val_data=val_data,it_n=i,name=tensorboard_name)

                res_b = compute_scores(flat_val_captions, pred_sentences_b)
                print(res_b)

                if tree_decoder:
                    tree_comparison_ris = Tree.compare_trees(target_val, batch_unsuperv.decoded_trees)
                dump_in_tensorboard(loss_validation={"struct" : loss_struct_val, "pos" :  loss_values_validation["POS_tag"] , \
                    "word" :  loss_values_validation["word"]},\
                res_sampling=res,res_beam=res_b,tree_comparison_ris=tree_comparison_ris,losses_current_it=None)


def compute_scores(flat_val_captions, pred_sentences):
    refs = dict()
    preds = dict()
    j = 0
    for pred, ref in zip(pred_sentences, flat_val_captions):
        preds[j] = [pred]
        refs[j] = ref
        j += 1
    scores = [Bleu(4), Rouge(), Cider()]
    res = dict()
    for scorer in scores:
        score, scores = scorer.compute_score(refs, preds)
        res[scorer.method()] = score
    #TODO calcolarsi meteor ex-post
    return res


def train_flat_decoder(decoder, root_emb,current_batch_target):
    batch_dec = decoder.call([], root_emb, current_batch_target)

    # compute loss among the non padded vector
    current_batch_target = tf.unstack(current_batch_target)
    batch_dec = tf.unstack(batch_dec[:,1:,:])
    targets = []
    preds = []
    for target, pred in zip(current_batch_target, batch_dec):
        padding = tf.where(tf.equal(target, 0))
        if padding.shape[0] == 0:
            padding = tf.constant([[target.shape[0]]])
        targets.append(target[1:padding[0][0]])
        preds.append(pred[:padding[0][0]-1])
    targets = tf.concat([t for t in targets], axis=0)
    preds = tf.concat([t for t in preds], axis=0)
    all_words_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=preds)
    loss_miniBatch = tf.reduce_mean(all_words_loss, axis=-1)

    return loss_miniBatch, None, None, None


def train_tree_decoder(decoder, root_emb,current_batch_target):
    batch_dec = decoder(encodings=root_emb, targets=current_batch_target, training=True, samp=True)
    # compute loss
    loss_struct_miniBatch, loss_values_miniBatch = batch_dec.reconstruction_loss()
    loss_value__miniBatch = loss_values_miniBatch["POS_tag"] + loss_values_miniBatch["word"]
    loss_miniBatch = loss_value__miniBatch + loss_struct_miniBatch
    return loss_miniBatch, loss_struct_miniBatch, loss_value__miniBatch, loss_values_miniBatch