import tensorflow as tf
import numpy as np
from myCode.tree_defintions import WordValue
import random
import sys
import myCode.shared_POS_words_lists as shared_list

def get_ordered_nodes(embeddings, ops,TR, trees):
    """
    function to sort nodes and corresponding target words (before processing them)
    :param embeddings: all node embeddings
    :param ops:list of item to select among the wall node embeddings
    :return: inp (parents of node to generate)
            target(targets node list)
            batch_idx (map nodes to corresponding trees)
            perm2unsort (permutation to apply to have nodes in orginal order)
    """
    #take node numbs
    node_numb = [o.meta['node_numb'] for o in ops]

    if TR:
        #compute permutation to sort it and to "unsort" it
        perm2sort = np.argsort(node_numb)
        perm2unsort = inv(perm2sort)
        #take targets and check its length, sort them if we are in TR
        targets = [o.meta['target'].value.representation for o in ops]
        assert len(node_numb) == len(targets)
        targets = [targets[i] for i in perm2sort]
        #sort nodes, targets and get batch_idxs and input
        node_numb = [node_numb[i] for i in perm2sort]
        ops = [ops[i] for i in perm2sort]
        batch_idxs = [o.meta['batch_idx'] for o in ops]
        inp =tf.gather(embeddings, node_numb)
    else:
        #build list to return
        targets=[]
        batch_idxs = []
        leafs_ordered = []
        for tree in trees:
            leafs_ordered.append([])
        #get leafs ordered
        for tree in trees:
            get_ordered_leafs(tree,leafs_ordered)
        #build batch idx, inp and perm2unsort
        for i in range(0, len(leafs_ordered)):
            for el in leafs_ordered[i]:
                batch_idxs.append(i)
        nodes_to_take = [node_numb for l in leafs_ordered for node_numb in l]
        inp = tf.gather(embeddings,nodes_to_take)
        perm2unsort = np.argsort(nodes_to_take)

    #assert
    assert len(node_numb) == inp.shape[0]
    assert len(node_numb) == len(batch_idxs)

    return inp,targets,batch_idxs,perm2unsort

#########################
#functions used in words prediction
#######################
def words_predictions(word_module, batch_idxs, inp, targets, TR,roots_emb,
                      root_only_in_fist_LSTM_time,perm2unsort, keep_rate, n_it):
    """
    function taking care of the wall word prediction (it calls several other functions)
    :param embedding:
    :param rnn:
    :param final_layer:
    :param batch_idxs:
    :param inp:
    :param targets:
    :param TR:
    :param roots_emb:
    :param root_only_in_fist_LSTM_time:
    :param perm2unsort:
    :param n_it: current iteration number
    :return:
    """
    #take sentences length
    sentences_len = get_sentences_length(batch_idxs,TR)
    #prepare data (reshape as expected)
    inputs, targets_padded_sentences = zip_data(inp, sentences_len, targets,TR, root_only_in_fist_LSTM_time)
    if TR:
        #if training or teacher forcing
        predictions =word_module.call(inputs, roots_emb,targets_padded_sentences)
    else:
        #otherwise sampling
        predictions =  word_module.sampling(roots_emb,inputs)
    #unzip data (reshape as 2D matrix)
    vals = unzip_data(predictions,sentences_len,perm2unsort)
    return vals

#TODO handle root in each time stamp
def zip_data(inp, sentences_len, targets,TR, root_only_in_fist_LSTM_time):
    """
    function to get data in format expected by RNN i.e. (n_senteces)*(sen_max_lenght)*(representation_size)
    :param inp:  input as 2D matrix
    :param sentences_len:  list of sentences length
    :param targets: list of targets nodes
    :param TR:  whether we are in Traning or not
    :param roots_emb:  roots embedding
    :param root_only_in_fist_LSTM_time: whether to use tree toot only in the first timestamp or not
    :return:
    """
    max_len = np.amax(sentences_len)
    padded_input = []
    targets_padded_sentences = []
    current_node = 0
    current_tree=0
    # reshape as (n_senteces)*(sen_max_lenght)*(representation_size)
    for el in sentences_len:
        # first input part

        # take nodes belonging to current sentence, pad them to max sentences length and concatenate them all together
        current_sen =inp[current_node:current_node+el]
        if not root_only_in_fist_LSTM_time:
            sys.exit("to implement")

        padding = tf.constant([[0, (max_len - el)], [0, 0]])
        current_sen = tf.pad(current_sen, padding, 'CONSTANT')
        padded_input.append(current_sen)
        if TR:
            # targets only if available i.e. if we are in TR
            current_targets = []
            for item in targets[current_node:(current_node + el)]:
                # take as target all words in current sentence except the last one (it will be never use as target)
                current_targets.append(item)
            # as before pad the sentence to max length, then concatenate with other ones
            current_targets = tf.convert_to_tensor(current_targets)
            padding = tf.constant( [ [0, (max_len - el)] ] )
            current_targets = tf.pad(current_targets, padding, 'CONSTANT')
            targets_padded_sentences.append(current_targets)
        #in any case update current node pointer
        current_node+=el
        current_tree+=1
    padded_input = tf.convert_to_tensor(padded_input)
    targets_padded_sentences = tf.convert_to_tensor(targets_padded_sentences)
    assert current_node == np.sum(sentences_len)
    return padded_input, targets_padded_sentences


def unzip_data(predictions,sentences_len,perm2unsort):
    """
    function to unzip rnn result i.e. go back in representation as 2D matrix and go back to previous order of nodes
    :param predictions:
    :param sentences_len:
    :param perm2unsort:
    :return:
    """
    vals = []
    for i in range (len(sentences_len)):
        current_sen_padded =predictions[i]
        current_sen = current_sen_padded[0:sentences_len[i]]
        vals.append(current_sen)
    vals = tf.concat([item for item in vals],axis=0)
    vals = tf.gather(vals, perm2unsort)
    assert np.sum(sentences_len) == vals.shape[0]
    return vals



######################
#NIC code
######################
class NIC_Decoder(tf.keras.Model):
    def __init__(self,embedding_dim, units, vocab_size):
        super(NIC_Decoder,self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim, name="embedding")
        self.rnn =tf.keras.layers.LSTM(units=units, return_state=True, return_sequences=True, name="LSTM")
        self.final_layer = tf.keras.layers.Dense(vocab_size, activation="linear",
                                     name="final_word_pred_layer")
        self.units = units


    def call(self,pos_embs, images_emb, targets):

        #TODO non passare one-hot vec ma direttamente indice utilizzare dtype=tf.uint16/8 e shape=(1)
        #TODO capire anche discrso di rnn_unts (usare debugger per capire cosa istanzia la LSTM)
        # from targets can discard last ones (no other words to predict after the last ones)
        word_embs = self.embedding_layer(targets[:,:-1])

        # concatenate image embedding as first time stamp
        images_emb = tf.expand_dims(images_emb,axis=1)
        word_embs = tf.concat([images_emb,word_embs],axis=1)

        # concatenate also pos tag as inputs in addition to the previous ones
        rnn_input = tf.concat([word_embs,pos_embs],axis=-1)

        # call LSTM
        states = [tf.zeros(shape=(pos_embs.shape[0],self.units))]*2
        rnn_output,state_h,state_c= self.rnn(rnn_input)#, initial_state = states)

        #get predictions from last layer
        predictions = self.final_layer(rnn_output)

        return predictions

    def sampling(self,features,pos_embs):

        states = [tf.zeros(shape=(pos_embs.shape[0], self.units))] * 2
        max_length = pos_embs.shape[1]
        to_return=[]

        #sampling of all word in parallel
        for i in range(max_length):
            if i==0:
                current_word_embs = tf.expand_dims(features, axis=1)
            else:
                current_word_embs= self.embedding_layer(tf.argmax(predictions, axis=-1))
            current_pos_embs = tf.expand_dims (pos_embs[:, i, :], axis=1)
            rnn_inputs = tf.concat([current_word_embs, current_pos_embs], axis=-1)
            rnn_output,states_h,state_c = self.rnn(rnn_inputs, initial_state = states)
            states=[states_h,state_c]

            predictions=self.final_layer(rnn_output)
            to_return.append(predictions)

        to_return = tf.concat([item for item in to_return],axis=1)
        return to_return

#######################
#helper functions
######################

def inv(perm):
    """
    function returning inverse of given permutation
    :param perm:
    :return:
    """
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def get_sentences_length(batch_idxs,TR):
    """
    function to get sentences length starting from batch_idxs
    :param batch_idxs:
    :return:
    """
    if TR:
        current_tree = 0
        sentences_len = []
        current_len = 0
        for el in batch_idxs:
            if el == current_tree:
                current_len += 1
            else:
                current_tree += 1
                sentences_len.append(current_len)
                current_len = 1
        sentences_len.append(current_len)
    else:
        sentences_len = []
        n_sentences = np.max(batch_idxs)+1
        for i in range(n_sentences):
            sentences_len.append(batch_idxs.count(i))
    assert np.sum(sentences_len) == len(batch_idxs)
    return sentences_len

def get_ordered_leafs(tree, l : list):
    """
    function to get ordered leafs from left to right
    :param tree: tree to visit
    :param l: list in which append node number
    :return:
    """
    if tree.node_type_id=="word":
        l[tree.meta['batch_idx']].append(tree.meta['node_numb'])
    for c in tree.children:
        get_ordered_leafs(c,l)
