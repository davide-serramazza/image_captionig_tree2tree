import tensorflow as tf
import numpy as np

def get_ordered_nodes(embeddings, ops,TR, trees):
    """
    #TODO man
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
        # compute permutation to sort it and to "unsort" it
        perm2sort = np.argsort(node_numb)
        perm2unsort = inv(perm2sort)
        # take targets and check its length, sort them if we are in TR
        targets = [o.meta['target'].value.representation for o in ops]
        assert len(node_numb) == len(targets)
        targets = [targets[i] for i in perm2sort]
        # sort nodes, targets and get batch_idxs and input
        node_numb = [node_numb[i] for i in perm2sort]
        ops = [ops[i] for i in perm2sort]
        batch_idxs = [o.meta['batch_idx'] for o in ops]
        inp =tf.gather(embeddings, node_numb)
    else:
        # build list to return
        targets=[]
        batch_idxs = []
        leaves_ordered = []
        for tree in trees:
            leaves_ordered.append([])
        # get leaves ordered
        for tree in trees:
            get_ordered_leafs(tree,leaves_ordered)
        # build batch idx, inp and perm2unsort
        for i in range(0, len(leaves_ordered)):
            for el in leaves_ordered[i]:
                batch_idxs.append(i)
        nodes_to_take = [node_numb for l in leaves_ordered for node_numb in l]
        inp = tf.gather(embeddings,nodes_to_take)
        perm2unsort = np.argsort(nodes_to_take)

    #assert
    assert len(node_numb) == inp.shape[0]
    assert len(node_numb) == len(batch_idxs)

    return inp,targets,batch_idxs,perm2unsort

#########################
#functions used in words prediction
#######################
def words_predictions(word_module, batch_idxs, inp, targets, TR,roots_emb,perm2unsort,samp):
    """
    #TODO man
    function taking care of the wall word prediction (it calls several other functions)
    :param embedding:
    :param rnn:
    :param final_layer:
    :param batch_idxs:
    :param inp:
    :param targets:
    :param TR:
    :param roots_emb:
    :param perm2unsort:
    :param n_it: current iteration number
    :return:
    """
    # take sentences length
    sentences_len = get_sentences_length(batch_idxs,TR)
    # prepare data (reshape as expected)
    inputs, targets_padded_sentences = zip_data(inp, sentences_len, targets,TR)
    if TR:
        # if training, teacher forcing
        predictions =word_module.call(inputs, roots_emb,targets_padded_sentences)
    else:
        # otherwise inference
        if samp:
            predictions = word_module.sampling(roots_emb,inputs)
        else:
            predictions =  word_module.beam_search(roots_emb,inputs,sentences_len)
    # unzip data (reshape as 2D matrix)
    vals = unzip_data(predictions,sentences_len,perm2unsort,samp)
    return vals

#TODO handle root in each time stamp
def zip_data(inp, sentences_len, targets,TR):
    """
    #TODO man
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


def unzip_data(predictions,sentences_len,perm2unsort,samp):
    """
    #TODO man
    function to unzip rnn result i.e. go back in representation as 2D matrix and go back to previous order of nodes
    :param predictions:
    :param sentences_len:
    :param perm2unsort:
    :return:
    """
    def unzip_sampling(predictions,sentences_len,perm2unsort):
        vals = []
        for i in range (len(sentences_len)):
            current_sen_padded =predictions[i]
            current_sen = current_sen_padded[0:sentences_len[i]]
            vals.append(current_sen)
        vals = tf.concat([item for item in vals],axis=0)
        vals = tf.gather(vals, perm2unsort)
        assert np.sum(sentences_len) == vals.shape[0]
        return vals

    def unzip_beam(predictions,perm2unsort):
        vals = tf.concat([t for t in predictions],axis=0)
        assert  vals.shape[0]==perm2unsort.shape[0]
        vals = tf.gather(vals, perm2unsort)
        return vals

    vals =  unzip_sampling(predictions,sentences_len,perm2unsort) if samp else unzip_beam(predictions,perm2unsort)
    return vals

#######################
#helper functions
######################

def inv(perm):
    #TODO man
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
    #TODO man
    """
    function to get sentences length starting from batch_idxs
    :param batch_idxs:
    :return:
    """
    sentences_len = []
    if TR:
        current_tree = 0
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
        n_sentences = np.max(batch_idxs)+1
        for i in range(n_sentences):
            sentences_len.append(batch_idxs.count(i))
    assert np.sum(sentences_len) == len(batch_idxs)
    return np.asarray(sentences_len)

def get_ordered_leafs(tree, l : list):
    #TODO man
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
