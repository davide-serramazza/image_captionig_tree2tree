from os import listdir
from os.path import join
import xml.etree.ElementTree as ET
from myCode.tree_defintions import *
import tensorflow as tf
from tensorflow_trees.definition import Tree

def extract_occ(train_data):
    word_occ = {}
    for example in train_data:
        for caption in example['sentence_trees']:
            count_word_tag_occ(caption, word_occ)
    TagValue.update_rep_shape(len(shared_list.tags_idx))
    sorted_tuples = sorted(word_occ.items(), key=lambda item: item[1],reverse=True)
    return [k for k,v in word_occ.items() if v > 5]

def count_word_tag_occ(sen_tree : ET.Element ,  words_occ : list):
    """
    function that count word occurency in dataset
    :param sen_tree:
    :param word_dict:
    :return:
    """
    value = sen_tree.attrib['value']
    #if current node is a word
    if sen_tree.tag=="leaf":
        try:
            words_occ[value.lower()]+=1
        except KeyError:
            words_occ[value.lower()]=1
    #if current word is a tag
    elif sen_tree.tag == "node":
        if (value not in shared_list.tags_idx) and (value!="ROOT"):
            shared_list.tags_idx.append(value)

    #recursion
    for child in sen_tree.getchildren():
        count_word_tag_occ(child, words_occ)


def read_tree_from_file(file):
    """
    function that read parse tree from xml file
    :param file:
    :param embeddings:
    :param dictionary:
    :param name:
    :return:
    """
    #open file
    tree = ET.parse(file)
    return tree.getroot()


def label_tree_with_real_data(xml_tree : ET.Element, final_tree : Tree,tokenizer):
    """
    function that given tree as read form xml file, "label" current tree with tree data for NN
    :param data:
    :param final_tree:
    :return:
    """
    value = xml_tree.attrib["value"]
    if xml_tree.tag == "node" and value!="ROOT":
        #check if in frequent word in dev set otherwise label as others (last dimension)
        try:
            idx = shared_list.tags_idx.index(value)
        except:
            idx = len(shared_list.tags_idx)-1
        final_tree.node_type_id="POS_tag"
        final_tree.value=TagValue(representation=tf.one_hot(idx, len(shared_list.tags_idx)))
        final_tree.children = []
        for child in xml_tree.getchildren():
            final_tree.children.append(Tree(node_type_id="tmp"))

    elif xml_tree.tag == "leaf":
        #check if in tag found in dev set otherwise label as others (last dimension)
        idx = tokenizer.texts_to_sequences([value])
        final_tree.node_type_id="word"
        final_tree.value=WordValue(representation=tf.one_hot(idx[0][0], WordValue.representation_shape))


    #RECURSION
    elif xml_tree.tag == "node" and value=="ROOT":
        label_tree_with_real_data(xml_tree.getchildren()[0], final_tree,tokenizer)

    for child_xml, child_real in zip(xml_tree.getchildren(), final_tree.children):
        label_tree_with_real_data(child_xml, child_real,tokenizer)


def label_tree_with_sentenceTree(train_data, val_data, base_path):
    """
    function that given a tree (target for NN one) without sentence "label" it also with it
    :param train_data:
    :param val_data:
    :param base_path:
    :return:
    """
    #read xml file first
    s=shared_list
    for data in train_data + val_data:
        name = data['name']
        #after got file name, read tree from xml file
        caption_list = []
        for el in listdir(join(base_path,name)):
            caption_list.append( read_tree_from_file(join(base_path,name,el)) )
        data['sentence_trees'] = caption_list

    #count occurency of words
    word_occ = extract_occ(train_data)
    tokenizer,_ = extraxt_topK_words(word_occ,filters="~")
    #label tree with real data
    label_final_trees_with_data(tokenizer, train_data, val_data)


def label_final_trees_with_data(tokenizer, train_data, val_data):
    #TODO troppo lento
    for data in train_data + val_data:
        caption_trees = []
        for el in data['sentence_trees']:
            final_tree = Tree(node_type_id="dummy root", children=[], value="dummy")
            label_tree_with_real_data(el, final_tree, tokenizer)
            final_tree = final_tree.children[0]
            if final_tree.value.abstract_value == "S":
                caption_trees.append(final_tree)
            else:
                idx = shared_list.tags_idx.index("S")
                tag = TagValue(representation=tf.one_hot(idx, len(shared_list.tags_idx)))
                S_node = Tree(node_type_id="POS_tag", children=[final_tree], value=tag)
                caption_trees.append(S_node)
        data['sentence_trees'] = caption_trees

def extraxt_topK_words(word_occ,filters):
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>", filters=filters,lower=True)
    tokenizer.fit_on_texts(word_occ)
    # word number with 5 or more occurrebcy in training
    tokenizer.word_index['<start>'] = 0
    tokenizer.index_word[0] = '<start>'
    shared_list.word_idx = tokenizer.word_index
    shared_list.idx_word = tokenizer.index_word
    WordValue.update_rep_shape(len(tokenizer.word_index.keys()))
    print(WordValue.representation_shape)
    return tokenizer,top_k


"""
def read_tree_from_file(file,embeddings,dictionary,name):

    #open file
    tree = ET.parse(file)
    root = tree.getroot()
    #dummy root
    dummy = Tree(node_type_id="dummy",children=[],value="dummy")
    #get tree really read the tree
    get_tree(dummy, root, dictionary, embeddings,name)
    #return child of dummy root i.e. the real root
    return dummy.children[0]
"""