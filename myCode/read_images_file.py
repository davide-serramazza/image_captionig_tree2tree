from myCode.tree_defintions import *
from tensorflow_trees.definition import Tree
import numpy as np
import tensorflow as tf
import os
import pickle
#from myCode.tree import *

def read_images(imgs_dir_file,cnn,tree_cnn_type,batch_size):
    """
    function that read images form file
    :param imgs_file:
    :return:
    """
    def read_tree_images(data, imgs_dir_file, tree_cnn_type):
        # open file and read lines
        f = open(imgs_dir_file, "r")
        while True:
            # create dict
            dict = {}

            line = f.readline()
            if line == "":
                break

            tmp = line.split(sep=":")

            # fill dict, begin with name
            dict["name"] = tmp[0][:-1]
            # remove image name form string and split it
            line = line[len(dict["name"]) + 2:]
            tmp = line.split(sep="data")

            # create a dummy root
            tree = reconstruct_tree(data, tmp, tree_cnn_type)
            label_with_node_type(tree, dict["name"])
            if tree_cnn_type == "inception":
                tree.node_type_id = "root"
            dict["img_tree"] = tree

            data.append(dict)

    def read_flat_images(cnn, data, imgs_dir_file,batch_size):
        files = os.listdir(imgs_dir_file)
        for i in range(0,len(files),batch_size):
            curr_batch=[]
            for j in range(i,min(len(files),i+batch_size)):
                file = files[j]
                data.append({'name' : file[:-4]})
                curr_img = tf.io.read_file(imgs_dir_file + "/" + file)
                curr_img = tf.image.decode_jpeg(curr_img, channels=3)
                curr_img = tf.image.resize_images(curr_img, size=(299, 299))
                curr_img  =tf.keras.applications.inception_v3.preprocess_input(curr_img)
                curr_batch.append(curr_img)
            curr_batch = tf.convert_to_tensor(curr_batch)
            batch_features = cnn(curr_batch)
            pool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
            batch_features = pool(batch_features)
            batch_features = tf.reshape(batch_features, shape=(batch_features.shape[0],2048))
            for j in range(batch_features.shape[0]):
                data[i+j]['img'] = batch_features[j]

    def read_tree_input(data,imgs_dir_file):
        with open(imgs_dir_file, "rb") as f:
            dataset = pickle.load(f)
            for el in dataset:
                d= {'name' : el['name'], 'img_tree' : reconstruct_tree2(el['tree'])}
                data.append(d)
    data = []

    if cnn==None:
        #read_tree_images(data, imgs_dir_file, tree_cnn_type)
        read_tree_input(data,imgs_dir_file)
    else:
        read_flat_images(cnn, data, imgs_dir_file,batch_size)
    return data

def label_with_node_type(tree,root):
    if root:
        return  "root"
    else:
        n = len(tree.children)
        if n==0:
            return "leaf"
        elif n>=1 and n<=4:
            return  "internal"
        else:
            raise ValueError

    #for child in tree.children:
    #    label_with_node_type(child,name)




def append_to_tree(in_reconstruction_tree,tree,root):
    in_reconstruction_tree.value = ImageValueResNetRoot(abstract_value=tree.data) if root else ImageValueResNet(abstract_value=tree.data)
    in_reconstruction_tree.node_type_id=label_with_node_type(tree,root)
    for el in tree.children:
        child = Tree(node_type_id='',children=[],meta={'label': el.label})
        in_reconstruction_tree.children.append(child)
        append_to_tree(child,el,root=False)

def reconstruct_tree2(tree):
    root = Tree(node_type_id='root',children=[],meta={'label': 'root'})
    append_to_tree(root,tree,root=True)
    return root




def reconstruct_tree(data, tmp,tree_cnn_type):
    dummy_root = Tree(node_type_id="dummy", children=[], meta={'dummy root'},
        value=ImageValueAlexNet(abstract_value=data) if tree_cnn_type=="alexnet" else ImageValueInception(abstract_value=data))
    last_node = dummy_root
    parent_node = None
    travesed_node = []

    for i in range(1, len(tmp)):
        # loop iterating trough tree nodes

        data = get_node_value(tmp[i])

        count = tmp[i - 1].count(")")

        # if is current node child
        if tmp[i - 1].__contains__("("):
            leaf_n = tmp[i - 1].split("(")
            new_node = Tree(node_type_id="", children=[],meta={'label': leaf_n[1]},
                value=ImageValueAlexNet(abstract_value=data) if tree_cnn_type=="alexnet" else ImageValueInception(abstract_value=data))
            parent_node = last_node

            # update list of all internal node traversed useful later
            travesed_node.append(parent_node)

            parent_node.children.append(new_node)
            last_node = new_node

        elif count:
            # if listed all child of same node

            # deleted from list all node no more used
            travesed_node = travesed_node[:len(travesed_node) - count]
            parent_node = travesed_node[-1]

            leaf_n = tmp[i - 1].split("),")
            new_node = Tree(node_type_id="leaf", children=[], meta={'label': leaf_n[1]},
            value=ImageValueAlexNet(abstract_value=data) if tree_cnn_type=="alexnet" else ImageValueInception(abstract_value=data))
            parent_node.children.append(new_node)
            last_node = new_node

        else:
            # if is current node sibling
            leaf_n = tmp[i - 1].split("],")
            new_node = Tree(node_type_id="leaf", children=[], meta={'label': leaf_n[1]},
            value=ImageValueAlexNet(abstract_value=data) if tree_cnn_type=="alexnet" else ImageValueInception(abstract_value=data))
            parent_node.children.append(new_node)
            last_node = new_node

    return dummy_root.children[0]


def get_node_value(str):
    """
    #TODO man
    function converting single line of file as tf vector
    :param str:
    :return:
    """
    val = str.split("[")
    val = val[1].split("]")
    data = np.fromstring(val[0], dtype=np.float32, sep=", ")
    return tf.convert_to_tensor(data, dtype=tf.float32)
