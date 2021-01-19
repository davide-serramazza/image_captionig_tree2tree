from myCode.tree_defintions import *
from tensorflow_trees.definition import Tree
import numpy as np
import tensorflow as tf
import os

def read_images(imgs_dir_file,cnn,tree_cnn_type):
    """
    function that read images form file
    :param imgs_file:
    :return:
    """
    data = []
    if cnn==None:
        #open file and read lines
        f= open(imgs_dir_file,"r")
        while True:
            # create dict
            dict ={}

            line = f.readline()
            if line=="":
                #if i'm here file is ended
                break

            tmp = line.split(sep=":")

            #fill dict, begin with name
            dict["name"] = tmp[0][:-1]
            #remove image name form string and split it
            line = line[len(dict["name"])+2:]
            tmp = line.split(sep="data")

            #create a dummy root
            tree = reconstruct_tree(data, tmp,tree_cnn_type)
            label_with_node_type(tree, dict["name"])
            if tree_cnn_type=="inception":
                tree.node_type_id="root"
            dict["img_tree"] = tree

            data.append(dict)
    else:
        files = os.listdir(imgs_dir_file)
        for file in files:
            name= file[:-4]
            img = tf.io.read_file(imgs_dir_file+"/"+file)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize_images(img,size=(299,299)) #non riesco a fare resize
            img = tf.keras.applications.inception_v3.preprocess_input(img)
            img = tf.expand_dims(img,axis=0)
            batch_features = cnn(img)
            batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))
            img = None
            data.append({"name" : name, "img":batch_features})
    return data


def label_with_node_type(tree,name):
    n = len(tree.children)
    if n==0:
        tree.node_type_id="leaf"
    elif n==2:
        tree.node_type_id="internal"
    elif n==4:
        tree.node_type_id="doubleInternal"
    elif n>4:
        tree.node_type_id="othersInternal"
    else:
        tree.node_type_id="othersInternal"
        print("child number is ", n, tree.node_type_id,name )

    for child in tree.children:
        label_with_node_type(child,name)


def reconstruct_tree(data, tmp,tree_cnn_type):
    dummy_root = Tree(node_type_id="dummy", children=[], meta={'dummy root'},
        value=ImageValueAlexNet(abstract_value=data) if tree_cnn_type=="alexnet" else ImageValueInception(abstract_value=data))
    last_node = dummy_root
    parent_node = None
    travesed_node = []

    for i in range(1, len(tmp)):
        #loop iterating trough tree nodes

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
    function converting single line of file as tf vector
    :param str:
    :return:
    """
    val = str.split("[")
    val = val[1].split("]")
    data = np.fromstring(val[0], dtype=np.float32, sep=", ")
    return tf.convert_to_tensor(data, dtype=tf.float32)
