from tensorflow_trees.definition import TreeDefinition, NodeDefinition
import tensorflow as tf
import myCode.shared_POS_words_lists as shared_list
import numpy as np
import sys

###########
#image tree
###########
class ImageValueAlexNet(NodeDefinition.Value):
    """
    class modelling single tree image node
    """
    representation_shape = 256
    class_value = False

    @staticmethod
    def representation_to_abstract_batch(t:tf.Tensor):
        return t.numpy()

    @staticmethod
    def abstract_to_representation_batch(v):
        return tf.Variable(v,dtype=tf.float32)

class ImageValueInception(NodeDefinition.Value):
    """
    class modelling single tree image node
    """
    representation_shape = 192
    class_value = False

    @staticmethod
    def representation_to_abstract_batch(t:tf.Tensor):
        return t.numpy()

    @staticmethod
    def abstract_to_representation_batch(v):
        return tf.Variable(v,dtype=tf.float32)

class ImageValueInceptionRoot(NodeDefinition.Value):
    """
    class modelling single tree image node
    """
    representation_shape = 2048
    class_value = False

    @staticmethod
    def representation_to_abstract_batch(t:tf.Tensor):
        return t.numpy()

    @staticmethod
    def abstract_to_representation_batch(v):
        return tf.Variable(v,dtype=tf.float32)


class ImageTree:
    """
    class modelling whole image tree
    """
    def __init__(self,tree_cnn_type):


        if tree_cnn_type=="alexnet":

            self.tree_def = TreeDefinition(node_types=[
                NodeDefinition("othersInternal",may_root=True,arity=NodeDefinition.VariableArity(min_value=5),value_type=ImageValueAlexNet),
                NodeDefinition("doubleInternal",may_root=True,arity=NodeDefinition.FixedArity(4),value_type=ImageValueAlexNet),
                NodeDefinition("internal",may_root=True,arity=NodeDefinition.FixedArity(2),value_type=ImageValueAlexNet),
                NodeDefinition("leaf",may_root=False,arity=NodeDefinition.FixedArity(0),value_type=ImageValueAlexNet)
            ])

            self.node_types = self.tree_def.node_types

        elif tree_cnn_type=="inception":
            self.tree_def = TreeDefinition(node_types=[
                NodeDefinition("root",may_root=True,arity=NodeDefinition.VariableArity(min_value=0),value_type=ImageValueInceptionRoot),
                NodeDefinition("othersInternal",may_root=False,arity=NodeDefinition.VariableArity(min_value=5),value_type=ImageValueInception),
                NodeDefinition("doubleInternal",may_root=False,arity=NodeDefinition.FixedArity(4),value_type=ImageValueInception),
                NodeDefinition("internal",may_root=False,arity=NodeDefinition.FixedArity(2),value_type=ImageValueInception),
                NodeDefinition("leaf",may_root=False,arity=NodeDefinition.FixedArity(0),value_type=ImageValueInception)
            ])

            self.node_types = self.tree_def.node_types

        else:
            sys.exit("cnn type should be inception or alexnet")



#############
#sentence tree
#############


class TagValue(NodeDefinition.Value):
    """
    class modelling POS tag vale as one hot encoding
    """
    representation_shape = 0 #n* of different pos tag in flick8k train set
    class_value = True

    @staticmethod
    def update_rep_shape(shape):
        TagValue.representation_shape = shape

    @staticmethod
    def representation_to_abstract_batch(t:tf.Tensor):
        s=shared_list
        idx = t[0].numpy()
        if type(idx)==np.int32:
            try:
                ris = shared_list.idx_tag[idx]
            except IndexError:
                ris = "not_found"
            return ris
        elif type(idx)==np.ndarray:
            return shared_list.idx_tag[np.argmax(idx)]
        else:
            raise ValueError ("tag value of unknown type")

    @staticmethod
    def abstract_to_representation_batch(v):
        """
        return the associated value to the key v(argument of the function)
        :param v:
        :return:
        """
        ris=[]
        for el in v:
            idx = shared_list.tag_idx[el]
            ris.append( tf.one_hot(idx, TagValue.representation_shape ) )
        return ris


class WordValue(NodeDefinition.Value):
    """
    class modelling word value i.e. emebedding vector
    """
    representation_shape = 1    #word number in dataset
    embedding_size=0    #embedding dimension
    class_value = True

    @staticmethod
    def update_rep_shape(shape):
        WordValue.representation_shape = shape

    @staticmethod
    def set_embedding_size(embedding_dim):
        WordValue.embedding_size=embedding_dim

    @staticmethod
    def representation_to_abstract_batch(t:tf.Tensor):
        idx = t[0].numpy()
        if type(idx)==np.int32:
            try:
                ris = shared_list.idx_word[idx]
            except IndexError:
                ris = ""
            return ris
        elif type(idx)==np.ndarray:
            return shared_list.idx_word[np.argmax(idx)]
        else:
            raise ValueError ("word value of unknown type")

    @staticmethod
    def abstract_to_representation_batch(v):
        """
        return the associated value to the key v(argument of the function)
        :param v:
        :return:
        """
        ris=[]
        for el in v:
            idx = shared_list.word_idx[el]
            ris.append( tf.one_hot(idx,WordValue.representation_shape) )
        return ris


class SentenceTree:
    """
    class representing sentence tree.
    """
    def __init__(self):
        self.tree_def = TreeDefinition(node_types=[
            NodeDefinition("POS_tag",may_root=True,arity=NodeDefinition.VariableArity(min_value=1),value_type=TagValue),
            NodeDefinition("word",may_root=False,arity=NodeDefinition.FixedArity(0),value_type=WordValue)
        ])

        self.node_types = self.tree_def.node_types
