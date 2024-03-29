from tensorflow_trees.definition import TreeDefinition, NodeDefinition
import tensorflow as tf
import myCode.shared_POS_words_lists as shared_list
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



class ImageValueResNetRoot(NodeDefinition.Value):
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

class ImageValueResNet(NodeDefinition.Value):
    """
    class modelling single tree image node
    """
    representation_shape = 1024
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
                NodeDefinition("internal",may_root=False,arity=NodeDefinition.FixedArity(2),value_type=ImageValueInception),
                NodeDefinition("leaf",may_root=False,arity=NodeDefinition.FixedArity(0),value_type=ImageValueInception)
            ])

            self.node_types = self.tree_def.node_types

        elif tree_cnn_type=="resNet":

            self.tree_def = TreeDefinition(node_types=[
                #NodeDefinition("othersInternal",may_root=True,arity=NodeDefinition.VariableArity(min_value=5),value_type=ImageValueResNet),
                #NodeDefinition("doubleInternal",may_root=True,arity=NodeDefinition.FixedArity(4),value_type=ImageValueResNet),
                NodeDefinition("root",may_root=True,arity=NodeDefinition.VariableArity(4),value_type=ImageValueResNetRoot),
                NodeDefinition("internal",may_root=False,arity=NodeDefinition.VariableArity(4),value_type=ImageValueResNet),
                NodeDefinition("leaf",may_root=False,arity=NodeDefinition.FixedArity(0),value_type=ImageValueResNet)
            ])

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
        if t.shape==(1,):
            # target
            idx = (t[0]).numpy()
        elif t.shape==(1,TagValue.representation_shape):
            # generated
            idx = tf.argmax(t[0]).numpy()
        else:
            raise ValueError("Word of unknown shape")
        return shared_list.idx_tags[idx]

    @staticmethod
    def abstract_to_representation_batch(v):
        """
        return the associated value to the key v(argument of the function)
        :param v:
        :return:
        """
        ris=[]
        for el in v:
            idx = shared_list.tags_idx[el]
            ris.append( tf.constant(idx ) )
        return tf.convert_to_tensor(ris)

class WordValue(NodeDefinition.Value):
    """
    class modelling word value i.e. emebedding vector
    """
    representation_shape = 0    #word number in dataset
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
        if t.shape==(1,):
            # target
            idx = (t[0]).numpy()
        elif t.shape==(1,WordValue.representation_shape):
            # generated
            idx = tf.argmax(t[0]).numpy()
        else:
            raise ValueError("Word of unknown shape")
        return shared_list.tokenizer.index_word[idx]

    @staticmethod
    def abstract_to_representation_batch(v):
        """
        return the associated value to the key v(argument of the function)
        :param v:
        :return:
        """
        #if type(v)==list:
        ris=[]
        for el in v:
            idx = shared_list.tokenizer.word_index[el]
            ris.append( tf.constant(idx) )
        return tf.convert_to_tensor(ris)


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
