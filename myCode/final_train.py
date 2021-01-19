import sys
sys.path.insert(0,'/home/serramazza/tf_tree_new')

from myCode.models import *
from myCode.tree_defintions import *
from myCode.helper_functions import *
from myCode.validation import train_model


def main():
    global embeddings
    global tag_dictionary

    #check arguments
    args = sys.argv[1:]
    if len(sys.argv)==0 or args[0] == "--help":
        help()
        exit(0)

    print ("begin")


    ##################
    # FLAGS
    ##################
    FLAGS = tf.flags.FLAGS
    define_flags()
    tf.enable_eager_execution()

    ###################
    # tree definition
    ###################
    image_tree = ImageTree()
    sentence_tree = SentenceTree()


    #TODO nominare le matrici dei pesi con nomi significativi
    #TODO gestire separatamente le due loss/accuracy di pos e parole
    #TODO rimplementare tutti i metoi che misurano accuracy
    #TODO gestire in abstract to rep e viceversa quando ho in input lista e qaundo stringa (forse ho già fatto??)
    #TODO sistmare anche load model
    #TODO una volta "fatto tutto" cambiare wor not found in "" o " "
    #TODO cancellare tutti pesi inutili tipo dense inflater

    #load tree
    dev_data, test_data = load_all_data(args)

    #get batch for traning
    input_dev = get_image_batch(dev_data)
    target_dev = get_sentence_batch(dev_data)
    input_test = get_image_batch(test_data)
    target_test = get_sentence_batch(test_data)

    image_max_arity, sen_max_arity = get_max_arity(input_dev, input_test, target_dev, target_test)

    #parameters

    embedding_size = 100
    cut_arity = 5
    hidden_coeff = 1
    lamb = 0.02
    beta= 0.005
    max_node_count = 100
    max_depth = 5
    clipping =  0.02
    batch_size = 32
    learning_rate = 0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


    activation = getattr(tf.nn, FLAGS.activation)

    decoder, encoder = get_encoder_decoder(emb_size=embedding_size,cut_arity=cut_arity,max_arity=
        max(image_max_arity,sen_max_arity),max_node_count=max_node_count,max_depth=max_depth,
        hidden_coeff=hidden_coeff,activation=activation,
        image_tree=image_tree.tree_def,sentence_tree=sentence_tree.tree_def)

    print("begin train")

    train_model(FLAGS=FLAGS,decoder=decoder,encoder=encoder,input_train=input_dev,input_val=input_test,
                target_train=target_dev,target_val=target_test,optimizer=optimizer,beta=beta,lamb=lamb,clipping=clipping,
                batch_size=batch_size,n_exp=203, name="esperimenti", final=True, test=test_data)

    print("finished")

if __name__ == "__main__":
    main()
