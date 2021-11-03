from tensorflow_trees.decoder import Decoder, DecoderCellsBuilder
from tensorflow_trees.encoder import Encoder, EncoderCellsBuilder
from myCode.tree_defintions import WordValue
from myCode.CNN_encoder import CNN_Encoder
from myCode.RNN_decoder import NIC_Decoder
import tensorflow_trees.decoder_cells as decoder_cell
import tensorflow_trees.encoder_cells as encoder_cell

def get_encoder_decoder(emb_tree_size, cut_arity, hidden_word,max_arity, max_node_count, max_depth, hidden_coeff,
                        activation,emb_word_size,image_tree, sentence_tree,drop_rate,drop_rate_input):

    decoder_cell.drop_rate = drop_rate
    encoder_cell.drop_rate = drop_rate

    if image_tree==None:
        encoder = CNN_Encoder(emb_tree_size,drop_rate)
    else:
        encoder = Encoder(tree_def=image_tree.tree_def, embedding_size=emb_tree_size, cut_arity=cut_arity, max_arity=max_arity,
                          variable_arity_strategy="FLAT",name="encoder",

                          cellsbuilder=EncoderCellsBuilder(EncoderCellsBuilder.simple_cell_builder(
                              hidden_coef=hidden_coeff, activation=activation,gate=True,drop_rate=drop_rate),

                              EncoderCellsBuilder.simple_dense_embedder_builder(activation=activation,drop_rate=drop_rate_input)))

    WordValue.set_embedding_size(emb_word_size)

    if sentence_tree==None:
        decoder = NIC_Decoder(WordValue.embedding_size,hidden_word,WordValue.representation_shape,drop_rate,beam=3)
    else:
        decoder = Decoder(tree_def=sentence_tree.tree_def, embedding_size=emb_tree_size, max_arity=max_arity,max_depth=max_depth,
                          max_node_count=max_node_count, cut_arity=cut_arity, variable_arity_strategy="FLAT",
                          word_module=NIC_Decoder(WordValue.embedding_size,hidden_word,WordValue.representation_shape,drop_rate,beam=3),

                          cellsbuilder=DecoderCellsBuilder(distrib_builder=DecoderCellsBuilder.simple_distrib_cell_builder(
                              hidden_coef=hidden_coeff,activation=activation,drop_rate=drop_rate),

                              categorical_value_inflater_builder=DecoderCellsBuilder.simple_1ofk_value_inflater_builder(
                                  hidden_coef=hidden_coeff,activation=activation,drop_rate=drop_rate),

                              dense_value_inflater_builder=None,

                              node_inflater_builder=DecoderCellsBuilder.simple_node_inflater_builder(hidden_coef=hidden_coeff,
                                                                                                     activation=activation,gate=True,drop_rate=drop_rate)))

    return decoder, encoder


