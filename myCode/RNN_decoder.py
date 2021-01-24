import tensorflow as tf
from myCode.word_processing import update_matrix
from myCode.shared_POS_words_lists import word_idx
#TODO unica attention

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden,keep_rate,parents=None):
        # defining attention as a separate model
        context_vector = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        x = tf.nn.dropout(x,keep_prob=keep_rate,noise_shape=[x.shape[0],x.shape[1],1])

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        ctx_vec = tf.expand_dims(context_vector, 1)
        x = tf.concat([ctx_vec, x], axis=-1) if parents==None else tf.concat([ctx_vec,x,parents],axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        x = tf.nn.dropout(x,keep_prob=keep_rate)
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = tf.nn.dropout(x,keep_prob=keep_rate)
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

    def sampling(self,features,wi,iw,max_length,parents=None):

        hidden = self.reset_state(batch_size=features.shape[0])
        dec_input = tf.expand_dims([wi['<start>']] * features.shape[0], 1)
        to_return=None
        #prepare result array
        result =[]
        for i in range(int(features.shape[0])):
            result.append([])

        #sampling of all word in parallel
        for i in range(max_length):
            current_parents = tf.expand_dims (parents[:,i,:],axis=1) if parents!=None else None
            predictions, hidden = self.call(dec_input, features, hidden,parents=current_parents,keep_rate=1.0)
            predicted_ids = tf.squeeze( tf.random.multinomial(predictions, 1), axis=1).numpy()
            #predicted_id = tf.argmax(predictions,axis=-1).numpy()[0]
            dec_input = tf.expand_dims(predicted_ids, 1)

            if parents==None:
                for j in range(len(predicted_ids)):
                    result[j].append(iw[predicted_ids[j]])
            else:
                to_return=update_matrix(tf.expand_dims(predictions,axis=1),to_return,ax=1)

        if parents==None:
            #extract pred sentence up to <end> token
            to_return = []
            for el in result:
                try:
                    end = el.index('<end>')
                    to_return.append(el[:end])
                except ValueError:
                    to_return.append(el)

        return to_return




class NIC_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, vocab_size):
        super(NIC_Decoder, self).__init__()

        self.voab_size=vocab_size
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.LSTM(vocab_size,return_sequences=True,return_state=True)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, tag_inputs,roots,targets_padded_sentences):
        #TODO al momento non uso ne start ne end token. si aggiungono in futuro?
        #TODO posso eliminare l'argmax da qui e non genrare proprio vettore one-hot
        #TODO meglio mettere tf.slice invece dei :
        embeddings = self.embedding_layer(targets_padded_sentences[:,:-1])
        assert (roots.shape[-1] == embeddings.shape[-1]), "embedding dimensions must be the same"
        #concatenate roots to zeros #TODO provare a radopppiarla?
        #TODO in caso cambiare anche in sampling!!!!!!!!!!!
        roots = tf.expand_dims(roots, axis=1)

        image_words_input = tf.concat([roots,embeddings],axis=1)
        full_inputs = tf.concat([image_words_input,tag_inputs],axis=-1)
        intial_state = [tf.zeros(shape=(targets_padded_sentences.shape[0],self.voab_size))]*2
        rnn_out,state_h,state_c= self.rnn(full_inputs,initial_state=intial_state)
        predictions = self.final_layer(rnn_out)
        return predictions

    def sampling(self,inputs,roots):

        predicted_sents = None
        #TODO seguire eventuali cambiamenti che faccio in call!!!!!!!!!!!
        max_sentences_len = inputs.shape[1]
        state = [tf.zeros(shape=(inputs.shape[0],self.voab_size))]*2
        for i in range(0, max_sentences_len):
            if i==0:
                current_input = tf.concat([roots,inputs[:,0,:]],axis=1)
                current_input = tf.expand_dims(current_input,axis=1)
            else:
                current_input = tf.concat([last_predicted_words,tf.expand_dims(inputs[:,i,:],axis=1)],axis=-1)

            rnn_out, state_h, state_c = self.rnn(current_input, initial_state=state)
            state = [state_h, state_c]

            predictions = self.final_layer(rnn_out)
            predicted_sents = update_matrix(predictions, predicted_sents, ax=1)

            last_predicted_words = self.embedding_layer(tf.argmax(predictions, axis=2))
            a=2
        return predicted_sents