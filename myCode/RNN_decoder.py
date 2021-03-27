import tensorflow as tf
import copy
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
        to_return=[]
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
                to_return.append(tf.expand_dims(predictions,axis=1))

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
    def __init__(self,embedding_dim, units, vocab_size,drop_rate,beam):
        super(NIC_Decoder,self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim, name="embedding")
        self.drop_word = tf.keras.layers.Dropout(drop_rate,noise_shape=(None,1,embedding_dim))
        self.rnn =tf.keras.layers.LSTM(units=units, return_state=True, return_sequences=True,recurrent_dropout=drop_rate, name="LSTM")
        self.final_layer = tf.keras.layers.Dense(vocab_size, activation="linear",name="final_word_pred_layer")
        self.drop_final = tf.keras.layers.Dropout(drop_rate)
        self.units = units
        self.beam = 5#beam
        self.vocab_size = vocab_size

    def call(self,pos_embs, images_emb, targets):

        # from targets can discard last ones (no other words to predict after the last ones)
        word_embs = self.embedding_layer(targets[:,:-1])
        word_embs = self.drop_word(word_embs,training=True)

        # concatenate image embedding as first time stamp
        images_emb = tf.expand_dims(images_emb,axis=1)
        word_embs = tf.concat([images_emb,word_embs],axis=1)

        # concatenate also pos tag as inputs in addition to the previous ones
        rnn_input = word_embs #tf.concat([word_embs,pos_embs],axis=-1)

        # call LSTM
        states = [tf.zeros(shape=(pos_embs.shape[0],self.units))]*2
        rnn_output,state_h,state_c= self.rnn(rnn_input, initial_state = states, training=True)
        rnn_output = self.drop_final(rnn_output,training=True)
        #get predictions from last layer
        predictions = self.final_layer(rnn_output)

        return predictions

    def sampling(self,features,pos_embs):

        states = [tf.zeros(shape=(pos_embs.shape[0], self.units))] * 2
        max_length = pos_embs.shape[1]
        to_return=[]

        # sampling of all words in parallel
        for i in range(max_length):
            if i==0:
                current_word_embs = tf.expand_dims(features, axis=1)
            else:
                current_word_embs= self.embedding_layer(tf.argmax(predictions, axis=-1))
            current_pos_embs = tf.expand_dims (pos_embs[:, i, :], axis=1)
            rnn_inputs = current_word_embs #tf.concat([current_word_embs, current_pos_embs], axis=-1)
            rnn_output,states_h,state_c = self.rnn(rnn_inputs, initial_state = states,training=False)
            states=[states_h,state_c]

            predictions=self.final_layer(rnn_output)
            to_return.append(predictions)

        to_return = tf.concat([item for item in to_return],axis=1)
        return to_return

    def beam_search(self,features,pos_embs,sentences_len):
        # helper functions
        def first_words_pred(features):
            # get sentences number and expand image features
            n_sentences = features.shape[0]
            rnn_input = tf.expand_dims(features, axis=1)

            # perform first words prediction and keep for each of them the top-k
            states = [tf.zeros(shape=(n_sentences, self.units))] * 2 # initial states are zero vectors
            rnn_output, state_h, state_c = self.rnn(rnn_input, initial_state=states, training=False)
            predictions = tf.nn.softmax(self.final_layer(rnn_output), axis=-1)
            beam_ris = tf.math.top_k(predictions, k=self.beam,sorted=True)

            # update current generated sequences
            new_sequences = []
            for n in range(n_sentences):
                new_sequences.append([])
                for k in range(self.beam):
                    new_sequences[-1].append([beam_ris.indices[n][0][k]])

            # update states for the next iteration
            idxs = [i for i in range(n_sentences) for j in range(self.beam)]
            states = [tf.gather(state_h,idxs) ,tf.gather(state_c,idxs) ]

            # save in a pair current computed sequences and their probability
            current_preds = (new_sequences, tf.squeeze(tf.math.log(beam_ris.values  )))
            return current_preds, tf.squeeze(beam_ris.indices),states

        def beam_update(beam_ris, current_seqs, states_h,states_c):
            new_sequencies = []
            new_states_h = []
            new_states_c = []
            n_sentences = beam_ris.indices.shape[0]
            #TODO capire se si può vettorizzare la cosa
            for n in range(n_sentences):
                new_sequencies.append([])
                for k in range(self.beam):
                    idx = beam_ris.indices[n][k] % self.vocab_size
                    run = int(tf.math.floor((beam_ris.indices[n][k] / self.vocab_size)))
                    new_sequencies[-1].append(copy.deepcopy(current_seqs[0][n][run]))
                    new_sequencies[-1][-1].append(idx)
                    new_states_h.append(tf.gather(states_h,n*self.beam+run))
                    new_states_c.append(tf.gather(states_c,n*self.beam+run))
            last_words = beam_ris.indices%self.vocab_size
            tmp = (new_sequencies, beam_ris.values)
            new_states = [tf.convert_to_tensor(new_states_h), tf.convert_to_tensor(new_states_c)]
            return tmp, new_states, last_words

        def single_beam_step(current_seqs,last_words,states,sen_len_offsets):
            # reshape last words from (n_sentence,beam) to (n_sentence*beam,1) in order perform a single call to rnn
            #TODO questo reshape spostarlo in dove si produce il vettore delle parole successive
            last_words = tf.reshape(last_words,shape=[-1,1])
            # get relative embedding and call rnn
            prev_word_embedding = self.embedding_layer(last_words)
            rnn_output, state_h, state_c = self.rnn(prev_word_embedding, initial_state=states, training=False)
            # reshaping rnn_output into (n_sentence,beam,n_rnn_units); then get predictions
            rnn_output =tf.reshape(rnn_output,shape=(-1,self.beam,self.units))
            predictions = tf.nn.softmax(self.final_layer(rnn_output), axis=-1)
            log_preds = tf.math.log(tf.squeeze(predictions))

            #TODO capire se si può migliorare sta chifezza guardare tf.stack tk.unstack tf.split , tf.slice
            tot_preds = []
            for n in range(predictions.shape[0]):
                tot_preds.append([])
                for k in range(self.beam):
                    tot_preds[-1].append( log_preds[n][k] + current_seqs[1][n][k])
            tot_preds = tf.convert_to_tensor(tot_preds)
            tot_preds= tf.reshape(tot_preds,shape=(predictions.shape[0],-1))

            # perform beam
            beam_ris = tf.math.top_k(tot_preds, k=self.beam,sorted=True)
            return beam_ris, state_h,state_c


        # main function
        final_pred = []
        max_length = pos_embs.shape[1]
        current_seqs, last_words,states = first_words_pred(features)

        for j in range(1,max_length):
            sen_len_offset = sentences_len-j
            beam_ris, current_states_h,current_states_c = single_beam_step(current_seqs, last_words, states,sen_len_offset)
            current_seqs, states,last_words = beam_update(beam_ris, current_seqs, current_states_h,current_states_c)

        for el in current_seqs[0]:
            final_pred.append(el[0])
        print(tf.argmax(current_seqs[1],axis=1))
        final_pred = tf.convert_to_tensor(final_pred)
        #si può fare anche senza one-hot?
        final_pred = tf.one_hot(final_pred,depth=self.vocab_size)
        return final_pred




"""
        beam_current_results = first_word_predition(features)
        for i in range(1,n_sentences):
            beam_current_results = single_beam_run(beam_current_results)
            a=2
"""