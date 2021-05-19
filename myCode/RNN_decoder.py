import tensorflow as tf
import numpy as np
import myCode.shared_POS_words_lists as shared_list

class NIC_Decoder(tf.keras.Model):
    def __init__(self,embedding_dim, units, vocab_size,drop_rate,beam):
        super(NIC_Decoder,self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim, name="embedding")
        self.drop_word = tf.keras.layers.Dropout(drop_rate,noise_shape=(1,None,embedding_dim))
        self.rnn =tf.keras.layers.LSTM(units=units, return_state=True, return_sequences=True,recurrent_dropout=drop_rate, name="LSTM")
        self.drop_final = tf.keras.layers.Dropout(drop_rate,noise_shape=(1,None,units))
        self.final_layer = tf.keras.layers.Dense(vocab_size, activation="linear",name="final_word_pred_layer")
        self.units = units
        self.beam = beam
        self.vocab_size = vocab_size

    def call(self,pos_embs, images_emb, targets):

        # from targets can discard last ones (no other words to predict after the last ones)
        word_embs = self.embedding_layer(targets[:,:-1])
        word_embs = self.drop_word(word_embs,training=True)

        # concatenate image embedding as first time stamp
        images_emb = tf.expand_dims(images_emb,axis=1)
        word_embs = tf.concat([images_emb,word_embs],axis=1)

        # concatenate also pos tag as inputs in addition to the previous ones
        rnn_input = word_embs

        # call LSTM
        states = [tf.zeros(shape=(images_emb.shape[0],self.units))]*2
        rnn_output,state_h,state_c= self.rnn(rnn_input, initial_state = states, training=True)
        rnn_output = self.drop_final(rnn_output,training=True)
        predictions = self.final_layer(rnn_output)

        return predictions

    def sampling(self,features,pos_embs=None,max_length=None):

        states = [tf.zeros(shape=(features.shape[0], self.units))] * 2
        if max_length is not None:
            end_token = shared_list.tokenizer.word_index['<end>']
            end_generation =[False]*features.shape[0]
        max_length = pos_embs.shape[1] if max_length==None else max_length
        to_return=[]


        # sampling of all words in parallel
        for i in range(max_length):
            if i==0:
                current_word_embs = tf.expand_dims(features, axis=1)
            elif i==1 and max_length is not None:
                start_token = shared_list.tokenizer.word_index['<start>']
                start_vectors = tf.expand_dims(tf.tile([start_token],multiples=[features.shape[0]]),axis=-1)
                current_word_embs = self.embedding_layer(start_vectors)
            else:
                current_word_embs= self.embedding_layer(tf.argmax(predictions, axis=-1))
            rnn_inputs = current_word_embs
            rnn_output,states_h,state_c = self.rnn(rnn_inputs, initial_state = states,training=False)
            states=[states_h,state_c]

            predictions=self.final_layer(rnn_output)
            to_return.append(predictions)

            ended_sequences = tf.where(tf.equal( tf.squeeze(tf.argmax(predictions,axis=-1)) , end_token))
            for idx in ended_sequences:
                end_generation[idx]=True
            if all(end_generation):
                break

        to_return = tf.concat([item for item in to_return],axis=1)
        return to_return

    def beam_search(self,features,pos_embs,sentences_len):
        # helper functions
        def first_words_pred(features,current_pos):
            # get sentences number and expand image features
            n_sentences = features.shape[0]
            rnn_input = tf.expand_dims( features,axis=1)

            # perform first words prediction and keep for each of them the top-k
            states = [tf.zeros(shape=(n_sentences, self.units))] * 2 # initial states are zero vectors
            rnn_output, state_h, state_c = self.rnn(rnn_input, initial_state=states, training=False)
            predictions = tf.nn.softmax(self.final_layer(rnn_output), axis=-1)
            beam_ris = tf.math.top_k(predictions, k=self.beam,sorted=True)

            # update current generated sequences
            new_sequences=tf.reshape(beam_ris.indices,shape=(n_sentences,-1,1))

            # update states for the next iteration
            idxs = [i for i in range(n_sentences) for j in range(self.beam)]
            states = [tf.gather(state_h,idxs) ,tf.gather(state_c,idxs) ]

            # save in a pair current computed sequences and their probability
            current_preds = {
                'k_sequences' : new_sequences,
                'probs' : tf.squeeze(tf.math.log(beam_ris.values)),
                'computed_seqs' : [],
                'computed_idxs' : [],
                'pred_probs' : []}
            return current_preds, tf.reshape(beam_ris.indices,shape=[-1,1]), states

        def beam_update(beam_ris, current_seqs,sen_len_offset,states_h,states_c):
            # get number of sentences and their 'length offset'
            n_sentences = beam_ris.indices.shape[0]
            sen_len_offset = sen_len_offset-1

            # update previous k sequences of length n-1 with the new computed words to rich length n
            current_pred_words = tf.unstack(beam_ris.indices % self.vocab_size)
            current_runs = tf.cast(tf.math.floor(beam_ris.indices / self.vocab_size),dtype=tf.int32)
            prev_sequencies = [tf.gather(current_seqs['k_sequences'][i],current_runs[i]) for i in range(n_sentences)]
            new_sequencies =[tf.concat([ prev , tf.expand_dims(curr,axis=-1) ],axis=-1) for (prev,curr) in
                             zip(prev_sequencies,current_pred_words)]

            # save the sequences which have reached their length
            computed_seqs = np.where(sen_len_offset==0)
            for i in computed_seqs[0]:
                predicted = new_sequencies[i][0]
                current_seqs['computed_seqs'].append(predicted)#tf.pad(predicted,[[0,max_length-len]],'CONSTANT'))
                current_seqs['computed_idxs'].append(i)
                current_seqs['pred_probs'].append(current_seqs['probs'][i][0])

            # keep track of RNN states, sequences and related probabilities and last words for the next iterations
            states_idxs = [current_runs[i]+i*self.beam for i in range(n_sentences)]
            states_idxs = tf.concat([t for t in states_idxs],axis=-1)
            new_states = [tf.gather(states_h,states_idxs),tf.gather(states_c,states_idxs)]
            last_words = tf.reshape(beam_ris.indices%self.vocab_size,shape=(-1,1))
            current_seqs['k_sequences'] = tf.convert_to_tensor(new_sequencies)
            current_seqs['probs'] = beam_ris.values

            return current_seqs, new_states, last_words,sen_len_offset

        def single_beam_step(current_seqs,last_words,current_pos,states,sen_len_offsets):
            # get relative embedding and call rnn
            n_sens = sen_len_offsets.shape[0]

            # from embedding to predictions
            prev_word_embedding = self.embedding_layer(last_words)
            rnn_input = prev_word_embedding
            rnn_output, state_h, state_c = self.rnn(rnn_input, initial_state=states, training=False)
            predictions = tf.nn.softmax(self.final_layer(rnn_output), axis=-1)
            # reshape preds as (n_sens, beam,vocab_dim)  and get logarithm
            predictions =tf.reshape(predictions,shape=(-1,self.beam,self.vocab_size))
            log_preds = tf.math.log(tf.squeeze(predictions))

            # compute current sentence probabilities
            prev_probs = tf.expand_dims(current_seqs['probs'],axis=-1)
            prev_probs = tf.tile(prev_probs,[1,1,self.vocab_size])
            current_preds = tf.reshape((log_preds+prev_probs),shape=(n_sens,-1))

            # perform beam search
            beam_ris = tf.math.top_k(current_preds, k=self.beam,sorted=True)
            return beam_ris, state_h,state_c


        # main function
        max_length = max(sentences_len)
        assert min(sentences_len)>=2
        current_seqs, last_words,states = first_words_pred(features,pos_embs[:,0,:])
        sen_len_offset = sentences_len-1

        # main loop
        for j in range(1,max_length):
            beam_ris, current_states_h,current_states_c = single_beam_step(current_seqs, last_words,pos_embs[:,j,:], states,sen_len_offset)
            current_seqs, states,last_words, sen_len_offset = \
                beam_update(beam_ris, current_seqs,sen_len_offset, current_states_h,current_states_c,)

        # after the loop, take the sentences predicted, sort them and transform indexes to one_hot vecotrs
        idxs = np.argsort(current_seqs['computed_idxs'])
        preds = current_seqs['computed_seqs']
        final_pred = [tf.one_hot(preds[i],depth=self.vocab_size) for i in idxs]
        for i in range(len(sentences_len)):
            assert  final_pred[i].shape[0]==sentences_len[i]
        print(tf.reduce_mean(tf.math.exp(current_seqs['pred_probs'])))
        return final_pred