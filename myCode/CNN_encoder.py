import tensorflow as tf

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim,drop_rate):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)
        self.drop = tf.keras.layers.Dropout(rate=drop_rate,name='dropout')

    def call(self, x,**kwargs):
        x = self.fc(x)
        x = self.drop(x,training = kwargs["training"])
        return x