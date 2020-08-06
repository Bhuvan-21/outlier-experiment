import tensorflow as tf
class model(tf.keras.Model):

    def __init__(self):
        super(model, self).__init__();

        self.dense1=tf.keras.layers.Dense(64,activation='relu')
        self.batch_norm1=tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(1, activation='linear')

    def call(self,x,training=True):

        x=self.dense1(x)
        x=self.batch_norm1(x,training=training)
        x = self.dense2(x)
        x = self.batch_norm2(x,training=training)
        x = self.dense3(x)
        return x