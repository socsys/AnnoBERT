import tensorflow as tf
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import Layer, Input, Concatenate, Dense, Dropout, Embedding
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


class AttentionDot(Layer):

    def __init__(self, **kwargs):
    
        super().__init__(** kwargs)

    def build(self, input_shape):
    
        self.W = self.add_weight(name='Attention_Dot_Weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 regularizer=L1L2(0.0000032),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='Attention_Dot_Bias',
                                 regularizer=L1L2(0.00032),
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, input):
    
        x_transpose = K.permute_dimensions(input, (0, 2, 1))
        x_tanh_softmax = K.softmax(K.tanh(K.dot(x_transpose, self.W) + self.b))
        outputs = K.permute_dimensions(x_tanh_softmax * x_transpose, (0, 2, 1))

        return outputs

    def compute_output_shape(self, input_shape):

        return input_shape[0], input_shape[1], input_shape[2]


class CVG_Layer(Layer):

    def __init__(self, embed_size, filter, label, **kwargs):

        self.embed_size = embed_size
        self.filter = filter
        self.label = label
        super().__init__(** kwargs)

    def build(self, input_shape):

        self._filter = self.add_weight(name=f'filter_{self.filter}',
                                       shape=(self.filter, self.label, 1, 1),
                                       regularizer=L1L2(0.00032),
                                       initializer='uniform',
                                       trainable=True)
        self.class_w = self.add_weight(name='class_w',
                                       shape=(self.label, self.embed_size),
                                       regularizer=L1L2(0.0000032),
                                       initializer='uniform',
                                       trainable=True)
        self.b = self.add_weight(name='bias',
                                 shape=(1,),
                                 regularizer=L1L2(0.00032),
                                 initializer='uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, input):

        # C * V / G
        # l2_normalize of x, y
        input_norm = tf.nn.l2_normalize(input)  # b * s * e
        class_w_relu = tf.nn.relu(self.class_w) # c * e
        label_embedding_reshape = tf.transpose(class_w_relu, [1, 0])  # e * c
        label_embedding_reshape_norm = tf.nn.l2_normalize(label_embedding_reshape)  # e * c

        # C * V
        G = K.dot(input_norm, label_embedding_reshape_norm)  # b * s * c
        G_transpose = tf.transpose(G, [0, 2, 1])  # b * c * s
        G_expand = tf.expand_dims(G_transpose, axis=-1)  # b * c * s * 1

        # text_cnn
        conv = tf.nn.conv2d(name='conv', input=G_expand, filters=self._filter,
                            strides=[1, 1, 1, 1], padding='SAME')
        pool = tf.nn.relu(name='relu', features=tf.nn.bias_add(conv, self.b))  # b * c * s * 1

        # max_pool
        pool_squeeze = tf.squeeze(pool, axis=-1)  # b * c * s
        pool_squeeze_transpose = tf.transpose(pool_squeeze, [0, 2, 1])  # b * s * c
        G_max_squeeze = tf.reduce_max(input_tensor=pool_squeeze_transpose, axis=-1, keepdims=True)  # b * s * 1
        
        # divide of softmax
        exp_logits = tf.exp(G_max_squeeze)
        exp_logits_sum = tf.reduce_sum(exp_logits, axis=1, keepdims=True)
        att_v_max = tf.compat.v1.div(exp_logits, exp_logits_sum)
        
        # β * V
        x_att = tf.multiply(input, att_v_max)
        x_att_sum = tf.reduce_sum(x_att, axis=1)
        
        return x_att_sum

    def compute_output_shape(self, input_shape):
        
        return None, K.int_shape(self.class_w)[1]


def LEAM(num_classes,vocab,emb_matrix,seq_length=75,embedding_dim=300,filters=[3,4,5],keep_prob=0.5):

    input = Input(name='input_text', shape=(seq_length,))

    embedding = Embedding(len(vocab) + 1, embedding_dim, input_length=seq_length, weights=[emb_matrix])(input)
    word_embedding_attention = AttentionDot()(embedding)

    # 1.C*V/G;  
    # 2.relu(text-cnn);  
    # 3.softmax;  
    # 4.β * V
    pools = []
    for filter in filters:
        x_cvg = CVG_Layer(embedding_dim, filter, num_classes)(word_embedding_attention)
        pools.append(x_cvg)

    pools_concat = Concatenate()(pools)

    # MLP
    x_cvg_dense = Dense(int(embedding_dim/2), activation="relu")(pools_concat)
    x = Dropout(keep_prob)(x_cvg_dense)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-7,momentum=0.9)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss = loss, optimizer=optimizer, metrics=['acc'])

    print(model.summary())

    return model

