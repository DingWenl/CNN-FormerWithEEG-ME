
from keras.layers import Permute,Reshape,Conv2D,BatchNormalization,Dense,Activation,Dropout,Flatten,Conv1D
import tensorflow as tf
import numpy as np
# Setting hyper-parameters
K = 40

drop_out1 = 0.5
drop_out2 = 0.1
drop_out3 = 0.95

channel = 9
out_channel = 8
out_channel2 = 128
out_channel3 = out_channel2//4
activation = 'elu'

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  

  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):

  matmul_qk = tf.matmul(q, k, transpose_b=True)  

  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  

  output = tf.matmul(attention_weights, v)  

  return output, attention_weights


def split_heads(x, num_heads, depth): 
  x = Reshape(( -1, num_heads, depth))(x)
  return Permute((2, 1, 3))(x)

def MultiHeadAttention_(x, d_model, num_heads):
    q = Dense(d_model)(x)
    k = Dense(d_model)(x)
    v = Dense(d_model)(x)

    depth = d_model // num_heads
    q = split_heads(q, num_heads, depth)
    k = split_heads(k, num_heads, depth)
    v = split_heads(v, num_heads, depth)

    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask=None)
    scaled_attention = Permute((2, 1, 3))(scaled_attention)  

    concat_attention = Reshape((-1, d_model))(scaled_attention)  

    output = Dense(d_model)(concat_attention)  
        
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='elu'),  
      tf.keras.layers.Dense(d_model)  
  ])

def EncoderLayer(x, d_model, num_heads,dff,rate,num):
    attn_output, _ = MultiHeadAttention_(x, d_model, num_heads)
    attn_output = Dropout(rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    ffn_output = point_wise_feed_forward_network(d_model, dff)(out1)
    ffn_output = Dropout(rate)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name = 'out2_%d'%num)(out1 + ffn_output)

    return out2

def conv1D_block_(x, k_size, stride, out_channel, drop_out,name):
    x = Dropout(drop_out)(x)
    x = Conv1D(out_channel,kernel_size = k_size,strides = stride,padding = 'same',name = name)(x)
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)

    return x

def multi_scale_1D(x, first_k,firt_step, res_channel, drop_out1):
    x1 = conv1D_block_(x, first_k, firt_step, res_channel,drop_out1,'1x1_x1')
    x2 = conv1D_block_(x, first_k, firt_step, res_channel,drop_out1,'1x1_x2')
    x3 = conv1D_block_(x, first_k, firt_step, res_channel,drop_out1,'1x1_x3')
    x4 = conv1D_block_(x, first_k, firt_step, res_channel,drop_out1,'1x1_x4')
    
    x2_2 = conv1D_block_(x2, 32, 1, res_channel,drop_out1,'1x32')
    x3_2 = x3 + x2_2
    x3_3 = conv1D_block_(x3_2,16, 1, res_channel,drop_out1,'1x16')
    x4_3 = x4 + x3_3
    x4_4 = conv1D_block_(x4_3, 11, 1, res_channel,drop_out1,'1x11')

    x = x1 + x2_2 + x3_3 + x4_4

    return x

def former_encoder(x, num_layers,d_model, num_heads, dff, rate,maximum_position_encoding):
    x += positional_encoding(maximum_position_encoding,d_model)[:, :x.shape[1], :]
    for i in range(num_layers):
        x = Dropout(rate)(x)
        x = EncoderLayer(x, d_model, num_heads,dff,rate,i)
    return x

def cnnformer(inputs):
    ### the CNN module
    # the first convolution layer
    x = Conv2D(out_channel,kernel_size = (inputs.shape[1], 1),strides = 1,padding = 'valid')(inputs)
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)
    x = Reshape((x.shape[2],x.shape[3]))(x)
    # the multi-scale block, different convolution kernels are designed to learn the information of different temporal scales
    x = multi_scale_1D(x, 1,1, out_channel*4, drop_out1)
    # the last convolution layer
    x = conv1D_block_(x, x.shape[1], 5, out_channel2, drop_out1,'en-code_5')
    # the Transformer module, for more details you can refer to : https://www.tensorflow.org/tutorials/text/transformer?hl=zh-cn#encoder_and_decoder
    x = former_encoder(x, num_layers=2,d_model=out_channel2, num_heads=4, dff=out_channel2//2, rate=drop_out2,maximum_position_encoding=5000)

    x = Flatten()(x)
    x = Dropout(drop_out3)(x)
    # # the fully connected layer and "softmax"
    x = Dense(K,activation='softmax')(x)
    return x

