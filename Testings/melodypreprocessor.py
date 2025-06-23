import numpy as np
import tensorflow as tf
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
)

def sinusoidal_position_encoding(num_positions, d_model):
    angles = _get_angles(
        np.arange(num_positions)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )
    sines = np.sin(angles[:, 0::2])
    cosines = np.cos(angles[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

def _get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        input_vocab_size,
        target_vocab_size,
        max_num_positions_in_pe_encoder,
        max_num_positions_in_pe_decoder,
        dropout_rate=0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers, d_model, num_heads, d_feedforward,
            input_vocab_size, max_num_positions_in_pe_encoder, dropout_rate
        )
        self.decoder = Decoder(
            num_layers, d_model, num_heads, d_feedforward,
            target_vocab_size, max_num_positions_in_pe_decoder, dropout_rate
        )
        self.final_layer = Dense(target_vocab_size)

    def call(
        self,
        input,
        target,
        *,
        training=False,
        enc_padding_mask=None,
        look_ahead_mask=None,
        dec_padding_mask=None,
    ):
        enc_output = self.encoder(input, training=training, mask=enc_padding_mask)
        dec_output = self.decoder(
            target, enc_output, training=training,
            look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask
        )
        return self.final_layer(dec_output)

class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers, d_model, num_heads, d_feedforward,
        input_vocab_size, maximum_positions_in_pe, dropout_rate=0.1,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = sinusoidal_position_encoding(maximum_positions_in_pe, d_model)
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, d_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, *, training=False, mask=None):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :x.shape[1], :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers, d_model, num_heads, d_feedforward,
        target_vocab_size, maximum_positions_in_pe, dropout_rate=0.1,
    ):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = sinusoidal_position_encoding(maximum_positions_in_pe, d_model)
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, d_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

    def call(
        self, x, enc_output, *,
        training=False, look_ahead_mask=None, padding_mask=None
    ):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :x.shape[1], :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, enc_output, training=training,
                look_ahead_mask=look_ahead_mask, padding_mask=padding_mask
            )
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_feedforward, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(d_feedforward, activation="relu"),
            Dense(d_model),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, *, training=False, mask=None):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_feedforward, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(d_feedforward, activation="relu"),
            Dense(d_model),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(
        self, x, enc_output, *,
        training=False, look_ahead_mask=None, padding_mask=None
    ):
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout1(attn1, training=training))
        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        out2 = self.layernorm2(out1 + self.dropout2(attn2, training=training))
        ffn_output = self.ffn(out2)
        out3 = self.layernorm3(out2 + self.dropout3(ffn_output, training=training))
        return out3

if __name__ == "__main__":
    # Define Transformer parameters
    num_layers = 2
    d_model = 64
    num_heads = 2
    d_feedforward = 128
    input_vocab_size = 100
    target_vocab_size = 100
    dropout_rate = 0.1
    pe_input = 100
    pe_target = 100

    # Instantiate the Transformer model
    transformer_model = Transformer(
        num_layers, d_model, num_heads, d_feedforward,
        input_vocab_size, target_vocab_size, pe_input, pe_target,
        dropout_rate
    )

    # Dummy input shapes for encoder and decoder
    dummy_inp = tf.random.uniform((1, 10), dtype=tf.int64, minval=0, maxval=input_vocab_size)
    dummy_tar = tf.random.uniform((1, 10), dtype=tf.int64, minval=0, maxval=target_vocab_size)

    # Build and run a forward pass
    output = transformer_model(
        dummy_inp,
        dummy_tar,
        training=False,
        enc_padding_mask=None,
        look_ahead_mask=None,
        dec_padding_mask=None,
    )

    # Display summary
    transformer_model.summary()
