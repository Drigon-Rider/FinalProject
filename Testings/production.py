import tensorflow as tf
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer
import numpy as np
from voice import get_start_sequence
from melodypreprocessor import tokens_to_midi
import json

BATCH_SIZE = 32
DATA_PATH = "dataset.json"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 100

if __name__ == "__main__":
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    melody_preprocessor.fit_tokenizer()
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        dropout_rate=0.1,
    )

    # Build model by calling it on dummy input with all required arguments
    dummy_input = np.zeros((1, 10), dtype=np.int32)
    dummy_target = np.zeros((1, 10), dtype=np.int32)
    dummy_mask = np.ones((1, 1, 1, 10), dtype=np.float32)

    transformer_model(
        input=dummy_input,
        target=dummy_target,
        enc_padding_mask=dummy_mask,
        look_ahead_mask=dummy_mask,
        dec_padding_mask=dummy_mask,
        training=False
    )

    transformer_model.load_weights("transformer_weights.h5")
    print("Model weights loaded.")

    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer
    )
    
    with open("start_sequence.json", "r")as f:
        start_sequence = json.load(f)
    
    print("Loaded start_sequence:", start_sequence)
    new_melody = melody_generator.generate(start_sequence)
    print(f"Generated melody: {new_melody}")
    new_melody = ', '.join(new_melody.strip().split())
    with open("generated_melody.json", "w") as outfile:
        json.dump(new_melody, outfile)
    print("Generated melody saved to generated_melody.json")