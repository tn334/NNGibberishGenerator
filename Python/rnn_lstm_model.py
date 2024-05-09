import numpy as np
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
import random, nltk, pronouncing
from nltk.corpus import words
import time
import matplotlib.pyplot as plt

# ****************************************************************
# *      FUNCTIONS TO USE NETWORK SEED AND PRODUCE A WORD        *
# ****************************************************************

# The temperature parameter in the sample function controls the randomness of the predictions. 
# A higher temperature will result in more random predictions, while a lower temperature will result in less random predictions. 
# You can adjust this parameter to control the “creativity” of the model.
def sample(preds : int, temperature=1.1) -> int:
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_word(length: int) -> str:
    start_index = random.randint(0, len(character_sequences) - 1)
    sentence = character_sequences[start_index]
    generated = ''

    for _ in range(length):
        x_pred = np.zeros((1, sequence_length, len(vocab_chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + [next_char]

        if len(generated) == length:
            break

    return generated

# callback class to produce new word after each epoch
class GenerateWordCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        print(f"\nEpoch {epoch + 1} finished in {epoch_duration:.2f} seconds. Generated word: {generate_word(5)}\n")

    def on_train_end(self, logs=None):
        print(f"\nTraining complete. Final generated word: {generate_word(5)}\n")
    
# Download the word corpus from NLTK
nltk.download('words')
english_words = words.words()

# define length of sequence
sequence_length = 10

# ****************************************************************
# *                TRAIN USING LETTER SEQUENCES                  *
# ****************************************************************

# create character list of words
tokenized_words = [[char for char in word] for word in english_words]

# create set of vocab characters
vocab_chars = set(''.join(english_words))

# Create a mapping of unique characters to integers
char_to_index = { char : idx for idx, char in enumerate(vocab_chars)}
index_to_char = { idx : char for char, idx in char_to_index.items()}

# create sequences and next character in sequence
character_sequences = []
next_chars = []

for word in tokenized_words:
    for idx in range(len(word) - sequence_length):
        character_sequences.append(word[idx : idx + sequence_length])
        next_chars.append(word[idx + sequence_length])

# vectorize character sequences
X_char = np.zeros((len(character_sequences), sequence_length, len(vocab_chars)), dtype=np.bool_)
y_char = np.zeros((len(character_sequences), len(vocab_chars)), dtype=np.bool_)

for i, sequence in enumerate(character_sequences):
    for t, char in enumerate(sequence):
        X_char[i, t, char_to_index[char]] = 1
    y_char[i, char_to_index[next_chars[i]]] = 1

# ****************************************************************
# *        VALIDATE WORD GENERATION USING PHONETICS              *
# ****************************************************************

# create phonetic representations of words
phonetic_reps = [pronouncing.phones_for_word(word) for word in english_words]

# create a set of vocab phonemes
vocab_phonemes = set()
for sublist in phonetic_reps:
    for items in sublist:
        for phoneme in items.split():
            vocab_phonemes.add(phoneme)


# create a mapping  of unique phonemes to integers
phoneme_to_index = {phoneme : idx for idx, phoneme in enumerate(vocab_phonemes)}
index_to_phoneme = {idx : phoneme for phoneme, idx in phoneme_to_index.items()}

# create sequences and next phoneme in sequence
phonetic_sequences = []
next_phonemes = []

for phonetic_rep in phonetic_reps:
    for phonetic_word in phonetic_rep:
        phonemes = phonetic_word.split()
        for idx in range(len(phonemes) - sequence_length):
            phonetic_sequences.append(phonemes[idx : idx + sequence_length])
            next_phonemes.append(phonemes[idx + sequence_length])

# vectorize phonetic sequences
X_phonetic = np.zeros((len(phonetic_sequences), sequence_length, len(vocab_phonemes)), dtype=np.bool_)
y_phonetic = np.zeros((len(phonetic_sequences), len(vocab_phonemes)), dtype=np.bool_)

for i, sequence in enumerate(phonetic_sequences):
    for t, phoneme in enumerate(sequence):
        X_phonetic[i, t, phoneme_to_index[phoneme]] = 1
    y_phonetic[i, phoneme_to_index[next_phonemes[i]]] = 1

# ****************************************************************
# *                DEFINE AND TRAIN THE MODEL                    *
# ****************************************************************

# define LSTM Model
model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, len(vocab_chars))))
model.add(Dense(len(vocab_chars), activation='softmax'))

# modify lr to make NN smarter or stupider
optimizer = RMSprop(learning_rate=0.03)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# train model with hyperparameters
batch_size = 120
epochs = 10
model.fit(X_char, y_char, batch_size=batch_size, epochs=epochs, callbacks=[GenerateWordCallback()])

