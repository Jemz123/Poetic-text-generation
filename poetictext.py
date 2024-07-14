import random
import numpy as np
import tensorflow as tf

# Download and read the Shakespeare dataset
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]

# Get unique characters from the text
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

# Generate sequences and corresponding next characters
seq_length = 40
step_size = 3
sentences = []
next_characters = []
for i in range(0, len(text) - seq_length, step_size):
    sentences.append(text[i:i + seq_length])
    next_characters.append(text[i + seq_length])

# One-hot encode the data
x = np.zeros((len(sentences), seq_length, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Load your pre-trained LSTM model
model = tf.keras.models.load_model('textgenerator.model')

# Define the function to sample from the predicted probabilities
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Define the function to generate text
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - seq_length - 1)
    generated_text = text[start_index:start_index + seq_length]
    generated = generated_text
    
    for i in range(length):
        x_pred = np.zeros((1, seq_length, len(characters)))
        for t, char in enumerate(generated_text):
            x_pred[0, t, char_to_index[char]] = 1.0
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        
        generated += next_char
        generated_text = generated_text[1:] + next_char
    
    return generated

# Example usage to generate text with different temperatures
print('------Temperature: 0.2------')
print(generate_text(300, 0.2))
print('\n------Temperature: 0.4------')
print(generate_text(300, 0.4))
print('\n------Temperature: 0.6------')
print(generate_text(300, 0.6))
print('\n------Temperature: 0.8------')
print(generate_text(300, 0.8))
print('\n------Temperature: 1.0------')
print(generate_text(300, 1.0))
