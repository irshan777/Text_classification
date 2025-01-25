import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# Add some data
texts = ["I love coding", "Machine learning is fascinating", 
         "Natural language processing is cool", 
         "I enjoy working with neural networks", 
         "This is a text classification example"]
labels = [1, 0, 0, 1, 1] # 1 for positive, 0 for negative

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Sequence to uniform length
maxlen = 10  # adjust according to your data
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, 
                                                    labels, test_size=0.2, random_state=42)

# Build your model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=64,
        input_length=maxlen
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
print(model.summary())

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16,
                    validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)

# Print test accuracy
print("Test Accuracy:", test_accuracy)
