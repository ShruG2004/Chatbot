import random
import pickle
import json
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout


Lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words=[]
classes=[]
documents=[]
ignore_letter = ["?","!",",","'","."]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tags']))
        if intent['tags'] not in classes:
            classes.append(intent['tags'])

import nltk
nltk.download('wordnet')

words = [Lemmatizer.lemmatize(word) for word in words if word not in ignore_letter]
words = sorted(set(words))
print(words)
classes = sorted(set(classes))
print(classes)

pickle.dump(words, open('words,pkl','wb'))
pickle.dump(classes, open('classes,pkl','wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [Lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag , output_row])

random.shuffle(training)
training = np.array(training , dtype= object)

train_x = np.array(list(training[:,0]))
train_y = np.array(list(training[:,1]))

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, epsilon=1e-6)
model.compile(loss= tf.keras.losses.CategoricalCrossentropy(), optimizer = adam, metrics=['accuracy'])

hist=model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.model', hist)
print("Done")





