import pandas as pd
import nltk,string
from gensim import corpora

data = pd.read_csv("news_cnn.csv",header=None, usecols=[1,2],names=["sentiment","review"])
data.head()
len(data)
#data['sentiment'].replace( [1,2],[0,1],inplace=True)


# if your computer does not have enough resource
# reduce the dataset
#data=data.loc[0:8000]

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
 
import numpy as np

# set the maximum number of words to be used
MAX_NB_WORDS=1330

# set sentence/document length
MAX_DOC_LEN=200

# get a Keras tokenizer
# https://keras.io/preprocessing/text/
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data["review"])

# get the mapping between word and its index
voc=tokenizer.word_index

# convert each document to a list of word index as a sequence
sequences = tokenizer.texts_to_sequences(data["review"])

# pad all sequences into the same length 
# if a sentence is longer than maxlen, pad it in the right
# if a sentence is shorter than maxlen, truncate it in the right
padded_sequences = pad_sequences(sequences, \
                                 maxlen=MAX_DOC_LEN, \
                                 padding='post', \
                                 truncating='post')


# Exercise 5.3: Create CNN model

from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, \
Dropout, Activation, Input, Flatten, Concatenate
from keras.models import Model

# The dimension for embedding
EMBEDDING_DIM=100

# define input layer, where a sentence represented as
# 1 dimension array with integers
main_input = Input(shape=(MAX_DOC_LEN,), dtype='int32', name='main_input')

# define the embedding layer
# input_dim is the size of all words +1
# where 1 is for the padding symbol
# output_dim is the word vector dimension
# input_length is the max. length of a document
# input to embedding layer is the "main_input" layer
embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, output_dim=EMBEDDING_DIM, \
                    input_length=MAX_DOC_LEN,name='embedding')(main_input)


# define 1D convolution layer
# 64 filters are used
# a filter slides through each word (kernel_size=1)
# input to this layer is the embedding layer
conv1d_1= Conv1D(filters=64, kernel_size=1, \
                 name='conv_unigram', activation='relu')(embed_1)

# define a 1-dimension MaxPooling 
# to take the output of the previous convolution layer
# the convolution layer produce 
# MAX_DOC_LEN-1+1 values as ouput (???)
pool_1 = MaxPooling1D(MAX_DOC_LEN-1+1, name='pool_unigram')(conv1d_1)

# The pooling layer creates output 
# in the size of (# of sample, 1, 64)  
# remove one dimension since the size is 1
flat_1 = Flatten(name='flat_unigram')(pool_1)

# following the same logic to define 
# filters for bigram
conv1d_2= Conv1D(filters=64, kernel_size=2, \
                 name='conv_bigram',activation='relu')(embed_1)
pool_2 = MaxPooling1D(MAX_DOC_LEN-2+1, name='pool_bigram')(conv1d_2)
flat_2 = Flatten(name='flat_bigram')(pool_2)

# filters for trigram
conv1d_3= Conv1D(filters=64, kernel_size=3, \
                 name='conv_trigram',activation='relu')(embed_1)
pool_3 = MaxPooling1D(MAX_DOC_LEN-3+1, name='pool_trigram')(conv1d_3)
flat_3 = Flatten(name='flat_trigram')(pool_3)

# Concatenate flattened output
z=Concatenate(name='concate')([flat_1, flat_2, flat_3])

# Create a dropout layer
# In each iteration only 50% units are turned on
drop_1=Dropout(rate=0.5, name='dropout')(z)

# Create a dense layer
dense_1 = Dense(192, activation='relu', name='dense')(drop_1)
# Create the output layer
preds = Dense(1, activation='sigmoid', name='output')(dense_1)

# create the model with input layer
# and the output layer
model = Model(inputs=main_input, outputs=preds)


model.summary()
#model.get_config()
#model.get_weights()

 

model.compile(loss="binary_crossentropy", \
              optimizer="adam", \
              metrics=["accuracy"])

BATCH_SIZE = 64
NUM_EPOCHES = 10

# fit the model and save fitting history to "training"
training=model.fit(padded_sequences, data['sentiment'], \
                   batch_size=BATCH_SIZE, \
                   epochs=NUM_EPOCHES,\
                   validation_split=0.3, verbose=2)

from keras.callbacks import EarlyStopping, ModelCheckpoint

# the file path to save best model
BEST_MODEL_FILEPATH="best_model"

# define early stopping based on validation loss
# if validation loss is not improved in 
# an iteration compared with the previous one, 
# stop training (i.e. patience=0). 
# mode='min' indicate the loss needs to decrease 
earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')

# define checkpoint to save best model
# which has max. validation acc
checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, \
                             monitor='val_acc', \
                             verbose=2, \
                             save_best_only=True, \
                             mode='max')

# compile model
model.compile(loss="binary_crossentropy", \
              optimizer="adam", metrics=["accuracy"])

# fit the model with earlystopping and checkpoint
# as callbacks (functions that are executed as soon as 
# an asynchronous thread is completed)
model.fit(padded_sequences, data['sentiment'], \
          batch_size=BATCH_SIZE, epochs=NUM_EPOCHES, \
          callbacks=[earlyStopping, checkpoint],
          validation_split=0.3, verbose=2)



# load the model using the save file
model.load_weights("best_model")

# evaluate the model
scores = model.evaluate(padded_sequences, data['sentiment'], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))