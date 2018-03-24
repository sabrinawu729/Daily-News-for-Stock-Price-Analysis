import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
MAX_NB_WORDS=4700

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


# total number of words
total_nb_words=len(tokenizer.word_counts)
print(total_nb_words)

# put word and its counts into a data frame
word_counts=pd.DataFrame(\
            tokenizer.word_counts.items(), \
            columns=['word','count'])
word_counts.head(3)

# get histogram of word counts
# after reset index, "index" column 
# is the word frequency
# "count" column gives how many words appear at 
# a specific frequency
df=word_counts['count'].value_counts().reset_index()
df.head(3)

# convert absolute counts to precentage
df['percent']=df['count']/len(tokenizer.word_counts)
# get cumulative percentage
df['cumsum']=df['percent'].cumsum()
df.head(5)

# plot the chart
# chart shows >90% words appear in less than 50 times
# if you like to include only words occur more than 50 times
# then MAX_NB_WORDS = 10% * total_nb_words
plt.bar(df["index"].iloc[0:50], df["percent"].iloc[0:50])
plt.plot(df["index"].iloc[0:50], df['cumsum'].iloc[0:50], c='green')

plt.xlabel('Word Frequency')
plt.ylabel('Percentage')
plt.show()

# create a series based on the length of all sentences
sen_len=pd.Series([len(item) for item in sequences])

# create histogram of sentence length
# the "index" is the sentence length
# "counts" is the count of sentences at a length
df=sen_len.value_counts().reset_index().sort_values(by='index')
df.columns=['index','counts']
df.head(3)

# sort by sentence length
# get percentage and cumulative percentage

df=df.sort_values(by='index')
df['percent']=df['counts']/len(sen_len)
df['cumsum']=df['percent'].cumsum()
df.head(3)

# From the plot, 90% sentences have length<500
# so it makes sense to set MAX_DOC_LEN=4~500 
plt.plot(df["index"], df['cumsum'], c='green')

plt.xlabel('Sentence Length')
plt.ylabel('Percentage')
plt.show()