import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter

label=pd.read_csv("/Users/Dido/midterm_C4/movement.csv",header=None, usecols=[0,1],names=["date","label"])
df=pd.read_csv("/Users/Dido/midterm_C4/newsclean.csv",header=None)
df['text'] = df[df.columns[1:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
df['date']=df[df.columns[0]]
df=df[['date','text']]
merge=pd.merge(label, df, how='left')
pos =merge[merge['label'] ==1]
neg =merge[merge['label'] ==0]
del pos['date']
del pos['label']
del neg['date']
del neg['label']
pos.to_csv("negative.csv", encoding='utf-8', index = False, header = False)
neg.to_csv("positive.csv", encoding='utf-8', index = False, header = False)
 

def clean_str(string):
    
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()




def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
