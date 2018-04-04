#!/usr/bin/env python
import pickle
import math
import random

import numpy as np
import tensorflow as tf

from graph_helpers import *

'''
generateBatch(batch_size, num_context_per_label, context_window, target, step)

batch_size: int, batch size for training
num_context_per_label: int, how many context examples to use per label (the label is the target) 
can't be greater than the context window size
context_window: int, size of the context window 
target: array, the list of targets for each context window
step: int, counter for how many times to step through the same context and target data

Generate the batch data for training. For each "context window", randomly sample a
set of context elements and configure them as training data by constructing column data of,

[target_0, context_0]
[target_0, context_1]
[target_0, context_3]
...
[target_n, context_3]
[target_n, context_2]
[target_n, context_1]

'''
def generateBatch(batch_size, num_context_per_label, context_window, target, step):

    batch = []
    passes_through_batch = batch_size//num_context_per_label
    for window_idx in range(passes_through_batch):
        
        current_window = list(context_window[window_idx + passes_through_batch*step])
        current_window = [i for i in current_window if i!=-1]
        current_target = target[window_idx + passes_through_batch*step]
        context_samples = []
        #while context_samples == -1:
        while len(context_samples)<num_context_per_label:
            context_samples.append(random.choice(current_window))    
#        context_samples = random.sample(current_window, num_context_per_label)
        
        data_samples =  [[context_sample, [current_target]] for context_sample in context_samples]

        for data_sample in data_samples:
            batch.append(data_sample)
            
    return batch

def main():

    max_step = 4# Window size and max_step must be connected
    num_skips = 2 #The number of context examples per label to create x-y data out of 
#i.e. the number of rows of "data" per window, label combo
    window_size = max_step//2 #where max step must be even
    embedding_size = 64  #Dimension of the embedding vector.
    #vocabulary_size = 27623#len(web_graph.nodes())
    vocabulary_size = len(nodes_only)

    num_sampled = 64 #Number of negative examples to sample. 
    #As this number goes to the total number of samples it reproduces softmax, 
    #this not quite correct as we still doing binary classification, except now we give every negative example to test against,
    #as opposed to true multi-class classification
    batch_size = 256 #must be a multiple of num_skips

    print ('%d nodes' % vocabulary_size)

    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Look up embeddings for inputs.
        embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

    avg_loss_record = []
    list_batch_labels = []
    list_batch_inputs = []
    print ('Begin session')
    data_list = ['random_walk_epoch_data_%d.pkl' % i for i in range(7)]
    
    with tf.Session(graph=graph) as session:

        session.run(tf.global_variables_initializer())
        print('Initialized')
        saver = tf.train.Saver()
        #saver.restore(session, 'chkpt/saved_directed_domain_only_weighted_sp500')

        average_loss = 0
        for data in data_list:
            #Shuffle the list of nodes at the start of each epoch
            random_walks = pickle.load(open('walk_data/' + data, 'rb'))
            
            #for epoch in range(len(data)//vocabulary_size):
                
            #random_walks = data[epoch*vocabulary_size:(epoch+1)*vocabulary_size]
            
            random.shuffle(random_walks)
            
            data_windows = np.vectorize(domain_map.get)(random_walks)

            target = data_windows[:,window_size]

            left_window = data_windows[:,:window_size]

            right_window = data_windows[:,window_size+1:]

            context_window = np.concatenate([left_window, right_window], axis=1)
            
            num_steps = len(random_walks)//batch_size
            print ('num steps in data %d' % num_steps)
            for step in range(num_steps-1):

                batch_data = generateBatch(batch_size, num_skips, context_window, target, step)
                batch_inputs = [row[0] for row in batch_data]
                batch_labels = [row[1] for row in batch_data]
               
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                list_batch_labels.append([batch_labels])
                list_batch_inputs.append([batch_inputs])
                
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                
                average_loss += loss_val
             
                if step % 100 == 0: 
                    
                    avg_loss_record.append(float(average_loss)/step)
                    print('num_steps:%d, Average loss:%.7g' % (step, float(average_loss)/step))
                
                if (step % 1000 == 0): 

                    saver.save(session, 'chkpt/saved_directed_domain_only_weighted_sp500_v8_step_%d' % (data*step)+step)

                    print ('Session saved')
                    
            average_loss = 0

        final_embeddings = normalized_embeddings.eval()
        print ('embeddings created')
    
if __name__ == "__main__":
    main()

