import sys
import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
L = keras.layers
K = keras.backend

import time
import zipfile
import json
from collections import defaultdict
import re
import random
from random import choice
import pickle
import os
import cv2


import preprocess

# generate batch via random sampling of images and captions for them,
# we use `max_len` parameter to control the length of the captions (truncating long captions)
def generate_batch(images_embeddings, indexed_captions, batch_size, max_len = None):

    batch_idx = np.random.choice(range(images_embeddings.shape[0]), batch_size, replace = False)
    batch_image_embeddings = images_embeddings[batch_idx,:]
    
    batch_captions_sentence = [ caption[np.random.randint(5)] for caption in indexed_captions[batch_idx] ]
    batch_captions_matrix = preprocess.batch_captions_to_matrix(batch_captions_sentence, pad_idx, max_len = max_len)
    
    return {decoder.img_embeds: batch_image_embeddings, 
            decoder.sentences: batch_captions_matrix}



if __name__ == "__main__":
    

    ## load train/valid_img_file_names and pretrained weights of imgs
    train_img_embeds = preprocess.read_pickle("train_img_embeds.pickle")
    train_img_fns = preprocess.read_pickle("train_img_fns.pickle") 
    val_img_embeds = preprocess.read_pickle("val_img_embeds.pickle")
    val_img_fns = preprocess.read_pickle("val_img_fns.pickle")

    ## Create train/valid_caption lists
    train_captions = preprocess.get_captions_for_fns(train_img_fns, "captions_train-val2014.zip", "annotations/captions_train2014.json")
    val_captions = preprocess.get_captions_for_fns(val_img_fns, "captions_train-val2014.zip", "annotations/captions_val2014.json")



    assert( len(train_img_fns) == len(train_captions) )
    assert( len(val_img_fns) ==  len(val_captions) )

    ## build a vocab
    vocab = preprocess.generate_vocabulary(train_captions)
    vocab_inverse = {idx: w for w, idx in vocab.items()}

    print( "Size of the dictionary: {}".format(len(vocab)) )

    # replace tokens with indices
    train_captions_indexed = preprocess.caption_tokens_to_indices(train_captions, vocab)
    val_captions_indexed = preprocess.caption_tokens_to_indices(val_captions, vocab)

    train_captions_indexed = np.array(train_captions_indexed)
    val_captions_indexed = np.array(val_captions_indexed)

    IMG_EMBED_SIZE = train_img_embeds.shape[1]
    IMG_EMBED_BOTTLENECK = 120
    WORD_EMBED_SIZE = 100
    LSTM_UNITS = 300
    LOGIT_BOTTLENECK = 120
    pad_idx = vocab['#PAD#']


    s = tf.InteractiveSession()
    tf.set_random_seed(42)


    class decoder:
        # [batch_size, IMG_EMBED_SIZE] of CNN image features
        img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE])
    
        # [batch_size, time steps] of word ids     Batch  *  time_ steps
        sentences = tf.placeholder('int32', [None, None])
    
        # we use bottleneck here to reduce the number of parameters
        # image embedding -> bottleneck
        img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK,    # 120 
                                      input_shape=(None, IMG_EMBED_SIZE), 
                                      activation='elu')
        # image embedding bottleneck -> lstm initial state
        img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
                                         activation='elu')
        # word -> embedding
        word_embed = L.Embedding(len(vocab), WORD_EMBED_SIZE)
        # lstm cell (from tensorflow)
        lstm = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)
    
        # we use bottleneck here to reduce model complexity
        # lstm output -> logits bottleneck
        token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK,   # 120
                                      input_shape=(None, LSTM_UNITS),
                                      activation="elu")
        # logits bottleneck -> logits for next token prediction
        token_logits = L.Dense(len(vocab),
                           input_shape=(None, LOGIT_BOTTLENECK))
    
        ########## Coding starts ##########

        # initial lstm cell state of shape (None, LSTM_UNITS),
        # we need to condition it on `img_embeds` placeholder.
        c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds))    # 2048 features to 120 Dense to 300 Dense

      
        word_embeds = word_embed(sentences[:,:-1])   
    
  
        hidden_states, _ = tf.nn.dynamic_rnn(lstm, word_embeds,
                                         initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))


        flat_hidden_states = tf.reshape(hidden_states, [-1, LSTM_UNITS])
    
        # Calculate logits for next tokens using `token_logits_bottleneck` and `token_logits` layers
        flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states))  # flattened -> 120 Dense -> len(vocab) Dense
    
        flat_ground_truth = tf.reshape(sentences[:,1:],[-1,])    # sentences but the first one ? 
    
        flat_loss_mask = tf.not_equal(flat_ground_truth, pad_idx)
    
        # compute cross-entropy between `flat_ground_truth` and `flat_token_logits` predicted by lstm
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=flat_ground_truth, 
            logits=flat_token_logits
        )

       
        loss = tf.reduce_mean(tf.boolean_mask(xent, flat_loss_mask))


    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_step = optimizer.minimize(decoder.loss)
    
    saver = tf.train.Saver()

    s.run(tf.global_variables_initializer())


    batch_size = 64
    n_epochs = 20
    n_batches_per_epoch = 1000
    n_validation_batches = 100  

    MAX_LEN = 20  # truncate long captions to speed up training

    np.random.seed(42)
    random.seed(42)

    for epoch in range(n_epochs):
    
        train_loss = 0
   
        counter = 0
        for iters in range(n_batches_per_epoch):
            train_loss += s.run([decoder.loss, train_step], 
                            generate_batch(train_img_embeds, 
                                           train_captions_indexed, 
                                           batch_size, 
                                           MAX_LEN))[0]
            counter += 1
            print('Batch: {}, Epoch: {}'.format(iters, epoch))
           
        
        train_loss /= n_batches_per_epoch
    
        val_loss = 0
        for _ in range(n_validation_batches):
            val_loss += s.run(decoder.loss, generate_batch(val_img_embeds,
                                                       val_captions_indexed, 
                                                       batch_size, 
                                                       MAX_LEN))
        val_loss /= n_validation_batches
    
        print('Epoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss))

        
        saver.save(s, os.path.abspath("weights_{}".format(epoch)))
    
    print("Finished!")

    saver.save(s, os.path.abspath("weights"))