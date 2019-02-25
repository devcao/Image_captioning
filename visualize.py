import sys
import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
L = keras.layers
K = keras.backend

import matplotlib.pyplot as plt


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


PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"

def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input   # what is this line -->  preprocess the input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model

def image_center_crop(img):
    h, w = img.shape[0], img.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2
    return img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]

def decode_image_from_buf(buf):
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def crop_and_preprocess(img, input_shape, preprocess_for_model):
    img = image_center_crop(img)  # take center crop
    img = cv2.resize(img, input_shape)  # resize for our model
    img = img.astype("float32")  # prepare for normalization
    img = preprocess_for_model(img)  # preprocess for model
    return img

# look at validation prediction example
def apply_model_to_image_raw_bytes(raw):
    img = decode_image_from_buf(raw)
    fig = plt.figure(figsize=(7, 7))
    plt.grid('off')
    plt.axis('off')
    plt.imshow(img)
    img = crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), final_model.preprocess_for_model)
    
    caps = ' '.join(generate_caption(img)[1:-1])
    plt.title(caps)
    print(caps)

    plt.show()

def show_valid_example(val_img_fns, example_idx=0):
    zf = zipfile.ZipFile("val2014_sample.zip")
    all_files = set(val_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    apply_model_to_image_raw_bytes(zf.read(example))
    


# this is an actual prediction loop
def generate_caption(image, t=1, sample=False, max_len=20):
    """
    Generate caption for given image.
    if `sample` is True, we will sample next token from predicted probability distribution.
    `t` is a temperature during that sampling,
        higher `t` causes more uniform-like distribution = more chaos.
    """
    # condition lstm on the image
    s.run(final_model.init_lstm, 
          {final_model.input_images: [image]})
    
    # current caption
    # start with only START token
    caption = [vocab[START]]
    
    for _ in range(max_len):
        next_word_probs = s.run(final_model.one_step, 
                                {final_model.current_word: [caption[-1]]})[0]
        next_word_probs = next_word_probs.ravel()
        
        # apply temperature
        next_word_probs = next_word_probs**(1/t) / np.sum(next_word_probs**(1/t))

        if sample:
            next_word = np.random.choice(range(len(vocab)), p=next_word_probs)
        else:
            next_word = np.argmax(next_word_probs)

        caption.append(next_word)
        if next_word == vocab[END]:
            break
       
    return list(map(vocab_inverse.get, caption))


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






    IMG_SIZE = 299
    IMG_EMBED_SIZE = 2048
    IMG_EMBED_BOTTLENECK = 120
    WORD_EMBED_SIZE = 100
    LSTM_UNITS = 300
    LOGIT_BOTTLENECK = 120
    pad_idx = 1

    s = tf.InteractiveSession()
    new_saver = tf.train.import_meta_graph('weights.meta')
    new_saver.restore(s, tf.train.latest_checkpoint('./'))

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

    class final_model:
        # CNN encoder
        encoder, preprocess_for_model = get_cnn_encoder()
        #tf.train.Saver().restore(s, os.path.abspath("weights"))  

        # containers for current lstm state
        lstm_c = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="cell")
        lstm_h = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="hidden")

        # input images
        input_images = tf.placeholder('float32', [1, IMG_SIZE, IMG_SIZE, 3], name='images')

        # get image embeddings
        img_embeds = encoder(input_images)

        # initialize lstm state conditioned on image
        init_c = init_h = decoder.img_embed_bottleneck_to_h0(decoder.img_embed_to_bottleneck(img_embeds))
        init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)
    
        # current word index
        current_word = tf.placeholder('int32', [1], name='current_input')

        # embedding for current word
        word_embed = decoder.word_embed(current_word)

        # apply lstm cell, get new lstm states
        new_c, new_h = decoder.lstm(word_embed, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]

        # compute logits for next token
        new_logits = decoder.token_logits(decoder.token_logits_bottleneck(new_h))
        # compute probabilities for next token
        new_probs = tf.nn.softmax(new_logits)

        # `one_step` outputs probabilities of next token and updates lstm hidden state
        one_step = new_probs, tf.assign(lstm_c, new_c), tf.assign(lstm_h, new_h)

    
    for idx in np.random.choice(range(len(zipfile.ZipFile("val2014_sample.zip").filelist) - 1), 10):
        show_valid_example(val_img_fns, example_idx=idx)
        time.sleep(1)