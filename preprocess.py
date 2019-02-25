import pickle
import zipfile
import json 
import re
from collections import defaultdict
from collections import Counter
import numpy as np
# special tokens

# load pickle file    
def read_pickle(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)

### json file sample:
''' 
{
'info':
{'description': 'This is stable 1.0 version of the 2014 MS COCO dataset.',
 'url': 'http://mscoco.org', 'version': '1.0', 'year': 2014, 
 'contributor': 'Microsoft COCO group', 'date_created': '2015-01-27 09:11:52.357475'},

'images':
{'license': 5, 'file_name': 'COCO_train2014_000000384029.jpg', 'coco_url': 'http://mscoco.org/images/384029', 
'height': 429, 'width': 640, 'date_captured': '2013-11-14 16:29:45', 
'flickr_url': 'http://farm3.staticflickr.com/2422/3577229611_3a3235458a_z.jpg', 'id': 384029},
..., {}, 

'licenses': 
{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}, 
..., {}, 

'annotations':
{'image_id': 116100, 'id': 67, 'caption': 'A panoramic view of a kitchen and all of its appliances.'},
..., {}
}
'''



def get_captions_for_fns(fns, zip_fn, zip_json_path):
    '''
    Extract captions from zip file, return a list of captions lists. Given
    '''
    zf = zipfile.ZipFile(zip_fn)
    json_file = json.loads(zf.read(zip_json_path).decode("utf8"))
    id_to_fn = {img["id"]: img["file_name"] for img in json_file["images"]}
    fn_to_caps = defaultdict(list)
    
    for cap in json_file['annotations']:
        fn_to_caps[ id_to_fn[cap['image_id']] ].append(cap['caption'])

    fn_to_caps = dict(fn_to_caps)
    return list(map(lambda x: fn_to_caps[x], fns))




# split sentence into tokens (split into lowercased words)
def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))


def generate_vocabulary(train_captions):
    """
    Return {token: index} for all train tokens (words) that occur 5 times or more, 
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    Use `split_sentence` function to split sentence into tokens.
    Also, add PAD (for batch padding), UNK (unknown, out of vocabulary), 
        START (start of sentence) and END (end of sentence) tokens into the vocabulary.
    """
    

    PAD = "#PAD#"
    UNK = "#UNK#"
    START = "#START#"
    END = "#END#"
    
    sentence_list = [sentence for captions in train_captions for sentence in captions]
    word_list = split_sentence(' '.join(sentence_list))   # big list with duplicates
    
    vocab = [key for key, value in Counter(word_list).items() if value >= 5] # 
    vocab += [PAD, UNK, START, END]

    return {token: index for index, token in enumerate(sorted(vocab))}


  
def caption_tokens_to_indices(captions, vocab):
    """
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    """
    PAD = "#PAD#"
    UNK = "#UNK#"
    START = "#START#"
    END = "#END#"
    
    def replace_words(w, vocab):
        if w in vocab: 
            return vocab[w]
        else:
            return vocab[UNK]
  
  
    res = []
    

    
    res = [[ [vocab[START]] + [ replace_words(x, vocab) for x in split_sentence(sentence) ] + [vocab[END]] for sentence in cap]  for cap in captions]
    
    
    return res



# we will use this during training
def batch_captions_to_matrix(batch_captions, pad_idx, max_len = None):
    """
    `batch_captions` is an array of arrays:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]
    Put vocabulary indexed captions into np.array of shape (len(batch_captions), columns),
        where "columns" is max(map(len, batch_captions)) when max_len is None
        and "columns" = min(max_len, max(map(len, batch_captions))) otherwise.
    Add padding with pad_idx where necessary.
    Input example: [[1, 2, 3], [4, 5]]
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None
    Output example: np.array([[1, 2], [4, 5]]) if max_len=2
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100
    Try to use numpy, we need this function to be fast!
    """
    
    
    
    if not max_len:
        max_len = max( map( lambda x: len(x), batch_captions ) )
    else:
        max_len = min( max_len,  max( map(lambda x: len(x), batch_captions) ) )
        
    matrix = [[index for i, index in enumerate(caption[:max_len])] + [pad_idx]*(max(max_len-len(caption),0)) for caption in batch_captions]
    
    return np.array(matrix)   




def batch_captions_to_matrix(batch_captions, pad_idx, max_len=None):
    """
    `batch_captions`:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]
    
    Add padding with pad_idx where necessary.
    Input example: [[1, 2, 3], [4, 5]]
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None
    Output example: np.array([[1, 2], [4, 5]]) if max_len=2
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100
    """
    
    
    
    if not max_len:
        max_len = max( map( lambda x: len(x), batch_captions ) )
    else:
        max_len = min( max_len,  max( map(lambda x: len(x), batch_captions) ) )
        
    matrix = [ [index for i, index in enumerate(caption[:max_len])] 
              + [pad_idx]*(max(max_len-len(caption),0)) for caption in batch_captions ]
    
    return np.array(matrix)    


