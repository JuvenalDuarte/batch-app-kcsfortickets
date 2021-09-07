import json
import logging
import pandas as pd
import numpy as np
import pickle
import requests
from pycarol import Carol, Storage
import gc
import re
import ftfy
from unidecode import unidecode

logger = logging.getLogger(__name__)

def transformSentences(m, custom_stopwords, preproc_mode):
    # Ensure the parameter type as string
    mproc0 = str(m)
    
    # Set all messages to a standard encoding
    mproc1 = ftfy.fix_encoding(mproc0)
    
    # Replaces accentuation from chars. Ex.: "férias" becomes "ferias" 
    mproc2 = unidecode(mproc1)
    
    if preproc_mode.lower() in ["advanced", "stopwords"]:
        # Removes special chars from the sentence. Ex.: 
        #  - before: "MP - SIGAEST - MATA330/MATA331 - HELP CTGNOCAD"
        #  - after:  "MP   SIGAEST   MATA330 MATA331   HELP CTGNOCAD"
        mproc3 = re.sub('[^0-9a-zA-Z]', " ", mproc2)

        if preproc_mode.lower() in ["stopwords"]:
        
            # Sets capital to lower case maintaining full upper case tokens and remove portuguese stop words.
            #  - before: "MP   MEU RH   Horario ou Data registrado errado em solicitacoes do MEU RH"
            #  - after:  "MP MEU RH horario data registrado errado solicitacoes MEU RH"
            mproc4 = " ".join([t.lower() for t in mproc3.split() if t not in custom_stopwords])
        
            return mproc4
        else:
            return mproc3

    else:
        return mproc2

def update_embeddings(df, cols_search, cols_keywords, app_name, url, model, cache=True, specificity_percent=0.3, preproc="Basic"):

    # Filling NAs with blank
    logger.info('Replacing NaN with \"\" to avoid crashes on fulfillment.')
    df.fillna('', inplace=True)

    try:
        cols_search_l = [c.lstrip().rstrip() for c in cols_search.split(",")]
    except Exception as e:
        logger.error(f'Unable to parse setting kb_search_fields:{cols_search}.')
        raise f"Stack: {e}"

    dfs = []
    for cs in cols_search_l:

        logger.info(f'Preparing \"{cs}\" for search.')

        # Get the search column all other expected to be retrieved
        dft = df.copy()

        # Assign uniform name sentence for the search column
        dft["sentence"] = dft[cs]
        dft["matched_on"] = cs

        # Adding to the list of "searcheable" data 
        dfs.append(dft)

    try:
        if cols_keywords:
            cols_keywords_l = [c.lstrip().rstrip() for c in cols_keywords.split(",")]
        else:
            cols_keywords_l = []
    except Exception as e:
        logger.error(f'Unable to parse setting kb_keywords_fields:{cols_keywords}.')
        raise f"Stack: {e}"

    for cs in cols_keywords_l:
        logger.info(f'Preparing \"{cs}\" for search.')

        # Get the search column all other expected to be retrieved
        dft = df.copy()

        # Parsing tags
        dft['tags'] = dft[cs].apply(get_tags)
        dft['tags_sentence'] = dft['tags'].apply(get_tags_sentence)

        # Denormalizing: generating one line per tag
        dft = dft.explode('tags_sentence')

        # Filtering out obvious tags (happens so often that doesn't helps differentiating)
        threshold = specificity_percent
        tags_frequencies = dft.groupby("tags_sentence")["id"].nunique().sort_values(ascending=False).reset_index()
        tags_frequencies = tags_frequencies[tags_frequencies["id"] > threshold]
        poor_tag = list(tags_frequencies["tags_sentence"].values)
        dft = dft[~dft["tags_sentence"].isin(poor_tag)].copy()

        # Assign uniform name sentence for the search column
        dft['sentence'] = dft['tags_sentence']

        # Removing unnecessary columns
        dft.drop(columns=['tags', 'tags_sentence'], inplace=True)
        
        # All the search fields are mapped to "sentece". Here we store the source 
        # field for debug purposes, so that we know which field in the article
        # lead to the match to the search.
        dft["matched_on"] = cs

        # Adding to the list of "searcheable" data 
        dfs.append(dft)

    # Concatenating all searcheable data into a single dataset
    logger.info(f'Packing all searchable fields.')
    df = pd.concat(dfs, ignore_index=True)

    # Applying preprocessing
    logger.info(f'Applying preproc mode: {preproc}.')
    # Reading stopwords to be removed
    if preproc == "Advanced":
        with open('/app/cfg/stopwords.txt') as f:
            custom_stopwords = f.read().splitlines()
    else:
        custom_stopwords = []
    df["sentence"] = df["sentence"].apply(lambda x: transformSentences(x, custom_stopwords, preproc_mode=preproc))
    
    logger.info(f'Cleaning variables.')
    # releasing memory from the full DF
    del dfs
    gc.collect()
    df.dropna(subset=['sentence'], inplace=True)

    # Generating Embeddings
    logger.info('Creating embeddings')
    # Efficience improvement: As there are many repetitions, generate a mapping [sentence] -> [embedding], 
    # such that the same message doesn't have the embedding calculated twice. 
    sentences_unique = df["sentence"].unique()
    sent2embd = getEmbeddingsCache(uniq_sentences=sentences_unique, model=model, cache=cache)

    logger.info('Translating sentences to embeddings')
    # Translate the senteces to the corresponding embeddings
    df["sentence_embedding"] = df["sentence"].apply(lambda x: sent2embd[x])

    ### Save objects in Storage
    if app_name:
        login = Carol()
        login.app_name = app_name
        stg = Storage(login)

        logger.info('Saving embeddings in Carol Storage, on {login.app_name} app.')
        save_object_to_storage(stg, df, 'kcs4tickets')

    if url:
        logger.info('Refreshing online app.')
        response = requests.get(url)

        if not response.ok:
            raise ValueError('Error updating embeddings on Carol App.')

    return 'done'

def getEmbeddingsCache(uniq_sentences, model, cache=True):
    login = Carol()
    storage = Storage(login)
    
    # Loads the dictionary containing all sentences whose embeddings are already calculated.
    filename = "embeddings_cache.pickle"
    if cache:
        logger.info('Loading cache from storage.')
        sent2embd = get_file_from_storage(storage, filename)

        if sent2embd is None:
            logger.warn('Unable to load file from storage. Reseting cache.')
            sent2embd = {}            

    else:
        # WARN: Full embeddings calculation demands high resources consumption.
        # Make sure the VM instance defined on manifest.json is big enough
        # before running on reset mode.
        logger.warn('Reseting cache.')
        sent2embd = {}
    
    # Gets the list of sentences for which embeddings are not yet calculated
    sentences_processed = list(sent2embd.keys())
    sentences_pending = list(np.setdiff1d(uniq_sentences,sentences_processed))

    if len(sentences_pending) > 0:

        logger.info(f'Calculating embeddings for {len(sentences_pending)} unprocessed sentences.')
        embeddings_pending = model.encode(sentences_pending, convert_to_tensor=False)
        dict_pending = dict(zip(sentences_pending, embeddings_pending))
        
        logger.info('Updating cache on storage.')
        sent2embd.update(dict_pending)
        save_object_to_storage(storage, sent2embd, filename)
    
    else:
        logger.info('All sentences already present on cache, no calculation needed.')
    
    return sent2embd

def save_object_to_storage(storage, obj, filename):
    logger.info(f'Saving {filename} to the storage.')

    with open(filename, "bw") as f:
        pickle.dump(obj, f)

    storage.save(filename, obj, format='pickle')

def get_file_from_storage(storage, filename):
    return storage.load(filename, format='pickle', cache=False)

def get_tags(label_names):
    if label_names:
        return json.loads(str(label_names))
    return list()

def get_tags_sentence(tags):
    tags_tmp = []
    for tag in tags:
        tag_arr = tag.split('_')
        if len(tag_arr) > 1 and 'versao' not in tag_arr:
            tag_sentence = ' '.join(tag_arr)
            tag_sentence = tag_sentence.replace('#', '')
            tags_tmp.append(tag_sentence)

    # Make sure to avoid repeated tags on the same article
    tags_tmp = list(set(tags_tmp))
    return tags_tmp