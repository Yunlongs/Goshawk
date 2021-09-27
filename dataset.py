import tensorflow as tf
from tensorflow import keras
from parse_func import parse_func_file,parse_func_line
import numpy as np
import config
import json
import os
from normalize import normalize_two_files,normalize_dir


def create_id_word_map():
    """
    Create a mapping relation which maps subwords to unique ids and maps ids to subwords.
    :return:word_id_map,id_word_map . dict.
    """
    index = 1
    vocab = {}
    word_id_map = {}
    id_word_map = {}
    with open(config.vocab_path,"r") as f:
        line = f.readline()
        for line in f.readlines():
            k,v = line.strip().split()
            vocab[k] = v
    for word in config.conserved_words:
        word_id_map[word] = index
        id_word_map[index] = word
        index += 1
    for word in vocab.keys():
        word_id_map[word] = index
        id_word_map[index] = word
        index += 1
    return word_id_map,id_word_map

word_id_map,id_word_map = create_id_word_map()
vocab_size = len(word_id_map)

def load_embedding_weight():
    """
    Load the word2vec pretrained subword embeddings matrix.
    :return: weight: subword embedding matrix. shape = [vocab_size,embedding_size]
    """
    from gensim.models import KeyedVectors
    wv = KeyedVectors.load_word2vec_format(config.embedding_path,binary=False)
    post_weight = wv.vectors
    conserve_size = len(config.conserved_words)
    pre_weight = np.zeros((conserve_size +1,config.embedding_size))
    for i in range(conserve_size +1):
        pre_weight[i,i] = 1
    weight = np.concatenate([pre_weight,post_weight],axis=0)
    print(weight.shape)
    return weight

def funcname_to_vector(funcname):
    """
    Map the function prototype to numeric vectors.
    :param funcname: function prototype. string.
    :return:padded numeric vector.
    """
    vector = []
    if len(funcname)<=1:
        print("find NULL funcname! please check and  remove it from your dataset!")
    for word in funcname.split():
        vector.append(word_id_map[word])
    padding = keras.preprocessing.sequence.pad_sequences([np.array(vector)],padding='post',maxlen=config.max_seq_length,truncating='post')
    padding = tf.squeeze(padding,axis=0)
    return padding


def random_choose_k_pairs(k,func_list,funcname,label):
    """
    random choose k samples from func_list. And return the labeled pair.
    :param k: .int.
    :param func_list: functions which to sample. list.
    :param funcname:The anchor function's prototype.string.
    :param label: -1 or 1. -1 represent negative samples, 1 represent positive samples.
    :return:
    """
    funcs = np.random.choice(func_list,k,replace=False)
    pairs = []
    for func in funcs:
        pair = (funcname_to_vector(funcname),funcname_to_vector(func),label)
        pairs.append(pair)
    return pairs

def dataset_merge_alloc(alloc_func,other_func,out_dir):
    """
    Split your alloc functions and negative functions to ['train','valid'] dataset.
    :param alloc_func: Alloc functions. list
    :param other_func: Negative functions.list.
    :param out_dir:which dir to save dataset.dir path.
    :return:
    """
    alloc_funcs = alloc_func.copy()
    others_func = other_func.copy()
    total = len(alloc_funcs) + len(others_func)
    test_num = int(total * config.test_ratio)
    alloc_num = (len(alloc_funcs) / total) * test_num
    alloc_num = int(alloc_num)
    others_num = test_num - alloc_num
    if config.banalace_split:
        alloc_num = 100
        others_num = 100

    valid_funcs = []
    for i in range(alloc_num):
        idx = np.random.randint(low=0,high=len(alloc_funcs)-1)
        func = alloc_funcs.pop(idx)
        valid_funcs.append(func)
    out_dir += os.sep + "alloc"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(out_dir + os.sep + "alloc.valid", "w") as f:
        f.writelines([func + "\n" for func in valid_funcs if len(func) > 1])
    with open(out_dir + os.sep + "alloc.train","w") as f:
        f.writelines([func+"\n" for func in alloc_funcs if len(func)>1])


    valid_neg = []
    for i in range(others_num):
        idx = np.random.randint(low=0,high=len(others_func)-1)
        func =others_func.pop(idx)
        valid_neg.append(func)

    with open(out_dir + os.sep + "neg.valid","w") as f:
        f.writelines([func+"\n" for func in valid_neg])
    with open(out_dir + os.sep + "neg.train","w") as f:
        f.writelines([func+"\n" for func in others_func])

def dataset_merge_free(free_funcs,others_func,out_dir):
    """
    Split your free functions and negative functions to ['train','valid'] dataset.
    :param free_func: Free functions. list
    :param other_func: Negative functions.list.
    :param out_dir:which dir to save dataset.dir path.
    :return:
    """
    free_func = free_funcs.copy()
    other_func = others_func.copy()
    total = len(free_func) + len(other_func)
    test_num = int(total * config.test_ratio)
    free_num= test_num * (len(free_func)/total)
    free_num = int(free_num)
    other_num = test_num - free_num
    if config.banalace_split:
        free_num = 100
        other_num = 100

    valid_funcs = []
    for i in range(free_num):
        idx = np.random.randint(low=0, high=len(free_func)-1)
        func = free_func.pop(idx)
        valid_funcs.append(func)
    out_dir += os.sep + "free"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(out_dir + os.sep + "free.valid", "w") as f:
        f.writelines([func + "\n" for func in valid_funcs if len(func)>1])
    with open(out_dir + os.sep + "free.train", "w") as f:
        f.writelines([func + "\n" for func in free_func if len(func)>1])

    valid_neg = []
    for i in range(other_num):
        idx = np.random.randint(low=0, high=len(other_func)-1)
        func = other_func.pop(idx)
        valid_neg.append(func)


    with open(out_dir + os.sep + "neg.valid", "w") as f:
        f.writelines([func + "\n" for func in valid_neg])
    with open(out_dir + os.sep + "neg.train", "w") as f:
        f.writelines([func + "\n" for func in other_func])

def dataset_merge(alloc_file,free_file,neg_file,out_dir):
    """
    Concat the free and negative functions to construct alloc's negative datasets.
    Concat the alloc and negative functions to construct free's negative datasets.
    将free_file和neg_file中的函数串接在一起，形成alloc的负样本。
    将alloc_file和neg_file中的函数串接在一起，形成free的负样本。
    并各取10%作为验证集。
    :param alloc_file: file that contains only alloc functions. path.
    :param free_file: file that contains only free functions. path.
    :param neg_file: file that not contains any alloc or free functions. path.
    :param out_dir:which dir you want to save result.dir path.
    :return:
    """
    alloc_funcs = []
    free_funcs = []
    neg_funcs = []
    with open(alloc_file,"r") as f:
        for line in f.readlines():
            if len(line)<=1:
                break
            alloc_funcs.append(line.strip())

    with open(free_file,"r") as f:
        for line in f.readlines():
            if len(line) <= 1:
                break
            free_funcs.append(line.strip())

    with open(neg_file,"r") as f:
        for line in f.readlines():
            if len(line)<=1:
                break
            neg_funcs.append(line.strip())
    dataset_merge_alloc(alloc_funcs,free_funcs+neg_funcs,out_dir)
    dataset_merge_free(free_funcs,alloc_funcs+neg_funcs,out_dir)

def dataset_split(out_dir,multi=None):
    """
    Split ['train','valid','test'] datasets.
    :param out_dir: which dir you want to save result. dir path.
    :param multi: for each function,the number of segmented result that you want.int.
    :return:
    """
    alloc_file = "./subword_dataset/kernel_dataset/merge_alloc.json"
    free_file = "./subword_dataset/kernel_dataset/merge_free.json"
    neg_file = "./subword_dataset/kernel_dataset/all_funcs.neg"
    out = "subword_dataset/tmp"
    if multi == None:
        dataset_merge(alloc_file,free_file,neg_file,out)

    alloc_dir = out_dir + os.sep + "alloc"
    if not os.path.exists(alloc_dir):
        os.mkdir(alloc_dir)
    normalize_dir("subword_dataset/tmp/alloc",alloc_dir,multi)

    free_dir = out_dir + os.sep + "free"
    if not os.path.exists(free_dir):
        os.mkdir(free_dir)
    normalize_dir("subword_dataset/tmp/free",free_dir,multi)


def remove_hardest_samples(hardest_mask,semi_hard_mask,func_neg):
    """
    Return the negative functions corresponding to the hardest_negative_mask and semi_hard_negative_mask.
    :param hardest_mask: True if is hardest negative sample.bool list.
    :param semi_hard_mask:True if is semi hard negative sample.bool list.
    :param func_neg:all negative functions.list.
    :return:hardest negative functions. semi hard negative functions.
    """
    neg_res = []
    semi_hard_neg = []
    for i,mask in enumerate(hardest_mask):
        if mask == False:
            neg_res.append(func_neg[i])
        if semi_hard_mask[i] == True:
            semi_hard_neg.append(func_neg[i])
    return neg_res,semi_hard_neg


def generate_func_pair(type = b"target", trainning=True):
    """
    This function generate func's dataset.
    positive : negative = 1 : 1
    :return:
    """
    assert type == b"target","data file type error!"
    if trainning == True:
        suffix = ".train"
    else:
        suffix = ".test"
    type = type.decode()
    with open(config.train_data_prefix + os.sep + type + suffix) as f:
        func_true = []
        for line in f.readlines():
            if len(line)<=1:
                break
            func_true.append(line.strip().lower())

    with open(config.train_data_prefix + os.sep + "neg" + suffix) as f:
        func_neg = []
        for line in f.readlines():
            if len(line)<=1:
                break
            func_neg.append(line.strip().lower())

    if trainning==True:
        k = config.k
    else:
        k = config.test_k
    for func in func_true:
        tp_funcs = random_choose_k_pairs(k,func_true,func,1)
        neg_funcs = random_choose_k_pairs(k,func_neg,func,-1)
        func_pairs = tp_funcs + neg_funcs
        np.random.shuffle(func_pairs)
        for pair in func_pairs:
            yield pair

def generate_resample_pair(type,semi_hard_positive_mask,hardest_negative_mask,semi_hard_negative_mask,training=True):
    """
    This function generate func's dataset.We use had train method to generate negative samples.
    About the hard train negative samples, positive : random negative : semi hard negative = 1 : 1 : 1
    :param type: 'alloc' or 'free'. string.
    :param semi_hard_positive_mask: True if is semi hard positive sample. bool list.
    :param hardest_negative_mask: True if is too hard negative sample. bool list.
    :param semi_hard_negative_mask: True if is semi hard negative sample. bool list.
    :return:
    """
    assert type == b"target","data file type error!"
    if training==True:
        suffix = ".train"
    else:
        suffix = ".test"
    type = type.decode()
    with open(config.train_data_prefix + os.sep + type + suffix) as f:
        func_true = []
        for line in f.readlines():
            if len(line)<=1:
                break
            func_true.append(line.strip())

    with open(config.train_data_prefix + os.sep + "neg" + suffix) as f:
        func_neg = []
        for line in f.readlines():
            if len(line)<=1:
                break
            func_neg.append(line.strip().lower())
    func_neg,func_semi_hard_neg = remove_hardest_samples(hardest_negative_mask,semi_hard_negative_mask,func_neg)
    #tmp = []
    #with open(config.train_data_prefix + os.sep + "hard_" + type) as f:
    #    for line in f.readlines():
    #        if len(line)<=1:
    #            break
    #        tmp.append(line.strip().lower())
    #func_semi_hard_neg += tmp
    if training==True:
        k = config.k
    else:
        k = config.test_k
    for i,func in enumerate(func_true):
        if semi_hard_positive_mask[i] == True:
            tp_funcs = random_choose_k_pairs(2 * k, func_true, func, 1)
            neg_funcs = random_choose_k_pairs(2 * k, func_neg, func, -1)
        else:
            tp_funcs = random_choose_k_pairs(k,func_true,func,1)
            neg_funcs = random_choose_k_pairs(k,func_neg,func,-1)
        tmp = min(k,len(func_semi_hard_neg))
        semi_hard_neg = random_choose_k_pairs(tmp,func_semi_hard_neg,func,-1)
        func_pairs = tp_funcs + neg_funcs + semi_hard_neg
        np.random.shuffle(func_pairs)
        for pair in func_pairs:
            yield pair

def batch_extract_embedding(model,input,batch_size):
    """
    Due to gpu memory limit,we need to batch the input functions to predict their embeddings.
    :param model:
    :param input: input functions. list.
    :param batch_size: .int.
    :return:
    """
    input_num = input.shape[0]
    embedding = []
    for i in range(input_num // batch_size):
        batch = input[batch_size * i : batch_size*(i+1),:]
        out = model((batch,batch),training=False)
        embedding.append(out)
    if input_num%batch_size != 0:
        batch = input[batch_size*(input_num // batch_size):,:]
        out = model((batch, batch), training=False)
        embedding.append(out)
    embedding = np.vstack(embedding)
    return embedding


def generate_hard_mask(model,type):
    """
    We defined three threshold in config.py,which are [semi_hard_positive_threshold,semi_hard_negative_threshold,hardest_threshold].
    The semi hard positive samples are which similarity <= semi_hard_positive_threshold.
    The semi hard negative samples are which semi_hard_negative_threshold<= similarity <= hardest_threshold.
    The hardest negative samples are wich similarity >= hardest_threshold.
    :param model:
    :param type:'alloc' or 'free' .string.
    :return:
    """
    with open(config.train_data_prefix + os.sep + type + "/" + type + ".train") as f:
        func_true = []
        for line in f.readlines():
            if len(line)<=1:
                break
            func = funcname_to_vector(line.strip())
            func_true.append(func)
    func_true = np.vstack(func_true)
    embedding = batch_extract_embedding(model,func_true,batch_size=config.inference_batch)
    embedding = embedding / tf.norm(embedding,axis=1,keepdims=True)
    mean_embedding = np.mean(embedding,axis=0,keepdims=True)
    np.save(config.train_data_prefix + os.sep + type + "/" + "mean_embedding",mean_embedding)
    distance = np.dot(embedding,mean_embedding.T)
    semi_hard_positive_mask = distance <= config.semi_hard_positive_threshold
    semi_hard_positive_count = np.sum(semi_hard_positive_mask)
    tp = np.sum(distance>=config.inference_threshold)
    fn = np.sum(distance< config.inference_threshold)

    with open(config.train_data_prefix +os.sep + type + "/" + "neg.train") as f:
        func_neg = []
        for line in f.readlines():
            if len(line)<=1:
                break
            func = funcname_to_vector(line.strip())
            func_neg.append(func)
    func_neg = np.vstack(func_neg)
    embedding = batch_extract_embedding(model,func_neg,batch_size=config.inference_batch)
    embedding = embedding / tf.norm(embedding,axis=1,keepdims=True)
    distance = np.dot(embedding,mean_embedding.T)

    hardest_negative_mask = distance >= config.hardest_threshold
    semi_hard_negative_mask = (distance<= config.hardest_threshold) & (distance>=config.semi_hard_negative_threshold)
    hardest_negative_count = np.sum(hardest_negative_mask)
    semi_hard_negative_count = np.sum(semi_hard_negative_mask)
    fp = np.sum(distance >=config.inference_threshold)

    accuracy = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2*accuracy*recall) / (accuracy + recall)
    print("semi_hard_positive_count:%s\nsemi_hard_negative_count:%s\nhardest_negative_count:%s\n"%(semi_hard_positive_count,semi_hard_negative_count,hardest_negative_count))
    return semi_hard_positive_mask,hardest_negative_mask,semi_hard_negative_mask,accuracy,recall,f1_score


def dataset_split_triplet():
    with open("subword_dataset/tmp/alloc","r") as f:
        alloc_len = len(f.readlines()) -2
    with open("subword_dataset/tmp/free","r") as f:
        free_len = len(f.readlines()) -2
    with open("subword_dataset/tmp/neg","r") as f:
        neg_len = len(f.readlines()) -2

    min_len = min([alloc_len,free_len,neg_len]) * config.test_ratio
    min_len = int(min_len)

    with open("subword_dataset/tmp/alloc", "r") as f:
        alloc_list = []
        for line in f.readlines():
            if len(line) <=1:
                break
            alloc_list.append(line)
        test_alloc_list = np.random.choice(alloc_list,size = min_len,replace=False)
        train_alloc_list = []
        for alloc in alloc_list:
            if alloc not in test_alloc_list:
                train_alloc_list.append(alloc)

    with open("subword_dataset/tmp/free","r") as f:
        free_list = []
        for line in f.readlines():
            if len(line) <=1 :
                break
            free_list.append(line)
        test_free_list = np.random.choice(free_list,size=min_len,replace=False)
        train_free_list = []
        for free in free_list:
            if free not in test_free_list:
                train_free_list.append(free)

    with open("subword_dataset/tmp/neg","r") as f:
        neg_list = []
        for line in f.readlines():
            if len(line) <=1 :
                break
            neg_list.append(line)
        test_neg_list = np.random.choice(neg_list,size=min_len,replace=False)
        train_neg_list = []
        for neg in neg_list:
            if neg not in test_neg_list:
                train_neg_list.append(neg)

    with open("subword_dataset/triplet_trainning/alloc.train","w") as f:
        f.writelines(train_alloc_list)
    with open("subword_dataset/triplet_trainning/alloc.test","w") as f:
        f.writelines(test_alloc_list)
    with open("subword_dataset/triplet_trainning/free.train","w") as f:
        f.writelines(train_free_list)
    with open("subword_dataset/triplet_trainning/free.test","w") as f:
        f.writelines(test_free_list)
    with open("subword_dataset/triplet_trainning/neg.train","w") as f:
        f.writelines(train_neg_list)
    with open("subword_dataset/triplet_trainning/neg.test","w") as f:
        f.writelines(test_neg_list)


def dataset_generation(type = "target",trainning=True):
    """
    initial dataset pipeline.
    :param type:
    :param trainning:
    :return:
    """
    data = tf.data.Dataset.from_generator(generate_func_pair,output_types=(tf.int32,tf.int32,tf.int32),args=[type,trainning])
    data = data.shuffle(buffer_size=config.Buffer_size)
    data = data.batch(batch_size=config.mini_batch)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data


def dataset_generation_hard(semi_hard_positive_mask,hardest_negative_mask,semi_hard_negative_mask,type = "target",trainning=True):
    """
    Use hard samples to guide the pipeline generation.
    :param semi_hard_positive_mask:
    :param hardest_negative_mask:
    :param semi_hard_negative_mask:
    :param type:
    :param trainning:
    :return:
    """
    data = tf.data.Dataset.from_generator(generate_resample_pair,output_types=(tf.int32,tf.int32,tf.int32),args=[type,semi_hard_positive_mask,hardest_negative_mask,semi_hard_negative_mask,trainning])
    data = data.shuffle(buffer_size=config.Buffer_size)
    data = data.batch(batch_size=config.mini_batch)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data

def generate_triplet_data():
    """
    混合alloc，free和neg中的函数，并添加标签返回
    采样策略： alloc : free : neg = 1 :1 :2.  生成一个随机数 rand ，取rand %4对应的一个样本。
    neg:0
    alloc:1
    free:2
    :return:
    """
    alloc_list = []
    with open("subword_dataset/triplet_trainning/alloc.train","r") as f:
        for line in f.readlines():
            if len(line) <=1:
                break
            alloc_list.append(line.strip())

    free_list = []
    with open("subword_dataset/triplet_trainning/free.train","r") as f:
        for line in f.readlines():
            if len(line) <=1 :
                break
            free_list.append(line.strip())

    neg_list = []
    with open("subword_dataset/triplet_trainning/neg.train","r") as f:
        for line in f.readlines():
            if len(line) <=1:
                break
            neg_list.append(line.strip())

    while True:
        rand = np.random.randint(low=0,high=999999,size=1)[0]
        y = rand % 3
        if y==0:
            sample = np.random.choice(neg_list,size=1)[0]

        elif y==1:
            sample = np.random.choice(alloc_list,size=1)[0]
        else:
            sample = np.random.choice(free_list,size=1)[0]
        sample = funcname_to_vector(sample)
        yield (sample,y)

def dataset_generation_triplet(trainning=True):
    """
    生成训练triplet用的数据集。
    :param trainning:
    :return:
    """
    data = tf.data.Dataset.from_generator(generate_triplet_data,output_types=(tf.int32,tf.int32))
    data = data.shuffle(buffer_size=config.Buffer_size)
    data = data.batch(config.mini_batch)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data


if __name__ == "__main__" :
    #_,_ = create_id_char_map()
    out_dir = config.train_data_prefix
    dataset_split(out_dir, multi=config.multi)
    #free_dir = out_dir + os.sep + "free"
    #if not os.path.exists(free_dir):
    #    os.mkdir(free_dir)
    #normalize_dir("subword_dataset/tmp/free", free_dir, None)
    #dataset_split_triplet()
