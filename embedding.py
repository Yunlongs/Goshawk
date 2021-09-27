import numpy as np
import tensorflow as tf
from dataset import funcname_to_vector
import time, os, config


def norm(vector):
    """
    l2 normalize.
    :param vector:
    :return:
    """
    res = np.sqrt(np.sum(np.square(vector)))
    return vector / res


def extract_embedding_per_file(model, in_file, out_file):
    """
    Extract embedding form in_file, which contains the normalized function prototypes, to out_file.
     The functions in the file must be normalized! If not, please normalize them by normalize.py.

    For example, the format of function prototypes in in_file should be as follow:
    <cls> <ptr> ef x alloc channel ( <ptr> ef x <dot> <noptr> i <dot> <ptr> old channel )
    <cls> <ptr> do kmalloc ( <noptr> size <dot> <noptr> flags <dot> <noptr> caller )
    <cls> <ptr> alloc unbound pw q ( <ptr> wq <dot> <ptr> attrs )
    ....
    :param model: Your trained instantiated model.
    :param in_file: input file. path.
    :param out_file:output embedding file. path.
    :return: None
    """
    with open(in_file, "r", encoding="utf-8") as f:
        input_batch = []
        file_embeddings = []
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) <= 1:
                break
            input_batch.append(funcname_to_vector(line.strip()))
            if (i + 1) % config.inference_batch == 0:
                input_batch = np.vstack(input_batch)
                inputs = (input_batch, input_batch)
                embeddings = model(inputs, training=False)
                file_embeddings.append(embeddings)
                input_batch = []
        if len(input_batch) != 0:  # 处理剩下的函数
            input_batch = np.vstack(input_batch)
            inputs = (input_batch, input_batch)
            embeddings = model(inputs, training=False)
            file_embeddings.append(embeddings)

    embeddings = np.vstack(file_embeddings)
    with open(out_file, "w") as f:
        for i, line in enumerate(lines):
            if len(line) <= 1:
                break
            vector = embeddings[i, :].tolist()
            vector_str = " ".join([str(v) for v in vector])
            f.write(line.strip() + "|" + vector_str + "\n")


def extract_embedding_per_dir(model, in_dir, out_dir):
    """
    Extract embedding for a dir which include some function files.
    (This function only used in train.py)
    :param model:Your trained instantiated model.
    :param in_dir:input dir. dir path.
    :param out_dir:output dir. dir path.
    :return:
    """
    list_files = os.listdir(in_dir)
    total = 0
    for _file in list_files:
        file_path = in_dir + os.sep + _file
        out_path = out_dir + os.sep + _file + ".embedding"
        if file_path.endswith(".test"):
            start = time.time()
        extract_embedding_per_file(model, file_path, out_path)
        if file_path.endswith(".test"):
            end = time.time()
            total += end - start
    print("embedding test time:", total)


def extract_embedding(in_path, out_path, model_prefix):
    """
    Extract embedding for both 'alloc' and 'free'
     (This function only used in train.py)
    :param in_path:
    :param out_path:
    :param model_prefix:
    :return:
    """
    types = ["alloc", "free"]
    for type in types:
        model_dir = out_path + type + os.sep + model_prefix
        model = tf.keras.models.load_model(model_dir)
        in_dir = in_path + type
        out_dir = out_path + type + os.sep + "embedding"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        extract_embedding_per_dir(model, in_dir, out_dir)


def load_embedding(file):
    """
    load embedding from a file.
    In the file, the embedding format is:
    func | embedding.

    :param file: filepath
    :return:
    """
    embedding_dict = {}
    with open(file, "r") as f:
        for line in f.readlines():
            if len(line) <= 1:
                continue
            sp = line.strip().split("|")
            funcname = sp[0]
            vector = sp[1:][0]
            vector = [float(v) for v in vector.split()]
            embedding_dict[funcname] = np.array(vector)
    return embedding_dict


def load_embedding_to_list(file):
    """
    load embedding from a file. In the file, the embedding format is:
    func | embedding.
    Note:Because some different functions can have the same prototype after segment.
    So we return the embedding list here.

    :param file: filepath
    :return:
    """
    embedding_list = []
    with open(file, "r") as f:
        for line in f.readlines():
            if len(line) <= 1:
                continue
            sp = line.strip().split("|")
            vector = sp[1:][0]
            vector = [float(v) for v in vector.split()]
            embedding_list.append(np.array(vector))
    return embedding_list


def get_reference_embedding(model_name):
    """
    A reference embedding a is the mean vector of embeddings from dataset.
    This reference embedding can be directly used to do comparision.

    :param type:"alloc" or "free"
    :return:
    """
    mean_embedding_path = os.path.join(config.model_dir, model_name, "embedding", "mean_embedding.npy")
    if os.path.exists(mean_embedding_path):
        return np.load(mean_embedding_path)
    train_embedding_path = os.path.join(config.model_dir, model_name, "embedding", "target.train.embedding")
    train_embedding = load_embedding(train_embedding_path)
    mean_embedding = np.zeros((config.feature_size,))
    for k, v in train_embedding.items():
        mean_embedding += norm(v)
    mean_embedding /= len(train_embedding)
    np.save(mean_embedding_path, mean_embedding)
    return mean_embedding


def create_link_func_string(proto_file, seged_file):
    """
    Since we have normalized and segmented function prototype, due to this irreversible procedure,
    we need to create a map relation from seged function prototype to original function prototype.

    We can recover the original function prototype when we need to inspect the inference result.

    :param proto_file:A json file, where each line is a json string of a function prototype.
    :param seged_file:Normalized function prototypes.
    :return:
    """
    import json
    link_func_string = []
    link_func_name = []
    with open(seged_file, "r") as f_seg:
        with open(proto_file, "r") as f_proto:
            for line in f_seg.readlines():
                if len(line) <= 2:
                    break
                try:
                    func = json.loads(f_proto.readline().strip())
                except:
                    break
                funcname = func['funcname']
                if func.get('file'):
                    file = func['file']
                    file = os.path.basename(file)
                else:
                    file = ""
                ret_type = func['return_type']
                params = func['params']
                func_string = ret_type + " " + funcname + "(" + params + ")" + "\t" + file
                link_func_string.append(func_string)
                link_func_name.append(funcname)
    return link_func_string, link_func_name
