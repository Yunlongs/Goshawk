import tensorflow as tf
from normalize import normalize_on_file
from embedding import extract_embedding_per_file, load_embedding, create_link_func_string, get_reference_embedding, \
    load_embedding_to_list
import os
import config
import numpy as np
import time
from data_proccess.get_label_data import convert_origin_to_prototype, convert_prototype_to_json


def norm(vector):
    """
    l2 norm.
    :param vector:
    :return:
    """
    res = np.sqrt(np.sum(np.square(vector)))
    return vector / res


def extract_embedding(model, input_path, out_path):
    """
    extract embedding for a file.
    """
    extract_embedding_per_file(model, input_path, out_path)


def calculate_similarity(target_embedding_file, link_func_string, link_func_name, model_name):
    """
    When getting the embedding of target functions, we can inference their similarities.
    :param target_embedding_file:
    :param out_path: save the similarities result to a file.
    :return:
    """
    target_embedding = load_embedding_to_list(target_embedding_file)
    mean_embedding = get_reference_embedding(model_name).reshape((config.embedding_size,))
    result = ""
    func_cos_dict = {}
    func_name_cos_dict = {}
    for index, v in enumerate(target_embedding):
        vector = norm(v)
        cos = np.dot(vector, mean_embedding)
        func_cos_dict[link_func_string[index]] = cos
        func_name_cos_dict[link_func_name[index]] = cos
    func_cos_dict = sorted(func_cos_dict.items(), key=lambda d: d[1], reverse=True)
    for k, cos in func_cos_dict:
        result += " ".join([k, str(cos)]) + "\n"
    result += "-----------------\n"
    with open(config.func_sim_path, "w") as f:
        f.write(result)
    with open(config.func_name_sim_path, "w") as f:
        for func_name, cos in func_name_cos_dict.items():
            f.write(" ".join([func_name, str(cos)]) + "\n")
    return func_name_cos_dict


def calculate_similarity_no_sort(test_embedding_file, type):
    test_embedding = load_embedding_to_list(test_embedding_file)
    mean_embedding = get_reference_embedding(type)
    similarity_list = []
    for v in test_embedding:
        vector = norm(v)
        cos = np.dot(vector, mean_embedding)
        similarity_list.append(cos)
    return similarity_list


def working_on_json_function_prototype(model, json_file, model_name):
    """
    Given a file of function prototypes with json type, we first normalize and segment these function prototypes.
    Then generate embeddings for function prototypes.
    Finally, calculate their similarity according to reference embedding, and save the result to a file.

    :param model: Your trained instantiated model.
    :param json_file: input json file of function prototypes.
    :return:
    """
    normalize_on_file(json_file, "alloc", "temp/func_seg")
    start = time.time()
    extract_embedding(model, "temp/func_seg", "temp/embedding")
    end = time.time()
    print("extracting time:", end - start)
    link_func_string, link_func_name = create_link_func_string(json_file, "temp/func_seg")
    func_similarity_as_name = calculate_similarity("temp/embedding", link_func_string, link_func_name, model_name)
    return func_similarity_as_name


def working_on_raw_function_prototype(model_name, filename):
    """
    Convert raw function prototypes to json type.
    And then we can get their similarities by function 'working_on_json_function_prototype'.
    :param model: Your trained instantiated model.

    :param funcs:
    :param type:
    :return:
    """
    model = get_model(model_name)
    convert_origin_to_prototype(filename)
    convert_prototype_to_json(filename)
    _ = working_on_json_function_prototype(model, filename, model_name)


def list_model():
    print("Current available model:")
    result = ", ".join(os.listdir("model"))
    print(result)


def get_model(model_name):
    list_model()
    assert model_name in os.listdir("model"), "model name %s not in directory \"model\""%(model_name)
    model = tf.keras.models.load_model(config.model_dir + os.sep + model_name + os.sep + "maxauc_model")
    return model

def similarity_inference(model_name, input_file):
    import shutil
    copy_file = config.temp_dir + os.sep + os.path.basename(input_file)
    shutil.copy(input_file, copy_file)
    working_on_raw_function_prototype(model_name, copy_file)

if __name__ == "__main__":
    working_on_raw_function_prototype("alloc","subword_dataset/test.func")



