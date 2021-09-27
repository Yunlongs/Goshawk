import numpy as np
import time,os,json
import config
from embedding import load_embedding,get_reference_embedding,extract_embedding,create_link_func_string
from normalize import normalize_on_file
"""
This file only used in experiment,not suit to product environment.
This because the paths in each function are inflexible, they are set according the experiment path.
"""

def norm(vector):
    """
    l2 normalize.
    :param vector:
    :return:
    """
    res = np.sqrt(np.sum(np.square(vector)))
    return vector/res


def get_func_string_from_json(func):
    """
    Give a json typed function prototype,
    we concat each part to a string to recover its raw function prototype.
    :param func: a json string.
    :return:
    """
    func = json.loads(func)
    ret_type= func['return_type']
    funcname = func['funcname']
    parms = func['params']
    func_string = ret_type + " " + funcname + "(" + parms + ")"
    return func_string


def cal_acc_recall(out_path,model_type,exper_id,type="alloc"):
    """
    Compute the accuracy,reall and f1-score. For test dataset.
    :param out_path:  The file path that you want to save result.
    :param model_type: The model type that you trained. "Transformer,SimpleRNN,SimpleLSTM,2L_LSTM"..
    :param exper_id:
    :param type:
    :return:
    """
    start = time.time()
    pos_embedding = load_embedding("output/%s/experiment_%s/%s/embedding/%s.test.embedding"%(model_type,exper_id,type,type))
    neg_test = load_embedding("output/%s/experiment_%s/%s/embedding/neg.test.embedding"%(model_type,exper_id,type))
    mean_embedding = get_reference_embedding(type)
    #pos_linkmap,_ = create_link_func_string("subword_dataset/tmp/%s/%s.test"%(type,type),"subword_dataset/training/%s/%s.test"%(type,type))
    #neg_linkmap,_ = create_link_func_string("subword_dataset/tmp/%s/neg.test"%(type),"subword_dataset/training/%s/neg.test"%(type))
    tp,fp,fn,tn = 0,0,0,0

    result  =""
    res_dict = {}
    i = 0
    for k,v in pos_embedding.items():
        vector = norm(v)
        cos = np.dot(vector,mean_embedding)
        res_dict[k] = cos
        if cos>0.5:
            tp +=1
        else:
            fn +=1
        i +=1
    res_dict = sorted(res_dict.items(),key=lambda d:d[1],reverse=True)
    for k,cos in res_dict:
        result += " ".join([k, str(cos)]) + "\n"
    result += "-----------------\n"
    alloc_tp = tp
    acc = tp/(fn + tp)
    result += "\n%s True:%s\n False: %s\nAccuracy:%s\n\n"%(type,alloc_tp,fn,acc)

    res_dict ={}
    i = 0
    for k,v in neg_test.items():
        vector = norm(v)
        cos = np.dot(vector,mean_embedding)
        res_dict[k] = cos
        if cos<0.5:
            tn +=1
        else:
            fp +=1
        i +=1
    res_dict = sorted(res_dict.items(),key=lambda d:d[1],reverse=True)
    for k,cos in res_dict:
        result += " ".join([k, str(cos)]) + "\n"
    neg_tp = tn
    acc = neg_tp/(neg_tp + fp)
    result += "-----------------\n\nNeg True:%s\nFalse:%s\n Accuracy:%s\n\n"%(neg_tp,fp,acc)
    end = time.time()
    accuracy = tp/(tp+fp)
    recall = tp/(tp + fn)
    f1_score = (2*accuracy * recall) / (accuracy + recall)
    result += "---------------\n"
    result += "tp:" + str(tp) + "\n"
    result += "fp:" + str(fp) + "\n"
    result += "fn:" + str(fn) + "\n"
    result += "accuracy:" + str(accuracy) + "\n"
    result += "recall:" + str(recall) + "\n"
    result += "f1_score:" + str(f1_score) + "\n"
    result += "total time:" + str(end-start)
    print(result)
    with open(out_path,"w") as f:
        f.write(result)

def check_returned_result(return_funcs,noreturn_funcs,funcname_funcjson_map):
    convert_origin_to_prototype("temp/free_label.func")
    convert_prototype_to_json("temp/free_label.func_prototype")
    label_funcs = []
    with open("temp/free_label.func_prototype.json","r") as f:
        for line in f.readlines():
            if len(line) <=1 :
                break
            func = json.loads(line)
            funcanme = func['function']
            label_funcs.append(funcanme)

    return_funcs = sorted(return_funcs.items(), key=lambda d: d[1], reverse=True)
    noreturn_funcs = sorted(noreturn_funcs.items(), key=lambda d: d[1], reverse=True)

    tp,fp,tn,fn = 0,0,0,0
    tp_funcs,fp_funcs,tn_funcs,fn_funcs = {},{},{},{}
    for funcname,sim in return_funcs:
        if funcname.lower() in label_funcs:
            tp +=1
            tp_funcs[funcname] = sim
        else:
            fp +=1
            fp_funcs[funcname] = sim
    for funcname,sim in noreturn_funcs:
        if funcname.lower() in label_funcs:
            fn +=1
            fn_funcs[funcname] = sim
        else:
            tn +=1
            tn_funcs[funcname] = sim

    accuracy = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * accuracy * recall / (accuracy + recall)

    result = "--------------------------\nTrue Positive:\n----------------------\n\n"
    for func,sim in tp_funcs.items():
        line = get_func_string_from_json(funcname_funcjson_map[func]) + "\t" + str(sim) + "\n"
        result += line
    result += "\n\n\n-----------------------\nFalse Positive:\n---------------------\n\n"
    for func,sim in fp_funcs.items():
        line = get_func_string_from_json(funcname_funcjson_map[func]) + "\t" + str(sim) + "\n"
        result += line
    result += "\n\n\n-----------------------\nFalse Negative:\n---------------------\n\n"
    for func, sim in fn_funcs.items():
        line = get_func_string_from_json(funcname_funcjson_map[func]) + "\t" + str(sim) + "\n"
        result += line
    result += "\n\n\n-----------------------\nTrue Negative:\n---------------------\n\n"
    for func, sim in tn_funcs.items():
        line = get_func_string_from_json(funcname_funcjson_map[func]) + "\t" + str(sim) + "\n"
        result += line
    result +=  "\n\n\n\ntp:%s\nfp:%s\nfn:%s\ntn:%s\nAccuracy:%s\nRecall:%s\nF1-score:%s\n"%(tp,fp,fn,tn,accuracy,recall,f1_score)

    with open("temp/free_result_consider_callee","w") as f:
        f.write(result)




if __name__ == "__main__":
    model_dir = "Transformer"
    exper_id = "103"
    model_prefix = "maxauc_model"
    in_path = "subword_dataset/training/"
    out_path = "output/%s/experiment_%s/" % (model_dir,exper_id)
    extract_embedding(in_path = in_path,out_path = out_path,model_prefix=model_prefix)
    type = "alloc"
    cal_acc_recall("output/%s/experiment_%s/%s/%s_result.txt"%(model_dir,exper_id,type,type),model_dir,exper_id,type=type)
    type = "free"
    cal_acc_recall("output/%s/experiment_%s/%s/%s_result.txt"%(model_dir,exper_id,type,type),model_dir,exper_id,type=type)