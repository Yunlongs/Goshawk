import glob
from parse_func import parse_func_line,parse_func_line_json
import re
import json
import os

def remove_dup_and_merge(dir_path,outpath):
    """
    将dir_path中文件中的函数进行合并成一个文件，并去重
    :param dir_path:
    :param outpath:
    :return:
    """
    paths = glob.glob(dir_path+"/*")
    func_dict ={}

    for path in paths:
        with open(path,"r") as f:
            for line in f.readlines():
                funcname = json.loads(line.strip())['funcname']
                if func_dict.get(funcname):
                    continue
                func_dict[funcname] = line.lower()

    with open(outpath,"w") as f:
        for line in  func_dict.values():
            f.write(line)

def remove_tp_funcs(target_file,alloc_file,free_file,out_file):
    """
    remove all the labeled alloc and free functions from the kernel API functions file.
    And write the rest functions into ".neg" file.
    :return:
    """
    with open(alloc_file, "r") as f:
        alloc_tp_funcs = []
        for line in f.readlines():
            if len(line)<=1:
                break
            func = json.loads(line.strip())
            funcname = func['function']
            print(funcname)
            alloc_tp_funcs.append(funcname.lower())

    with open(free_file, "r") as f:
        free_tp_funcs = []
        for line in f.readlines():
            if len(line)<=1:
                break
            func = json.loads(line.strip())
            funcname = func['function']
            free_tp_funcs.append(funcname.lower())

    pure_negative_funcs= []
    with open(target_file, "r") as f:
        for line in f.readlines():
            if len(line) <=1:
                break
            func = json.loads(line.strip())
            funcname = func['funcname'].lower()
            if funcname in alloc_tp_funcs or funcname in free_tp_funcs:
                continue
            else:
                pure_negative_funcs.append(line)

    with open(out_file,"w") as f:
        f.writelines(pure_negative_funcs)

def purify_dataset(input_dir):
    """
    对输入的目录中的函数，过滤掉那些库函数，只保留.c结尾文件中定义的函数
    :param input_dir: 从kernel中提取的subsystem函数目录
    :return:
    """
    in_paths = glob.glob(input_dir + "/*")
    for path in in_paths:
        res = []
        with open(path,"r") as f:
            for line in f.readlines():
                if line =="":
                    break
                term = line.strip().replace("'","\"")
                func = json.loads(term)
                if not func['file'].endswith(".c"):
                    continue
                res.append(term + "\n")
        with open(path,"w") as f:
            f.writelines(res)



if __name__ == "__main__":
    #purify_dataset("../Dataset/bluetooth/")
    #remove_dup_and_merge("../Dataset/labeled_dataset/mm","../Dataset/labeled_dataset/all_funcs")
    remove_tp_funcs("../../Dataset/labeled_dataset/all_funcs","../../Dataset/labeled_dataset/merge_alloc.json","../../Dataset/labeled_dataset/merge_free.json","../../Dataset/labeled_dataset/all_funcs.neg")

