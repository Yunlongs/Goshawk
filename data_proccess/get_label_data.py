import os
import json
from parse_func import parse_func_file_yield,parse_origin_func,parse_func_line,parse_func_file

def get_funcname_dict():
    """
    因为alloc_list和free_list中只有函数名，我们需要从kernel函数中找到它们的原型
    :return:
    """
    alloc_dcit = {}
    free_dict = {}
    with open("../Dataset/alloc_list","r") as f:
        for line in f.readlines():
            if line == "":
                break
            alloc_dcit[line.strip()] = 1
    with open("../Dataset/free_list","r") as f:
        for line in f.readlines():
            if line == "":
                break
            free_dict[line.strip()] = 1
    return alloc_dcit,free_dict

def get_func_prototype():
    """
        因为alloc_list和free_list中只有函数名，我们需要从kernel函数中找到它们的原型
    :return:
    """
    alloc_dict,free_dict = get_funcname_dict()
    alloc_write_path = "../Dataset/alloc_prototype"
    free_write_path = "../Dataset/free_prototype"
    f_alloc = open(alloc_write_path,"w")
    f_free = open(free_write_path,"w")

    i = 0
    for funcname,line in parse_func_file_yield("../Dataset/kernel_funcs"):
        i +=1
        if i%1000 ==0:
            print(i)
        if alloc_dict.get(funcname) != None:
            f_alloc.write(line)
            alloc_dict[funcname] = 0
        if free_dict.get(funcname) != None:
            f_free.write(line)
            free_dict[funcname] = 0
    f_alloc.close()
    f_free.close()
    for k,v in alloc_dict.items():
        if v==1:
            print(k)
    print("-----------------------")
    for k,v in free_dict.items():
        if v==1:
            print(k)



def convert_origin_to_prototype(filename):
    """
    将一个文件中，原始c-type的函数原型转化成'|'分隔后的函数原型。
    Convert the original c-type function prototype in the file "filename"
        to '|' separated function prototype.
    :return:
    """
    funcs  = []
    with open(filename,"r") as f:
        for line in f.readlines():
            if line=="\n":
                break
            full_name = parse_origin_func(line)
            funcs.append(full_name)

    with open(filename, "w") as f:
        for func in funcs:
            f.write(func + "\n")

def merge_and_removedup(file1,file2,outpath):
    """
    将两个文件中的函数原型prototype文件进行合并，并且从去重
    Merge the function prototypes of two files, and do deduplicate.
    :param file1:
    :param file2:
    :param outpath:
    :return:
    """
    f1 = open(file1,"r")
    f2 = open(file2,"r")
    f_out = open(outpath,"w")

    funcname_dict = {}
    f2_lines = f2.readlines()
    for line in f2_lines:
        funcname = parse_func_line(line)
        funcname_dict[funcname] = line

    for line in f1.readlines():
        funcname = parse_func_line(line)
        funcname_dict[funcname] = line

    for k,v in funcname_dict.items():
        f_out.write(v)
    f1.close()
    f2.close()
    f_out.close()

def convert_prototype_to_json(filename):
    """
    将一个文件中用'|'做分割的函数原型，转化成json的格式，方便读取
    Convert '|' separated function prototype in the file "filename" to json format,
        which is easier to handle.
    For example:
        '|‘ separated function prototype: |void *@kmalloc_array||size_t@n||size_t@size||gfp_t@flags|
        json-format function prototype: {"function": "kmalloc", "return_type": "void *", "parms": "size_t@size,gfp_t@flags,"}
    :param filename:
    :return:
    """
    tmp = parse_func_file(filename)
    with open(filename,"w") as f:
        for funcname,ret_type,argnames,argtypes in tmp:
            func = {}
            func['funcname'] = funcname
            func['return_type'] = ret_type
            fullarg = ""
            for i in range(len(argnames)):
                fullarg += argtypes[i].replace(',',' ') + "@" + argnames[i] + ","
            func["params"] = fullarg
            f.write(json.dumps(func).lower() + "\n")

def remove_dup_json_file(filename):
    """
    从json file中移除重复的函数
    :param filename:
    :return:
    """
    func_dict = {}
    with open(filename,"r") as f:
        for line in f.readlines():
            if len(line)<=1:
                break
            func = json.loads(line)
            func_dict[func['funcname']] = line
    with open(filename,"w") as f:
        for v in func_dict.values():
            f.write(v)


if __name__ =="__main__":
    convert_origin_to_prototype("../temp/mm_alloc_all.txt")
    convert_prototype_to_json("../temp/mm_alloc_all.txt")
