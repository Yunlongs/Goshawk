import re
import json

def parse_func_line(line):
    pattern = re.compile(r"\|(.*?)@(.*?)\|")
    ret = pattern.findall(line)
    funcname = ret[0][1]
    return funcname

def parse_func_file(func_list_path):
    pattern = re.compile(r"\|(.*?)@(.*?)\|")
    f = open(func_list_path,"r")
    tmp = []
    for line in f:
        ret = pattern.findall(line)
        if ret == []:
            continue
        funcname = ret[0][1].lower()
        ret_type = ret[0][0].lower()
        argnames = []
        argtype = []
        ret.pop(0)
        for i in ret:
            argnames.append(i[1].lower())
            argtype.append(i[0].lower())
        if len(argnames) != len(argtype):
            print("find a error function!len(argnames) != len(argtypes)! :",funcname)
            continue
        tmp.append([funcname,ret_type,argnames,argtype])
    f.close()
    return tmp

def parse_func_line_json(line):
    func = json.loads(line.strip)
    return func

def parse_func_file_yield(func_list_path):
    pattern = re.compile(r"\|(.*?)@(.*?)\|")
    with open(func_list_path) as f:
        line = f.readline()
        while line:
            ret = pattern.findall(line)
            if ret == []:
                line = f.readline()
                continue
            funcname = ret[0][1]
            yield funcname,line
            line = f.readline()

def parse_origin_func(line):
    """
    将原始的函数原型转化为 kernel中的函数原型格式
    :param line: 原始的c类函数原型
    :return: 用@|分割的函数原型格式
    """
    lbrace,rbrace = line.index("("),line.rindex(")")
    args = line[lbrace+1:rbrace]
    arg_type =[]
    arg_param = []
    for arg in args.split(","):
        if arg.find("(") != -1 or arg.find(")") != -1:
          type = "void *"
          param = "ptr"
        elif arg.find("*") != -1:
            idx = arg.rindex("*")
            type = arg[:idx+1]
            param = arg[idx+1 :]
        elif arg=="void" or arg=="":
            continue
        else:
            idx = arg.rindex(" ")
            type = arg[:idx]
            param = arg[idx+1:]
        arg_type.append(type.strip())
        arg_param.append(param.strip())

    funcs = line[:lbrace]

    index = len(funcs) -1
    while index>=1:
        char = funcs[index]
        if char not in "abcdefghijklmnopqrstuvwxyz_0123456789":
            break
        index -=1
    funcname = funcs[index +1 :]
    ret_type = funcs[:index+1]

    full_name = "|" + ret_type.strip() + "@" + funcname.strip() + "|"
    for i in range(len(arg_type)):
        full_name += "|" + arg_type[i] + "@" + arg_param[i] + "|"
    print(full_name)
    return full_name


if __name__ == "__main__":
    pass