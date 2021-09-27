import json
import random


def to_json(file):
    with open(file, "r") as f:
        res = f.read()
    with open(file, "w") as f:
        res = res.replace("'", "\"")
        # res = res.replace(", funcname",", \"funcname")
        f.write(res)


def remove_dup_caller(in_file, out_file):
    """
    移除重复的caller函数，并只保留含有最多callee函数的那个caller.
    :param in_file:
    :param out_file:
    :return:
    """
    caller_funcs = []
    children = {}
    with open(in_file, "r") as f:
        line = f.readline()
        current_caller = [line]
        while line and len(line) >= 1:
            line = f.readline()
            if line.startswith("\t"):
                func = json.loads(line.strip())
                funcname = func['funcname']
                if children.get(funcname) is None:
                    if func.get('number') is None:
                        func['number'] = 1
                    children[funcname] = "\t" + json.dumps(func) + "\n"
                else:
                    func = json.loads(children[funcname])
                    func['number'] += 1
                    children[funcname] = "\t" + json.dumps(func) + "\n"

            elif line and len(line) > 1:
                current_caller.append(children)
                caller_funcs.append(current_caller)
                current_caller = [line]
                children = {}

    clean_caller_funcs = {}
    for caller_func in caller_funcs:
        caller = caller_func[0]
        children = caller_func[1]
        num_of_children = len(children)
        # print(caller.strip())
        caller_funcname = json.loads(caller.strip())["funcname"]
        if clean_caller_funcs.get(caller_funcname) is None:
            clean_caller_funcs[caller_funcname] = caller_func
        else:
            exist_caller_func = clean_caller_funcs[caller_funcname]
            exist_children = exist_caller_func[1]
            exist_num_of_children = len(exist_children)
            if num_of_children > exist_num_of_children:
                clean_caller_funcs[caller_funcname] = caller_func

    with open(out_file, "w") as f:
        for caller_func in clean_caller_funcs.values():
            caller = caller_func[0]
            children = caller_func[1]
            f.write(caller)
            for v in children.values():
                f.write(v)
    print(len(clean_caller_funcs))


def read_caller_and_callee(in_caller_callee_file):
    """
    Read a caller-callee json file.

    We read the caller-callee relationship to the follow format:
    [
        {
            "caller": {...}
            "callees":[
                    {...},
                    {...}
            ]
        }
    ]

    :param in_caller_callee_file:
    :return:
    """
    caller_callee_funcs = []
    children = {}
    with open(in_caller_callee_file, "r") as f:
        current_func = {}
        line = f.readline()
        current_func["caller"] = json.loads(line)
        while line and len(line) > 1:
            line = f.readline()
            if line and line.startswith("\t"):
                func = json.loads(line.strip())
                funcname = func['funcname']
                if children.get(funcname) is None:
                    children[funcname] = func
            else:
                current_func["callees"] = list(children.values())
                caller_callee_funcs.append(current_func)
                if len(line) <= 1:
                    break
                current_func = {"caller": json.loads(line)}
                children = {}
    print("Call Graph read finished!\n total func number:%s" %len(caller_callee_funcs))
    return caller_callee_funcs


def write_call_graph_to_file(call_graph, file):
    with open(file, "w") as f:
        for func in call_graph:
            caller = func["caller"]
            callees = func["callees"]
            f.write(json.dumps(caller) + "\n")
            for callee in callees:
                f.write("\t" + json.dumps(callee) + "\n")


def convert_call_graph_to_dict(call_graph):
    call_graph_dict = {}
    for func in call_graph:
        caller = func["caller"]
        funcname = caller["funcname"]
        call_graph_dict[funcname] = func
    return call_graph_dict


def random_select(in_file, out_file):
    """
    随机选择3k个函数的call relationships
    :param in_file:
    :param out_file:
    :return:
    """
    k = 3000
    caller_funcs = []
    children = {}
    with open(in_file, "r") as f:
        line = f.readline()
        current_caller = [line]
        while len(line) >= 1:
            line = f.readline()
            if line.startswith("\t"):
                func = json.loads(line.strip())
                funcname = func['funcname']
                if children.get(funcname) is None:
                    children[funcname] = line
            else:
                current_caller.append(children)
                caller_funcs.append(current_caller)
                current_caller = [line]
                children = {}
    random.shuffle(caller_funcs)
    selected_funcs = []
    i = 0
    k = 0
    while k < 3000:
        caller_func = caller_funcs[i]
        caller = caller_func[0]
        caller_file = json.loads(caller)['file']
        if caller_file.endswith('.c') or caller_file.endswith('.h'):
            selected_funcs.append(caller_funcs[i])
            k += 1
        i += 1

    print(len(selected_funcs))

    with open(out_file, "w") as f:
        for caller_func in selected_funcs:
            caller = caller_func[0]
            children = caller_func[1]
            f.write(caller)
            for v in children.values():
                f.write(v)


def get_caller_func(in_file, out_file):
    """
    仅获得所有的caller函数
    :param in_file:
    :param out_file:
    :return:
    """
    caller_funcs = []
    with open(in_file, "r") as f:
        for line in f.readlines():
            if line.startswith("\t"):
                continue
            caller_funcs.append(line)
    with open(out_file, "w") as f:
        for func in caller_funcs:
            f.write(func)


def count_caller_number(in_caller_callee_file):
    """
    统计文件中有多少个caller函数。
    :param in_caller_callee_file:
    :return:
    """
    counter = 0
    with open(in_caller_callee_file, "r") as f:
        for line in f.readlines():
            if line.startswith("\t"):
                continue
            counter += 1
    print(counter)


if __name__ == "__main__":
    # label_file = "funcs/free_label.txt"
    # out_file = "funcs/free_label.func"
    # get_origin_prototype(label_file,out_file)
    in_file = "subword_dataset/FreeBSD/call_graph.json"
    out_file = "subword_dataset/FreeBSD/call_graph.json"
    to_json(in_file)
    remove_dup_caller(in_file, out_file)

    #graph = read_caller_and_callee(in_file)
    #graph = convert_call_graph_to_dict(graph)
    #calless = graph["kmem_cache_free"]
    pass
    #count_caller_number("temp/whole_belief_call_graph")