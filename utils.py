import json,os
import config
from parse_call_graph import read_caller_and_callee, convert_call_graph_to_dict


class Node(object):
    def __init__(self, funcanme):
        self.next = []
        self.funcname = funcanme
        self.depth = 1



def argmax(indexs):
    max = 0
    max_arg = 0
    for i, index in enumerate(indexs):
        if index > max:
            max = index
            max_arg = i
    return max_arg


def cleanup_free_null_check(file):
    func_maps = {}
    if not os.path.exists(file):
        return
    with open(file) as f:
        for line in f.readlines():
            if len(line) <= 4:
                break
            func, index = line.strip().split()
            index = int(index)
            if func_maps.get(func) is None:
                func_maps[func] = [0, 0, 0, 0, 0, 0, 0]
            if (index >= len(func_maps[func])):
                continue
            func_maps[func][index] += 1

    new_map = {}
    for k, v in func_maps.items():
        max_arg = argmax(v)
        max = v[max_arg]
        new_func = k + "\t" + str(max_arg)
        new_map[new_func] = max
    new_map = sorted(new_map.items(), key=lambda d: d[1], reverse=True)
    with open(file, "w") as f:
        for func, num in new_map:
            f.write(func + "\t" + str(num) + "\n")


def add_primitive_functions(out_dir="output/free/"):
    normal_free = []
    customized_free = []
    with open("temp/seed_free.txt", "r") as f:
        for line in f.readlines():
            func, index = line.strip().split()
            index = int(index)
            if index == 0:
                normal_free.append(func)
            else:
                func_json = {}
                func_json["funcname"] = func
                func_json["param_names"] = [index, "map"]
                func_json["member_name"] = []
                customized_free.append(func_json)
    with open(out_dir + "FreeNormalFile.txt", "a") as f:
        for free in normal_free:
            f.write(free + "\n")
    with open(out_dir + "FreeCustomizedFile.txt", "a") as f:
        for free in customized_free:
            f.write(json.dumps(free) + "\n")


def count_candidate_deallocation():
    count = 0
    with open("temp/func_similarity", "r") as f:
        for line in f.readlines():
            if "*" in line:
                count += 1
    print(count)


def decide_minrest():
    with open("temp/extract_all_func", "r") as f:
        nr_funcs = len(f.readlines())
    if nr_funcs < 10000:
        config.min_reset = 2
    elif nr_funcs < 50000:
        config.min_reset = 5
    elif nr_funcs < 100000:
        config.min_reset = 10
    else:
        config.min_reset = 20


def get_func_call_chains(func, seed_funcs, candidate_funcs, call_graph, call_chains):
    if func == "ENGINE_unregister_RAND":
        print(1)

    callees = call_graph[func]["callees"]
    if call_chains.get(func) is not None:
        return call_chains[func]
    else:
        call_chains[func] = Node(func)

    depths = []
    for callee in callees:
        callee_name = callee["funcname"]

        if callee_name in seed_funcs:
            node = Node(callee_name)
            call_chains[func].next.append(node)
            depths.append(node.depth)
            continue

        if callee_name in candidate_funcs:
            if call_chains.get(callee_name) is None:
                node = get_func_call_chains(callee_name, seed_funcs, candidate_funcs, call_graph, call_chains)
                call_chains[func].next.append(node)
            else:
                node = call_chains[callee_name]
                call_chains[func].next.append(node)
            depths.append(node.depth)

    if len(depths) != 0:
        call_chains[func].depth = max(depths) + 1
    return call_chains[func]

def MMD_call_chains(call_graph):
    seed_funcs = []
    candidate_funcs = []
    call_chains = {}

    # read the seed funcs
    with open(config.seed_free_path, "r") as f:
        for line in f.readlines():
            func = line.strip().split()[0]
            seed_funcs.append(func)
    # read the candidate funcs
    with open(config.candidate_free_path, "r") as f:
        for line in f.readlines():
            func = line.strip()
            candidate_funcs.append(func)

    for func in candidate_funcs:
        call_chains[func] = get_func_call_chains(func, seed_funcs, candidate_funcs, call_graph, call_chains)
    return call_chains


def retrive_next_step_TU(project_dir, call_graph, call_chains, curr_iter):
    next_step_funcs = set()
    next_step_file = set()
    next_step_TU = list()

    for func, node in call_chains.items():
        if node.depth == curr_iter + 1:
            next_step_funcs.add(func)

    for func in next_step_funcs:
        caller = call_graph[func]["caller"]
        file = caller["file"]
        next_step_file.add(file)

    compilation_file = project_dir + os.sep + "compilation.json"
    with open(compilation_file, "r") as f:
        compilation_json = json.load(f)

    for TU in compilation_json:
        file = TU["file"]
        if file in next_step_file:
            next_step_TU.append(TU)

    if len(next_step_TU) == 0:
        return None
    return next_step_TU


def concat_two_file(file1, file2):
    with open(file1, "r") as f_r:
        with open(file2, "a") as f_w:
            f_w.write(f_r.read())


if __name__ == "__main__":
    # cleanup_null_check("temp/free_check.txt")
    #count_candidate_deallocation()
    call_graph = read_caller_and_callee(config.call_graph_path)
    call_graph = convert_call_graph_to_dict(call_graph)
    call_chains = MMD_call_chains(call_graph)
    retrive_next_step_TU("temp", call_graph, call_chains, 0)
# consume_skb	0	261
