import config, json, os
from similarity_inference import working_on_json_function_prototype
from parse_call_graph import read_caller_and_callee, convert_call_graph_to_dict, write_call_graph_to_file
from normalize import parse_params, normalize_type_1
import tensorflow as tf
from utils import decide_minrest, MMD_call_chains

memory_flows = {}



class CallTree(object):
    def __init__(self):
        self.childs = []
        self.depth = 0

    def add_child(self, child):
        self.childs.append(child)
        if child.depth >= self.depth:
            self.depth = child.depth + 1


def get_all_funcs(in_file, out_file=None):
    """
    Given a json file, whether is it a caller-callee relationships, get all unique function prototypes.
    :param in_file:
    :param out_file:
    :return:
    """
    funcs = {}
    with open(in_file) as f:
        for line in f.readlines():
            if len(line) <= 1:
                break
            line = line.strip()
            func = json.loads(line)
            func_name = func['funcname']
            funcs[func_name] = func

    with open(out_file, "w") as f:
        for v in funcs.values():
            f.write(json.dumps(v) + "\n")
    return list(funcs.values())


def get_candidate_alloc_function_set(func_similarity, call_graph):
    """
    return the target function only according to their prototype similarity scores.
    filter out the functions whose sim_alloc smaller than infer_similarity.
    :param func_similarity:
    :param call_graph:
    :return:
    """
    candidate_function_set = []
    for func in call_graph:
        caller = func["caller"]
        if func_similarity.get(caller["funcname"]) is None:
            continue
        if func_similarity[caller["funcname"]] <= config.inference_threshold:
            continue

        pointer_arg_num = 0 # number of pointer type in return value and parameters.
        return_type = caller["return_type"]
        params = caller["params"]
        if normalize_type_1(return_type) != "<noptr>":
            pointer_arg_num += 1
        for argtype, argname in parse_params(params):
            if argtype != "<noptr>":
                pointer_arg_num += 1
                break
        if pointer_arg_num == 0:
            continue
        candidate_function_set.append(func)

    with open(config.candidate_alloc_path, "w") as f:
        for func in candidate_function_set:
            caller = func["caller"]
            funcname = caller["funcname"]
            f.write(funcname + "\n")
    return candidate_function_set


def get_target_callees(func_similarity, func, belief_bitmaps):
    """
    得到这个函数所调用的allocation 函数，作为之后数据流跟踪的目标函数。
    Get the allocation function that current func calledf,
    then, these function are the targets to track the data flow.
    :param func_similarity:
    :param func:
    :return:
    """
    target_callees = []
    callees = func["callees"]
    for callee in callees:
        func_name = callee["funcname"]
        if func_name == "malloc":
            target_callees.append(callee)
            break
        if belief_bitmaps.get(func_name) is None:
            continue
        if belief_bitmaps[func_name] == 1:
            target_callees.append(callee)
            continue
        if func_similarity.get(func_name) is None:
            continue
        ## 目标callee的函数返回类型应当为指针类型
        if normalize_type_1(callee["return_type"]) == "<noptr>":
            continue
        ## 目标callee的相似性分数应当大于一个阈值
        if func_similarity[func_name] < config.inference_threshold:
            continue
        target_callees.append(callee)
    return target_callees

def dedup_memory_flow(in_file):
    memory_flow = {}
    result = []
    with open(in_file,"r") as f:
        for line in f.readlines():
            func = json.loads(line)
            if len(func["passed_params"]) == 1 and len(func["return_values"]) == 1 and func["direct_return"] == 0:
                continue
            caller_name = func["caller"]
            if caller_name == "create_cache":
                a = 1
            if memory_flow.get(caller_name) == None:
                memory_flow[caller_name] = []
            callee_name = func["callee"]
            if callee_name not in memory_flow[caller_name]:
                memory_flow[caller_name].append(callee_name)
                result.append(line)
    with open(in_file,"w") as f:
        f.writelines(result)


def write_allocation_set(allocation_set, out_path="temp/allocation_set"):
    with open(out_path, "w") as f:
        for func in allocation_set:
            caller = func["caller"]
            f.write(caller["funcname"] + "\n")
            callees = func["target_callees"]
            for callee in callees:
                f.write(callee["funcname"] + "\n")
            f.write("-\n")

def load_memory_flow(in_file):
    dedup_memory_flow(in_file)
    global memory_flows
    with open(in_file, "r") as f:
        for line in f.readlines():
            func_flow = json.loads(line)
            caller_name = func_flow["caller"]
            if memory_flows.get(caller_name) is None:
                memory_flows[caller_name] = []
            memory_flows[caller_name].append(func_flow)
    print("load_memory_flow finished!")


def load_func_name_similarity(in_path="temp/func_name_similarity"):
    func_similarity = {}
    with open(in_path, "r") as f:
        for line in f.readlines():
            func_name_split = line.strip().split()
            if len(func_name_split) > 2:
                func_name = "".join(func_name_split[:-1])
                cos = func_name_split[-1]
            else:
                func_name,cos = func_name_split
            func_similarity[func_name] = float(cos)
    print("function similarity about name finished!")
    return func_similarity


def get_belief_func_bitmap(func_similarity, belief_functions):
    """
    We maintain a bit flag for each function, where +1 means is a belief function,
    0 means this function is unknown, -1 means is not a belief function. 2 used for recursion check.
    :param func_similarity:
    :param belief_functions:
    :return:
    """
    bitmaps = {}
    for func_name in func_similarity:
        bitmaps[func_name] = 1 if func_name in belief_functions else 0
    return bitmaps


def check_callee_is_belief(callees, belief_bitmaps):
    for callee in callees:
        func_name = callee["funcname"]
        if belief_bitmaps[func_name] == 1:
            return callee
    return None


def get_beleif_callee_flows(caller_name,belief_callees):
    callee_flows = memory_flows[caller_name]
    new_callee_flows = []
    for callee_flow in callee_flows:
        callee_name = callee_flow["callee"]
        for belief_callee in belief_callees:
            belief_callee_name = belief_callee["funcname"]
            if callee_name == belief_callee_name:
                new_callee_flows.append(callee_flow)
    return new_callee_flows


def is_abnormal_alloc(funcname):
    if funcname in param_function_name_set or funcname in return_function_name_set:
        return True
    return False


def check_caller_is_belief(func_similarity, func, belief_bitmaps, call_graph):
    """
    1. 对于每个函数，看它是否直接调用强可信函数
    2. 如果直接调用强可信函数，则对此函数进行check 3
    3. 如果没有直接调用强可信函数，则对包含的allocation函数进行递归验证。
    :param call_graph:
    :param func_similarity:
    :param func:
    :param belief_bitmaps:
    :return:
    """
    caller = func["caller"]
    caller_func_name = caller["funcname"]

    # 若函数自身已经是强可信函数，则不需要再次验证
    if belief_bitmaps[caller_func_name] == 1:
        return

    # 这里为每个正在进行查询的，函数标记为2，这样在遍历函数调用链的过程中，如果遇到一个已经为2的，那么说明这个函数存在递归调用。
    belief_bitmaps[caller_func_name] = 2

    # 获得此函数调用的目标allocation函数
    if func.get("target_callees") is not None:
        callees = func["target_callees"]
    else:
        callees = get_target_callees(func_similarity,func,belief_bitmaps)

    for target_callee in callees:
        # 解决递归调用，认为递归的函数都是不可信的。
        if target_callee["funcname"] == caller_func_name or belief_bitmaps[target_callee["funcname"]] ==2:
            belief_bitmaps[caller_func_name] = -1
            return
        # 如果目标allocation函数在call graph中缺失的话，那么将callee标记为不可信函数
        if call_graph.get(target_callee["funcname"]) is None:
            #belief_bitmaps[caller["funcname"]] = -1
            belief_bitmaps[target_callee["funcname"]] = -1
            continue

        # 若没有经过验证，则需要验证callee是否为belief function
        while belief_bitmaps[target_callee["funcname"]] == 0:
            callee_func = call_graph[target_callee["funcname"]]
            check_caller_is_belief(func_similarity, callee_func, belief_bitmaps, call_graph)
            pass  # iteration

    #如果所有的target_callee都是不可信的，那么caller也是不可信的。
    belief_callees = []
    for target_callee in callees:
        callee_belief_flag = belief_bitmaps[target_callee["funcname"]]
        assert callee_belief_flag == 1 or callee_belief_flag == -1, "callee belief flag must be -1 or +1."
        if callee_belief_flag == 1:
            belief_callees.append(target_callee)
        # 如果这个callee函数，我们早已确定，是那种非常规类型的allocation函数，那么这个caller就不应当再为常规allocation函数
        if is_abnormal_alloc(target_callee["funcname"]) is True:
            belief_bitmaps[caller_func_name] = -1
            return
    if len(belief_callees) == 0:
        belief_bitmaps[caller_func_name] = -1
        return

    # 此时，一定存在belief的callee。那么对这些可信的callee进行check 3
    ret = belief_chain_check_3(caller,belief_callees)
    if ret is True:
        belief_bitmaps[caller_func_name] = 1
        return
    else:
        belief_bitmaps[caller_func_name] = -1


def initial_special_alloc_function(func_similarity, call_graph):
    """
    识别得到各项目自定义的特殊原语allocators的函数。 筛选策略如下：
    1. 返回类型为指针, return a pointer
    2. 相似度大于一个阈值, sim_alloc is greater than a threshold
    3. 如果返回类型为一个结构体，内部不能调用超过两个allocation 函数
    4. 如果内部需调用alloc函数, should call an allocation function
    :param in_file:
    :return:
    """
    special_alloc_function = []

    for func in call_graph:
        caller = func["caller"]

        flag = 0
        # firstly, check the return type of this caller function whether is '<ptr>'
        if normalize_type_1(caller["return_type"]) != '<ptr>':
            continue

        #secondly, check the parameters of this caller function whether contains '<ptr>' or '<dptr>'.
        # Here "kmem_cache" is a special case in Linux Kernel. Such as "kmem_cache_alloc" has a pointer-type parameter "struct kmem_cache*"
        for raw_argtype,arg_type, arg_name in parse_params(caller["params"], arg_normalize=False):
            if (arg_type == '<ptr>' and "kmem_cache" not in raw_argtype) or arg_type == '<dptr>':
                flag += 1
                break
        if flag > 1:
            continue

        callees = func["callees"]
        if len(callees) == 0:
            continue

        # thirdly, check the similarity score of this caller function whether is large enough.
        if func_similarity.get(caller['funcname']) == None:
            continue
        if func_similarity[caller['funcname']] <= config.strong_belief_threshold:
            continue


        # forth, if this function called no allocation functions, filter out it.
        alloc_callee_numbers = 0
        for callee in callees:
            if func_similarity[callee['funcname']] >= config.inference_threshold:
                alloc_callee_numbers += callee['number']
        if alloc_callee_numbers < 1:
            continue

        # forth, if the return pointer is a structure object, then check whether it called alloc function more than twice.
        if "struct" in caller["return_type"]:
            if alloc_callee_numbers != 1:
                continue

        # finally, all checks are passed.
        special_alloc_function.append(caller["funcname"])
    print("initial_strong_belief_alloc_function finished!")
    return special_alloc_function


def getBaseName(param):
    if param.find("-") == -1 and param.find(".") == -1:
        return None
    basename = ""
    for c in param:
        if c == "-" or c == ".":
            break
        basename += c
    if param.startswith("&"):
        return basename[1:]
    else:
        return basename


def isMemberName(param):
    if param.find("-") != -1 or param.find(".") != -1:
        return True
    return False


def getMemberName(param):
    arrow_index,dot_index = param.find("-"), param.find(".")
    if arrow_index != -1 and dot_index !=-1:
        index = min(arrow_index,dot_index)
    else:
        index = max(param.find("-"),param.find("."))
    if index == -1:
        return param
    return param[index:]


def check_return_and_param_is_equal(callee_flow,params):
    """
    来检查函数的返回值在这个参数的集合中有没有一样的，如果有，那么就忽略分配到参数中的内存。
    :param callee_flow:
    :param params:
    :return:
    """
    return_values = callee_flow["return_values"][:-1]
    arg_names = []
    for arg in parse_params(params, normalize=False):
        arg_type, arg_name = arg
        arg_names.append(arg_name)
    for return_value in return_values:
        if return_value in arg_names:
            return True
    return False


invalidate_list = ["kfree_skb_list", "sock_release"]

def initial_candidate_free_function(func_similarity, call_graph):
    """
    得到关于free的(候选)函数，选择策略如下；
    1. 相似性分数大于一个阈值, similarity score greater than a threshold.
    2. 函数返回类型为void 或非指针类型, the return type is non-pointer type.
    3. 函数包含参数且为指针, has at least a parameter with pointer type.
    4. 函数实现中至少调用了一个高于相似性阈值的函数, function body has called at least one function with similarity greater than threshold.
    :param func_similarity:
    :param call_graph:
    :return:
    """
    experiment_candidates = [] # Only for recording experiment data
    candidate_free_functions = []
    for func in call_graph:
        caller = func["caller"]
        caller_funcname = caller["funcname"]

        # 相似性分数大于一个阈值
        if func_similarity.get(caller_funcname) == None:
            continue
        if func_similarity[caller_funcname] <= config.inference_threshold:
            continue

        # 函数返回类型为非指针类型
        if normalize_type_1(caller["return_type"]) in ["<ptr>", "<dptr>"]:
        #if "void" not in caller["return_type"]:
            continue

        # 函数至少包含一个指针参数
        flag = 0
        params = parse_params(caller["params"])
        if len(params) < 1:
            continue
        for arg_type, arg_name in params:
            if arg_type in ["<ptr>","<dptr>"]:
                flag = 1
        if flag == 0:
            continue
        experiment_candidates.append(caller_funcname)

        # 函数中调用了一个及以上的free函数
        free_number = 0
        callees = func["callees"]
        new_callees = []
        for callee in callees:
            callee_funcname = callee["funcname"]
            if func_similarity.get(callee_funcname) == None:
                continue
            if func_similarity[callee_funcname] > config.inference_threshold:
                free_number += callee["number"]
                new_callees.append(callee)
        if free_number < 1:
            continue

        if caller_funcname in invalidate_list:
            continue
        # 通过所有的验证
        new_func = {"caller": caller, "callees": new_callees}
        candidate_free_functions.append(new_func)

    with open(config.candidate_free_path, "w") as f:
        for func in candidate_free_functions:
            caller = func["caller"]
            func_name = caller["funcname"]
            f.write(func_name + "\n")
    with open("temp/experiment_candidate_free.txt", "w") as f:
        f.write("\n".join(experiment_candidates))
    return candidate_free_functions


def call_chain_check_1(func_similarity, allocation_set, belief_bitmaps):
    """
    Check the input allocation functions, whether they call a allocation function.
    If call, return.
    检查输入的这些allocation函数，它们是否调用了其他的allocation函数，如果没有那么说明这些allocation函数不应当返回。
    否则，返回这些经过check的函数。
    :param func_similarity:
    :param allocation_set:
    :return:
    """
    new_allocation_set = []
    for func in allocation_set:
        target_callees = get_target_callees(func_similarity, func, belief_bitmaps)
        if len(target_callees) == 0 or belief_bitmaps[func["caller"]["funcname"]] == 1:
            continue
        # passed the check, and then return them.
        new_func = {"caller": func["caller"], "target_callees": target_callees}
        new_allocation_set.append(new_func)
    print("belief_chain_check_1 finished!")
    return new_allocation_set


def call_chain_check_2(func_similarity, call_graph, allocation_set, belief_bitmaps):
    """
    在这一步的验证中，我们将进行如下步骤：
    1. 对于每个函数，看它是否直接调用强可信函数
    2. 如果直接调用强可信函数，则对此函数进行check 3
    3. 如果没有直接调用强可信函数，则对包含的allocation函数进行递归验证。
    :param call_graph:
    :param func_similarity:
    :param allocation_set:
    :param belief_functions:
    :return:
    """
    call_graph = convert_call_graph_to_dict(call_graph)
    for func in allocation_set:
        check_caller_is_belief(func_similarity, func, belief_bitmaps, call_graph)
    return belief_bitmaps



def belief_chain_check_3(caller,belief_callees):
    """
    如果目标函数调用了强可信函数，则需要进行此check，来确定此函数的操作对象。
    此check由如下几个步骤组成：
    1. 获得强可信函数的返回值、返回值经过的变量集合。
    2. 如果目标函数的返回值，直接调用强可信函数进行返回，若目标函数返回值为指针类型，则归类目标函数为强可信函数。
    3. 根据目标函数的返回值类型，检查强可信函数经过的变量集合是否与参数列表或者返回值存在交集。若存在交集，则此函数也为强可信函数。
    :return:
    """
    func_name = caller["funcname"]
    # 如果在clang收集的信息流中，没有找对对应的，那么认为此函数不可信
    if memory_flows.get(func_name) is None:
        return False

    returned_memory_set = []
    param_memory_set = []
    caller_flows = get_beleif_callee_flows(func_name,belief_callees)
    if len(caller_flows) ==0:
        return False
    arg_return_equal_flag = check_return_and_param_is_equal(caller_flows[0], caller["params"])
    has_matched_return_flag = 0
    for caller_flow in caller_flows:
        # 如果存在一个直接返回内存的callee，那么就认为这个caller也通过返回值返回
        if caller_flow["direct_return"] == 1:
            return True

        # 看这个caller函数，分配的内存是否在返回值的集合中找到
        passed_params = caller_flow["passed_params"][:-1]
        return_values = caller_flow["return_values"][:-1]
        if normalize_type_1(caller["return_type"]) != "<noptr>":
            for passed_param in passed_params:
                if passed_param in return_values:
                    has_matched_return_flag = 1

        # 如果返回值是一个成员变量，且和内存数据流遍历的参数一致，那么直接认为True
        #for return_value in return_values:
        #    if isMemberName(return_value):
        #        if return_value in passed_params:
        #            return True
        return_values = [getBaseName(x) for x in return_values if isMemberName(x) and x.startswith("&")] + [x for x in return_values if not (isMemberName(x) and x.startswith("&"))]

        # 记录下这个caller函数，通过返回值，返回的额外成员空间变量
        for passed_param in passed_params:
            basename = getBaseName(passed_param)
            if basename is None:
                continue
            if basename in return_values and normalize_type_1(caller["return_type"]) == "<ptr>":
                function = (func_name,passed_param)
                return_function_name_set.add(func_name)
                returned_memory_set.append(function)
                return_function_list.append(function)

        # 如果存在参数和返回值一样的话，那么就忽略参数返回的对象。因为这种场景下，参数的存在仅是个摆设
        if arg_return_equal_flag:
            continue

        # 记录下这个caller函数，通过传参形式，申请的内存对象变量
        for passed_param in passed_params:
            basename = getBaseName(passed_param)
            for index, arg in enumerate(parse_params(caller["params"], normalize=False)):
                arg_type, arg_name = arg
                if basename == arg_name or (passed_param == arg_name and arg_type == "<dptr>"):
                    param_function = (func_name, index + 1, passed_param)
                    param_function_name_set.add(func_name)
                    param_memory_set.append(param_function)
                    param_function_list.append(param_function)

    # 当且仅当 通过返回值返回，且没有其他额外的内存对象分配时，返回True。
    if has_matched_return_flag ==1 and len(returned_memory_set) == 0 and len(param_memory_set) == 0:
        return True
    return False


def generate_primitve_deallocators(call_graph, func_similarity, min_call, min_reset):
    func_freq = {}
    with open("temp/Dealloc_Number.txt", "r") as f:
        for line in f.readlines():
            func, freq = line.strip().split()[0],line.strip().split()[-1]
            if line.find(" ") == -1:
                continue
            func_freq[func] = int(freq)

    belief_topk = get_belief_topk(func_freq,min_reset=min_reset)
    func_freq = primitive_deallocator(func_freq,call_graph,func_similarity,min_call=min_call)
    #func_freq["dev_kfree_skb"] = 1260
    sorted_func_freq = sorted(func_freq.items(), key=lambda d: d[1], reverse=True)
    sorted_func_freq,truth_funcs = filter_out_untruth(sorted_func_freq,min_reset=min_reset,topk=belief_topk)
    with open("temp/primitive_deallocator","w") as f:
        for funcname,freq in sorted_func_freq:
            line = funcname.ljust(50, " ") + str(freq)
            f.write(line + "\n")

    with open("temp/seed_free.txt","w") as f:
        f.writelines(truth_funcs)


def count_free_call_site(candidate_functions, call_graph):
    """
    上一步会返回符合规则的candidate free函数。这里统计每个candidate free函数出现的次数。

    :param candidate_functions:
    :param call_graph:
    :return:
    """
    func_freq = {}
    # Initialize the frequency of each candidate free function as 1.
    for func in candidate_functions:
        caller = func["caller"]
        funcname = caller["funcname"]
        if "unregister" in funcname or "remove" in funcname or "cleanup" in funcname or "unpin" in funcname:
            continue
        func_freq[funcname] = 1

    # Walking the call graph and count the frequency.
    for func in call_graph:
        callees = func["callees"]
        for callee in callees:
            callee_name = callee["funcname"]
            number = callee["number"]
            if func_freq.get(callee_name) is None:
                continue
            func_freq[callee_name] += number

    sorted_func_freq = sorted(func_freq.items(), key=lambda d: d[1], reverse=True)

    with open("temp/Dealloc_Number.txt", "w") as f:
        for funcname, freq in sorted_func_freq:
            line = funcname.ljust(50," ") + str(freq)
            f.write(line + "\n")
    return func_freq


def get_belief_topk(func_freq,min_reset):
    top10_ratio = []
    func_freq["dev_kfree_skb"] = 1270
    with open(config.free_check_file, "r") as f:
        for i,line in enumerate(f.readlines()):
            line = line.strip().split()
            func = line[0]
            count = line[2]
            if i ==10:
                break
            nr_call = func_freq[func]
            ratio = int(count) / nr_call
            top10_ratio.append(ratio)
    min_ratio = min(top10_ratio)
    topk = min_reset/min_ratio
    return int(topk)


def filter_out_untruth(func_freq,min_reset,topk):
    func_null_freq  = {}
    func_argindex = {}
    with open("temp/free_check.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split()
            func = line[0]
            arg_index = line[1]
            count = line[2]
            func_null_freq[func] = int(count)
            func_argindex[func] = int(arg_index)

    untruth = []
    truth = []
    for funcname,nr_call in func_freq:
        if nr_call <=topk:
            break
        if func_null_freq.get(funcname) is None:
            untruth.append(funcname)
            print(funcname)
            continue
        if func_null_freq[funcname] < min_reset:
            untruth.append(funcname)
            print(funcname)
            continue
        truth_func_string = funcname + "\t" + str(func_argindex[funcname]) + "\n"
        truth.append(truth_func_string)

    new_func_freq = []
    for func,freq in func_freq:
        if func in untruth:
            continue
        new_func_freq.append((func,freq))
    return new_func_freq,truth


def parse_param_reassignment(file):
    func = {}
    if not os.path.exists(file):
        return func
    with open(file,"r") as f:
        for line in f.readlines():
            funcname,index,number = line.split()
            func[funcname] = (int(index), int(number))
    return func


def special_deallocator_identification(call_graph, func_similarity, free_check_file, min_reassignment=10):
    """
    根据通过语义识别出来的释放函数和其被调用次数，寻找并扩充其中的每个项目中独自实现的原语deallocator。
    规则：
    1. 返回值类型为void
    2. 相似性大于strong_threshold
    3. 被确定的参数reassignment次数大于min_reassignemnt
    4. 被确定的参数，返回类型为struct,则只调用一次deallocation 函数。
    :param func_freq:
    :param call_graph:
    :return:
    """
    decide_minrest()
    func_reassign = parse_param_reassignment(free_check_file)
    deallocators = []
    func_freq = {}
    with open("temp/Dealloc_Number.txt", "r") as f:
        for line in f.readlines():
            func, freq = line.strip().split()[0],line.strip().split()[-1]
            if line.find(" ") == -1:
                continue
            func_freq[func] = int(freq)

    for funcname, sim in func_similarity.items():
        #2. 相似性大于strong_threshold
        if sim <= config.strong_belief_threshold:
            continue

        #1.返回值类型为void
        if call_graph.get(funcname) is None:
            continue
        caller = call_graph[funcname]["caller"]
        ret_type = caller["return_type"]
        if "void" not in ret_type:
            continue

        #3.被确定的参数reassignment次数大于min_reassignemnt
        if func_reassign.get(funcname) is None:
            continue
        index, reassign_num = func_reassign[funcname]
        if reassign_num < min_reassignment:
            continue

        #4.被确定的参数，返回类型为struct, 则只调用一次deallocation函数。
        callees = call_graph[funcname]["callees"]
        dealloc_num = 0
        for callee in callees:
            callee_name = callee["funcname"]
            if func_similarity.get(callee_name) is None:
                continue
            if func_similarity[callee_name] >= config.inference_threshold:
                dealloc_num += callee["number"]
        if dealloc_num == 0:
            continue

        params = caller["params"]
        param_list = params.split(",")
        if index > len(param_list):
            continue
        determined_param = param_list[index]
        if "struct" in determined_param or dealloc_num > 1:
            continue

        ## 额外的验证： 过滤因为数据流引擎不精确导致的极个别误报
        call_times = func_freq[funcname]
        call_times = int(call_times)
        if 2*reassign_num >= call_times and reassign_num < min_reassignment+10:
            continue
        deallocators.append([funcname, index, reassign_num])
    new_map = sorted(deallocators, key=lambda d: d[2], reverse=True)
    with open("temp/special_deallocator.txt","w") as f:
        for func,index,num in new_map:
            f.write(func + "\t" + str(index) + "\t" + str(num) + "\n")
    return new_map


def primitive_deallocator(func_freq,call_graph,func_similarity,min_call=50):
    """
    根据通过语义识别出来的释放函数和其被调用次数，寻找其中的原语deallocator。
    规则：
    1. 函数调用的释放函数不在freq>min_call的的函数列表中
    2. 函数的参数被重新赋值或置空的次数>min_null
    :param func_freq:
    :param call_graph:
    :return:
    """
    not_primitive_funcs = []
    call_graph_map = convert_call_graph_to_dict(call_graph)
    for funcname,freq in func_freq.items():
        if call_graph_map.get(funcname) is None:
            continue
        callees = call_graph_map[funcname]["callees"]
        for callee in callees:
            callee_name = callee["funcname"]
            if func_similarity.get(callee_name) is None:
                continue
            sim = func_similarity[callee_name]
            ## 不是释放函数则直接略过
            if sim < config.inference_threshold:
                continue
            if func_freq.get(callee_name) is None:
                continue
            callee_freq = func_freq[callee_name]
            # 如果调用了一个释放函数，且此释放函数出现的频率很高，那么此函数就不是原语函数
            if callee_freq > min_call:
                not_primitive_funcs.append(funcname)
                break
    new_func_freq = {}
    for funcname,freq in func_freq.items():
        if funcname in not_primitive_funcs:
            continue
        new_func_freq[funcname] = freq
    return new_func_freq


def count_belief_function_free(belief_functions,call_graph):
    """
    根据Callee出现的次数，寻找原语deallocators，并且统计出现频率
    规则：
    1.  callee的返回值为'void'
    2.
    :param belief_functions:
    :return:
    """
    func_number = {}
    for func in belief_functions:
        callees = func["callees"]
        for callee in callees:
            callee_name = callee["funcname"]
            callee_params = callee["params"]
            return_type = callee["return_type"]
            if "void" not in return_type:
                continue
            params = parse_params(callee_params)
            flag = 0
            for param in params: # 参数是否全部是指针
                arg_type, arg_name = param
                if arg_type != "<ptr>":
                    flag = 1
            if flag == 1:
                continue
            param_num = len(parse_params(callee_params))
            if param_num == 0:
                continue
            if func_number.get(callee_name) is None:
                func_number[callee_name] = 1
            else:
                number = callee["number"]
                func_number[callee_name] += number
    func_number = get_seed_file(func_number,call_graph)
    func_number = sorted(func_number.items(), key=lambda d: d[1], reverse=True)
    with open("temp/belief_free_function_number", "w") as f:
        for func_name, number in func_number:
            if "unregister" in func_name:
                continue
            f.write(func_name + "\t" + str(number) + "\n")


def get_seed_file(func_number, call_graph):
    """
    这里你可以添加自己想要的规则，或者过滤某些函数
    :param func_number:
    :param call_graph:
    :return:
    """
    call_graph = convert_call_graph_to_dict(call_graph)
    prev_func_list = [funcname for funcname,num in func_number.items() if num >=50]
    filter_list = []
    for func_name ,num in func_number.items():
        if num < 30:
            continue
        if call_graph.get(func_name) is None:
            continue
        func = call_graph[func_name]
        callees = func["callees"]
        for callee in callees:
            callee_name = callee["funcname"]
            if callee_name in prev_func_list:
                filter_list.append(func_name)
                break
    seed_func = {}
    for func_name, num in func_number.items():
        if func_name in filter_list:
            continue
        seed_func[func_name] = num
    return seed_func


def generate_seed_free(sp_deallocators):
    seeds = []
    for funcname, index, number in sp_deallocators:
        seeds.append([funcname,index])
    with open(config.seed_free_path, "w") as f:
        with open("subword_dataset/official_deallocator.txt", "r") as f_r:
            f.write(f_r.read() + "\n")
        for seed in seeds:
            funcname,index = seed[0],seed[1]
            f.write(funcname + "\t" + str(index) + "\n")


def generate_seed_allocators(belief_function):
    seed_allocators = set()
    with open("subword_dataset/official_allocator.txt", "r") as f:
        for func in f.readlines():
            seed_allocators.add(func.strip())
        for func in belief_function:
            seed_allocators.add(func)
    return list(seed_allocators)


def run_free(call_graph_file, step=1):
    """
        Step 0: Only debug use, do not need to perform similarity inference from the beginning,
            directly load the similarity from file.
        Step 1: Perform Similarity inference for each function,
            to infer the similarity scores with allocation/deallocation functions.
        Step 2:
    """
    if step == 1:
        model = tf.keras.models.load_model(os.path.join(config.model_dir, "free", "maxauc_model"))
        _ = get_all_funcs(call_graph_file, "temp/extract_all_func")
        func_similarity = working_on_json_function_prototype(model, "temp/extract_all_func", "free")
    else:
        func_similarity = load_func_name_similarity()
    call_graph = read_caller_and_callee(call_graph_file)

    if step == 2:
        call_graph = convert_call_graph_to_dict(call_graph)
        sp_funcs = special_deallocator_identification(call_graph, func_similarity, config.free_check_file, config.min_reset)
        generate_seed_free(sp_funcs)
        call_chains = MMD_call_chains(call_graph)
        return call_graph, call_chains
    # step1 : Find the initial strong belief deallocation functions
    candidate_functions = initial_candidate_free_function(func_similarity, call_graph)
    count_free_call_site(candidate_functions, call_graph)


def write_final_alloc_result(belief_bitmaps, out_dir = "output/alloc/"):
    belief_functions = []
    for k, v in belief_bitmaps.items():
        if v == 1:
            belief_functions.append(k)
    # 将那些通过返回值返回，且不额外分配空间的函数写入文件
    with open(out_dir + "AllocNormalFile.txt", "w") as f:
        for func in belief_functions:
            f.write(func + "\n")

    function_json = {}
    for func in return_function_list:
        func_name,param = func
        if function_json.get(func_name) is None:
            function_json[func_name] = {"funcname":func_name,"returned_object":[],"param_object":[]}
        member_name = getMemberName(param)
        if member_name not in function_json[func_name]["returned_object"]:
            function_json[func_name]["returned_object"].append(member_name)

    for func in param_function_list:
        func_name, index, param = func
        if function_json.get(func_name) is None:
            function_json[func_name] = {"funcname":func_name,"returned_object":[],"param_object":[]}
        member_name = getMemberName(param)
        if member_name not in function_json[func_name]["param_object"]:
            function_json[func_name]["param_object"].append(index)
            function_json[func_name]["param_object"].append(member_name)

    with open(out_dir + "AllocCustomizedFile.txt", "w") as f:
        for k,v in function_json.items():
            f.write(json.dumps(v) + "\n")


def run_alloc(in_file=config.call_graph_path, step=1):
    """
    1. 为所有的函数原型计算sim_alloc相似性分数
    2. 过滤掉sim_alloc < infer_similarity的函数
    3. 得到剩余函数的函数调用链
    4. 调用插件获得目标函数的数据流
    5. 合并过程间的数据流，并生成MOS
    :param in_file:
    :param memory_flow_file:
    :param step:
    :return:
    """
    import time
    if step == 1:
        start = time.time()
        model = tf.keras.models.load_model(os.path.join(config.model_dir, "alloc", "maxauc_model"))
        _ = get_all_funcs(in_file, "temp/extract_all_func")
        func_similarity = working_on_json_function_prototype(model, "temp/extract_all_func", "alloc")
        end = time.time()
        print("alloc similarity score generated time:%s"% (end - start))
    else:
        func_similarity = load_func_name_similarity()
    call_graph = read_caller_and_callee(in_file)
    candidate_set = get_candidate_alloc_function_set(func_similarity, call_graph)

    # step1 : Find the initial strong belief allocation functions
    belief_functions = initial_special_alloc_function(func_similarity, call_graph)
    seed_allocators = generate_seed_allocators(belief_functions)
    belief_bitmaps = get_belief_func_bitmap(func_similarity, seed_allocators)
    allocation_set = call_chain_check_1(func_similarity, candidate_set, belief_bitmaps)
    write_allocation_set(allocation_set)
    if step == 1:
        return

    if not os.path.exists(config.mos_alloc_outpath):
        write_final_alloc_result(belief_bitmaps)
        return
    # step2: Check whether the allocation functions are strong belief functions.
    load_memory_flow(config.mos_alloc_outpath)
    belief_bitmaps = call_chain_check_2(func_similarity, call_graph, allocation_set, belief_bitmaps)
    write_final_alloc_result(belief_bitmaps)



return_function_list = []
return_function_name_set = set()
param_function_list = []
param_function_name_set = set()


if __name__ == "__main__":
    run_alloc(config.call_graph_path, step=1)
    run_alloc(config.call_graph_path, step=2)
