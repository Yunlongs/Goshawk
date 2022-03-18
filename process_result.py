import json, os
from normalize import parse_params
from parse_call_graph import read_caller_and_callee


def deduplicate_dataflow(in_file, type=0):
    """
    对所有的函数进行去重，并且去掉最后一个"<nops>"
    :param in_file:
    :param type: 0: pop the last <nops>, 1: don't need to pop.
    :return:
    """
    if not os.path.exists(in_file):
        print("\n Warning. Goshawk have not obtained the data flows that the rest steps required."
              "That might be there is no customized MM functions in your project.")
        return -1
    funcs = {}
    with open(in_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line.strip())
            func_name = func["funcname"]
            if type == 0:
                func["param_names"].pop(-1)
                func["member_name"].pop(-1)
            if funcs.get(func_name) is None:
                funcs[func_name] = func
            else:
                func2 = funcs[func_name]
                func["param_names"] = remove_dup_list(func["param_names"] + func2["param_names"])
                func["member_name"] = remove_dup_list(func["member_name"] + func2["member_name"])
                funcs[func_name] = func

    with open(in_file, "w") as f:
        for func in funcs.values():
            f.write(json.dumps(func) + "\n")


def comparative_analysis(old_file, new_file):
    old_res = {}
    new_res = {}
    with open(old_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            funcname = func["funcname"]
            old_res[funcname] = line
    with open(new_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            funcname = func["funcname"]
            new_res[funcname] = line
    print("The new file has, but the old file not has!\n\n\n----------------------")
    for k, v in new_res.items():
        if old_res.get(k) is None:
            print(v)
    print("\n\n\nThe old file has,but the new file not has!\n\n\n------------------")
    for k, v in old_res.items():
        if new_res.get(k) is None:
            print(v)


def get_can_direct_use_deallocation(memory_flow_file="temp/free/memory_flow_free.json",
                                    seed_free_file="temp/free/seed_free.txt"):
    """
    这里我们来根据识别的Deallocation函数的结果，先仅返回那些能够被checker所直接使用的函数。
    即和Free规则一样的函数：只释放第一个参数所指向的对象，不进行其他额外对象的释放。
    :return:
    """
    return_results = []
    with open(memory_flow_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            if len(func["member_name"]) >= 1:
                continue
            if len(func["param_names"]) >= 3:
                continue
            param_names = func["param_names"]
            index = param_names[0]
            name = func["funcname"]
            return_results.append((index, name))
    with open("temp/seed_free.txt", "r") as f:
        for line in f.readlines():
            func, index = line.strip().split()
            return_results.append((index, func))
    with open(seed_free_file, "w") as f:
        for index, name in return_results:
            f.write(name + "\t" + str(index) + "\n")
    return return_results


def classify_alloc_data(param_file="temp/alloc/csa/AllocCustomizedFile.txt", out_dir="temp/alloc/classify"):
    only_return, only_param, return_param = 0, 0, 0
    for_param, for_param_member, param_member = 0, 0, 0
    Member_list = []
    Param_list = []
    Param_Member_list = []
    Param_and_Member_list = []
    Others = []
    with open(param_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            if len(func["returned_object"]) == 0:
                only_param += 1
            elif len(func["param_object"]) == 0:
                only_return += 1
                Member_list.append(line)
            else:
                return_param += 1
                Others.append(line)
            if len(func["returned_object"]) == 0 and len(func["param_object"]) != 0:
                params = func["param_object"]
                params = [x for x in params if type(x) == str]
                member_flag = 0
                param_flag = 0
                for param in params:
                    if "->" in param or "." in param:
                        member_flag = 1
                    else:
                        param_flag = 1
                if member_flag == 1 and param_flag == 1:
                    param_member += 1
                    Param_and_Member_list.append(line)
                elif member_flag == 1:
                    for_param_member += 1
                    Param_Member_list.append(line)
                elif param_flag == 1:
                    for_param += 1
                    Param_list.append(line)
    print("only_return:%s\nonly_param:%s\nreturn_param:%s\n" % (only_return, only_param, return_param))
    print("for param:%s\n for param member:%s\n param and member:%s" % (for_param, for_param_member, param_member))
    with open(out_dir + os.sep + "return_value_with_structure_object.txt", "w") as f:
        f.writelines(Member_list)
    with open(out_dir + os.sep + "escape_to_parameter.txt", "w") as f:
        f.writelines(Param_list)
    with open(out_dir + os.sep + "escape_to_parameter_members.txt", "w") as f:
        f.writelines(Param_Member_list)
    with open(out_dir + os.sep + "escape_to_parameter_and_parameter_members.txt", "w") as f:
        f.writelines(Param_and_Member_list)
    with open(out_dir + os.sep + "others.txt", "w") as f:
        f.writelines(Others)


def classify_free_data(memory_flow_file, out_dir="output/free/"):
    if not os.path.exists(memory_flow_file):
        return
    only_param, only_member, param_member = 0, 0, 0
    FreeNormalFuncs = []
    with open(memory_flow_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line.strip())
            funcname = func["funcname"]
            param_names = func["param_names"]
            member_names = func["member_name"]
            if len(param_names) > 2 or len(param_names) == 0 or len(member_names) != 0:
                continue
            index = param_names[0]
            if index == 0:
                FreeNormalFuncs.append(funcname)

    with open(out_dir + "FreeNormalFile.txt", "w") as f:
        for func in FreeNormalFuncs:
            f.write(func + "\n")
    FreeCustomized = []
    Parameters = []
    Parameter_Member = []
    Parameter_and_Member = []
    with open(memory_flow_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            if (func["funcname"] not in FreeNormalFuncs):
                FreeCustomized.append(line)
            if len(func["param_names"]) != 0 and len(func["member_name"]) != 0:
                param_member += 1
                Parameter_and_Member.append(line)
            elif len(func["param_names"]) != 0:
                only_param += 1
                Parameters.append(line)
            elif len(func["member_name"]) != 0:
                only_member += 1
                Parameter_Member.append(line)
                print(line.strip())
    with open(out_dir + "FreeCustomizedFile.txt", "w") as f:
        f.writelines(FreeCustomized)
    print("only param: %s \n only member :%s param_member:%s" % (only_param, only_member, param_member))


def get_next_iteration_funcs(mem_file, call_graph_file):
    """
    因为在我们第一轮迭代得到的memory free函数中，有些函数存在着迭代调用。
    比如说： mlx5e_encap_dealloc 函数还额外调用了 kvfree_call_rcu来释放内存。
    而且kvfree_call_rcu释放的内存也被我们在第一轮中标记了出来，所以我们最终要将
    kvfree_call_rcu释放的内存归还给mlx5e_encap_dealloc。
    因此，在下一轮迭代中，我们需要用kvfree_call_rcu和seed function的mos帮助mlx5e_encap_dealloc生成MOS。
    :param mem_file:
    :param call_graph_file:
    :return:
    """
    call_graph = read_caller_and_callee(call_graph_file)
    mem_funcs = {}
    # 读取memory_flow_free中所有的函数名
    with open(mem_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            funcname = func["funcname"]
            mem_funcs[funcname] = line
    overlap_res = []

    # 遍历call_graph
    for call in call_graph:
        caller = call["caller"]
        callees = call["callees"]
        caller_funcname = caller["funcname"]
        # 遇到一个free函数
        if mem_funcs.get(caller_funcname):
            overlap_callees = []
            # 遍历他的callee
            for callee in callees:
                callee_funcname = callee["funcname"]
                # 如果它的一个callee也在free函数列表中，那么就添加到overlap的列表中
                if mem_funcs.get(callee_funcname):
                    overlap_callees.append(mem_funcs[callee_funcname].strip())
            if len(overlap_callees) != 0:
                res = [caller_funcname] + overlap_callees
                overlap_res.append(res)
    with open("temp/overlap_func.txt", "w") as f:
        for res in overlap_res:
            f.writelines([x + "\n" for x in res])
            f.write("-\n")


def read_overlap_file(file):
    overlap_funcs = []
    with open(file, "r") as f:
        for line in f.readlines():
            if line.startswith("-") or line.startswith("{"):
                continue
            overlap_funcs.append(line.strip())
    print("overlap file read finished! total func: ", len(overlap_funcs))
    return overlap_funcs


def remove_overlap_from_memory_flow(memory_flow_file="temp/free/memory_flow_free.json",
                                    overlap_file="temp/free/overlap_func.txt"):
    """

    :param memory_flow_file:
    :param overlap_file:
    :return:
    """
    overlap_funcname = read_overlap_file(overlap_file)
    overlap_memory_flow = []
    new_memory_flow = []

    # 赌气memory_flow_free.json文件
    with open(memory_flow_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            funcname = func["funcname"]
            # 检测当前函数是否是存在overlap问题的函数
            if funcname in overlap_funcname:
                overlap_memory_flow.append(line)
            else:
                new_memory_flow.append(line)
    with open(memory_flow_file, "w") as f:
        f.writelines(new_memory_flow)
    return overlap_memory_flow


def remove_dup_list(name_list):
    entry_list = []
    for i in range(len(name_list)):
        if i % 2 != 0:
            continue
        index = name_list[i]
        name = name_list[i + 1]
        entry_list.append((index, name))

    appeared_list = []
    new_list = []
    for index, name in entry_list:
        if name in appeared_list:
            continue
        appeared_list.append(name)
        new_list.append(index)
        new_list.append(name)
    return new_list


def add_new_memory_flow(checked_file, memory_file, overlap_file):
    """
    在得到了新的checked 函数之后，将其和之前的函数流进行合并
    :param checked_file:
    :param memory_file:
    :param overlap_file:
    :return:
    """
    new_memory_flow = {}
    # 读取overlap checked file中所有的memory flow
    with open(checked_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            funcname = func["funcname"]
            new_memory_flow[funcname] = line
    overlap_memory_flow = remove_overlap_from_memory_flow(memory_file, overlap_file)
    final_memory_flow = []
    new_funcs = []
    ## 对存在overlap问题的函数进行合并
    for overlap_func in overlap_memory_flow:
        func_1 = json.loads(overlap_func)
        funcname = func_1["funcname"]
        if new_memory_flow.get(funcname):
            func_2 = json.loads(new_memory_flow[funcname])
            func_1["param_names"] += func_2["param_names"]
            func_1["param_names"] = remove_dup_list(func_1["param_names"])
            func_1["member_name"] += func_2["member_name"]
            func_1["member_name"] = remove_dup_list(func_1["member_name"])
            new_funcs.append(func_1)
            print(func_1)
        else:
            new_funcs.append(func_1)
    with open(memory_file, "a") as f:
        for func in new_funcs:
            string = json.dumps(func) + "\n"
            f.write(string)


def get_new_round_free(FreeNormalFile, FreeCustomiezdFile, AllFuncFile):
    """
    我们根据上一轮迭代出的结果，把那些只释放第一个参数的函数找出来，当作新一轮的种子
    :param FreeNormalFile:
    :param FreeCustomiezdFile:
    :param AllFuncFile:
    :return:
    """
    NewSeedFuncs = []
    AllFuncMap = {}
    with open(AllFuncFile, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            func_name = func["funcname"]
            params = func["params"]
            AllFuncMap[func_name] = params
    if FreeNormalFile is not None:
        with open(FreeNormalFile, "r") as f:
            for line in f.readlines():
                funcname = line.strip()
                if AllFuncMap.get(funcname) is None:
                    continue
                params = AllFuncMap[funcname]
                param_nums = len(parse_params(params, False))
                if param_nums == 1:
                    NewSeedFuncs.append(line)

    with open(FreeCustomiezdFile, "r") as f:
        for line in f.readlines():
            func = json.loads(line.strip())
            funcname = func["funcname"]
            if len(func["member_name"]) != 0:
                continue
            if len(func["param_names"]) != 1:
                continue
            if AllFuncMap.get(funcname) is None:
                continue
            params = AllFuncMap[funcname]
            param_nums = len(parse_params(params, False))
            if param_nums == 1:
                NewSeedFuncs.append(funcname + "\n")
    with open("temp/free/new_seed_file.txt", "w") as f:
        f.writelines(NewSeedFuncs)
    print("finished")


def get_added_free(new_round_file):
    add_func = []
    prev_func = {}
    with open("temp/free/memory_flow_free.json", "r") as f:
        for line in f.readlines():
            func = json.loads(line.strip())
            funcname = func["funcname"]
            prev_func[funcname] = line
    with open(new_round_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line.strip())
            funcname = func["funcname"]
            if prev_func.get(funcname) is None:
                add_func.append(line)

    with open("temp/free/memory_flow_free.json", "a") as f:
        f.writelines(add_func)


def get_initial_alloc():
    func_names = []
    with open("temp/whole_belief_call_graph", "r") as f:
        for line in f.readlines():
            if not line.startswith("\t"):
                func = json.loads(line.strip())
                funcname = func["funcname"]
                func_names.append(funcname)
    with open("temp/Primitive_Allocators", "w") as f:
        for funcname in func_names:
            f.write(funcname + "\n")


def analysis_result_free_Parameter():
    all_func_file = "temp/extract_all_func"
    funcname_param_map = {}
    with open(all_func_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            func_name = func["funcname"]
            params = func["params"]
            funcname_param_map[func_name] = params
    funcs = []
    unfuncs = []
    with open("temp/free/classify/only_release_param.txt", "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            funcname = func["funcname"].strip()
            param_names = func["param_names"]
            param_pairs = []
            if funcname_param_map.get(funcname) is not None:
                params = funcname_param_map[funcname]
                args = parse_params(params, False)
                for index, arg in enumerate(args):
                    arg_name = arg[1]
                    if arg_name in param_names:
                        pair = (index, arg_name)
                        param_pairs.append(str(pair))
                if len(param_pairs) != len(param_names):
                    print(line)
                    funcname = funcname.ljust(50, " ")
                    param_pairs = []
                    for param in param_names:
                        if len(args) == 1:
                            index = 0
                        else:
                            index = 99
                        pair = (index, param)
                        param_pairs.append(str(pair))
                    params_string = ", ".join(param_pairs)
                    func_strings = funcname + params_string + "\n"
                    unfuncs.append(func_strings)
                    continue
            else:
                funcname = funcname.ljust(50, " ")
                param_pairs = []
                for param in param_names:
                    index = 99
                    pair = (index, param)
                    param_pairs.append(str(pair))
                params_string = ", ".join(param_pairs)
                func_strings = funcname + params_string + "\n"
                unfuncs.append(func_strings)
                continue
            funcname = funcname.ljust(50, " ")
            params_string = ", ".join(param_pairs)
            func_strings = funcname + params_string + "\n"
            funcs.append(func_strings)
    with open("temp/free/classify/temp.txt", "w") as f:
        f.writelines(funcs)
        f.writelines(unfuncs)


def GetBaseName(string):
    new_string = ""
    for c in string:
        if c == '.' or c == '-':
            break
        new_string += c
    return new_string


def analysis_result_free_Parameter_Member():
    all_func_file = "temp/extract_all_func"
    funcname_param_map = {}
    with open(all_func_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            func_name = func["funcname"]
            params = func["params"]
            funcname_param_map[func_name] = params
    funcs = []
    unfuncs = []
    with open("temp/free/classify/only_release_param_member.txt", "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            funcname = func["funcname"].strip()
            member_names = func["member_name"]
            member_pairs = []
            if funcname_param_map.get(funcname) is not None:
                params = funcname_param_map[funcname]
                args = parse_params(params, False)
                for index, arg in enumerate(args):
                    arg_name = arg[1]
                    for member in member_names:
                        basename = GetBaseName(member)
                        if basename == arg_name:
                            pair = (index, member)
                            member_pairs.append(str(pair))
                if len(member_pairs) != len(member_names):
                    print(line)
                    funcname = funcname.ljust(50, " ")
                    param_pairs = []
                    for member in member_names:
                        if len(args) == 1:
                            index = 0
                        else:
                            index = 99
                        pair = (index, member)
                        param_pairs.append(str(pair))
                    params_string = ", ".join(param_pairs)
                    func_strings = funcname + params_string + "\n"
                    unfuncs.append(func_strings)
                    continue
            else:
                funcname = funcname.ljust(50, " ")
                param_pairs = []
                for member in member_names:
                    index = 99
                    pair = (index, member)
                    param_pairs.append(str(pair))
                params_string = ", ".join(param_pairs)
                func_strings = funcname + params_string + "\n"
                unfuncs.append(func_strings)
                continue
            funcname = funcname.ljust(50, " ")
            params_string = ", ".join(member_pairs)
            func_strings = funcname + params_string + "\n"
            funcs.append(func_strings)
    with open("temp/free/classify/temp.txt", "w") as f:
        f.writelines(funcs)
        f.writelines(unfuncs)


def analysis_result_free_Parameter_and_Member():
    all_func_file = "temp/extract_all_func"
    funcname_param_map = {}
    with open(all_func_file, "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            func_name = func["funcname"]
            params = func["params"]
            funcname_param_map[func_name] = params
    funcs = []
    unfuncs = []
    with open("temp/free/classify/release_both_param_and_param_member.txt", "r") as f:
        for line in f.readlines():
            func = json.loads(line)
            funcname = func["funcname"].strip()
            param_names = func["param_names"]
            param_pairs = []
            member_names = func["member_name"]
            member_pairs = []
            if funcname_param_map.get(funcname) is not None:
                params = funcname_param_map[funcname]
                args = parse_params(params, False)
                for index, arg in enumerate(args):
                    arg_name = arg[1]
                    if arg_name in param_names:
                        pair = (index, arg_name)
                        param_pairs.append(str(pair))
                    for member in member_names:
                        basename = GetBaseName(member)
                        if basename == arg_name:
                            pair = (index, member)
                            member_pairs.append(str(pair))
                if len(member_pairs) != len(member_names) or len(param_pairs) != len(param_names):
                    print(line)
                    funcname = funcname.ljust(50, " ")
                    param_pairs = []
                    member_pairs = []
                    for member in member_names:
                        if len(args) == 1:
                            index = 0
                        else:
                            index = 99
                        pair = (index, member)
                        member_pairs.append(str(pair))
                    for param in param_names:
                        if len(args) == 1:
                            index = 0
                        else:
                            index = 99
                        pair = (index, param)
                        param_pairs.append(str(pair))
                    params_string = ", ".join(param_pairs)
                    params_string = params_string.ljust(35, " ")
                    member_string = ", ".join(member_pairs)
                    func_strings = funcname + params_string + member_string + "\n"
                    unfuncs.append(func_strings)
                    continue
            else:
                funcname = funcname.ljust(50, " ")
                param_pairs = []
                member_pairs = []
                for member in member_names:
                    index = 99
                    pair = (index, member)
                    member_pairs.append(str(pair))
                for param in param_names:
                    index = 99
                    pair = (index, param)
                    param_pairs.append(str(pair))
                params_string = ", ".join(param_pairs)
                params_string = params_string.ljust(35, " ")
                member_string = ", ".join(member_pairs)
                func_strings = funcname + params_string + member_string + "\n"
                unfuncs.append(func_strings)
                continue
            funcname = funcname.ljust(50, " ")
            params_string = ", ".join(param_pairs)
            params_string = params_string.ljust(35, " ")
            member_string = ", ".join(member_pairs)
            func_strings = funcname + params_string + member_string + "\n"
            funcs.append(func_strings)
    with open("temp/free/classify/temp.txt", "w") as f:
        f.writelines(funcs)
        f.writelines(unfuncs)


def get_CSA_format(out_dir="output/free/"):
    funcs = []
    with open(out_dir + "FreeCustomizedFile.txt") as f:
        for line in f.readlines():
            func = json.loads(line.strip())
            member_list = func["member_name"]
            new_list = []
            for member in member_list:
                if type(member) is str:
                    new_list.append(member)
            func["member_name"] = new_list
            funcs.append(func)
    with open(out_dir + "FreeCustomizedFile.txt", "w") as f:
        for func in funcs:
            line = json.dumps(func) + "\n"
            f.write(line)


if __name__ == "__main__":
    checked_file = "temp/free/memory_flow_free_checked.json"
    memory_free_file = "temp/free/memory_flow_free.json"
    func_file = "subword_dataset/FreeBSD/call_graph.json"
    overlap_file = "temp/free/overlap_func.txt"
    AllFuncFile = "temp/extract_all_func"
    FreeNormalFile = "temp/free/FreeNormalFile.txt"
    FreeCustomiezdFile = "temp/free/FreeCustomizedFile.txt"

    classify_free_data()
    get_CSA_format()
