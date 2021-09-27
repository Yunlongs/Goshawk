import os
import multiprocessing
import subprocess
from parse_call_graph import remove_dup_caller, to_json
import argparse
#project_dir = "/home/lyl/source_code/openssl"

def shell(cmd):
    os.system(cmd)
    print(cmd)

def dir_shell(dir,cmd):
    os.chdir(dir)
    print(cmd)
    #os.system(cmd)
    subprocess.call(cmd,timeout=120, shell=True)

#plugin_dir = "/home/lyl/plugins"

def parser_cmd(in_file, flag):
    assert flag == "print-fns" or flag == "point-memory" or flag == "point-memory-free-1" or "point-memory-free-2" or "free-check", "please choose the correct plugin name: print-fns or point-memory"
    with open(in_file, "r") as f:
        cmd = f.readline()
        try:
            cmd = cmd[cmd.index(".o.d") + 5:]
        except:
            print(cmd)
            return ""
        if cmd.find(".c") == -1:
            return ""
    if flag == "print-fns":
        new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/PrintFunctionNames.so -Xclang -plugin -Xclang print-fns -Xclang -plugin-arg-print-fns -Xclang  {0}/call_graph.json -Xclang -plugin-arg-print-fns -Xclang  {0}/indirect_call.json ".format(plugin_dir)
    elif flag == "point-memory":
        new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/MemoryDataFlow.so -Xclang -plugin -Xclang point-memory -Xclang -plugin-arg-point-memory -Xclang {0}/allocation_set -Xclang -plugin-arg-point-memory -Xclang {0}/memory_flow_alloc.json ".format(plugin_dir)
    elif flag == "point-memory-free-1":
        new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/MemoryDataFlowFree.so -Xclang -plugin -Xclang point-memory-free -Xclang -plugin-arg-point-memory-free -Xclang {0}/free_set.txt -Xclang -plugin-arg-point-memory-free -Xclang {0}/seed_free.txt -Xclang -plugin-arg-point-memory-free -Xclang {0}/memory_flow_free.json ".format(plugin_dir)
    elif flag == "point-memory-free-2":
        new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/MemoryDataFlowFree.so -Xclang -plugin -Xclang point-memory-free -Xclang -plugin-arg-point-memory-free -Xclang {0}/overlap_func.txt -Xclang -plugin-arg-point-memory-free -Xclang 2 -Xclang -plugin-arg-point-memory-free -Xclang {0}/memory_flow_free_checked.json ".format(plugin_dir)
    else:
        new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/FreeNullCheck.so -Xclang -plugin -Xclang free-check -Xclang -plugin-arg-free-check -Xclang {0}/free_funcs.txt -Xclang -plugin-arg-free-check  -Xclang {0}/free_check.txt ".format(plugin_dir)
    new_cmd += cmd
    return new_cmd


def walk_dir(in_dir, flag):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith(".cmd"):
                file_path = os.path.join(root, file)
                cmd = parser_cmd(file_path, flag)
                pool.apply_async(shell, (cmd,))
    pool.close()
    pool.join()



def walk_compile_database(file, flag, plugin_dir, temp_dir):
    assert flag == "print-fns" or flag == "point-memory" or flag == "point-memory-free-1" or flag =="point-memory-free-2" or flag == "free-check", "please choose the correct plugin name: print-fns or point-memory"
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    import json
    with open(file,"r") as f:
        compile_database = json.load(f)
    for i, command in enumerate(compile_database):
        cmd = " ".join(command["command"].split(" ")[1:])
        if flag == "print-fns":
            new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/PrintFunctionNames.so -Xclang -plugin -Xclang print-fns -Xclang -plugin-arg-print-fns -Xclang  {1}/call_graph.json -Xclang -plugin-arg-print-fns -Xclang  {1}/indirect_call.json ".format(
                plugin_dir, temp_dir)
        elif flag == "point-memory":
            new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/MemoryDataFlow.so -Xclang -plugin -Xclang point-memory -Xclang -plugin-arg-point-memory -Xclang {1}/allocation_set -Xclang -plugin-arg-point-memory -Xclang {1}/memory_flow_alloc.json ".format(
                plugin_dir, temp_dir)
        elif flag == "point-memory-free-1":
            new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/MemoryDataFlowFree.so -Xclang -plugin -Xclang point-memory-free -Xclang -plugin-arg-point-memory-free -Xclang {1}/free_set.txt -Xclang -plugin-arg-point-memory-free -Xclang {1}/seed_free.txt -Xclang -plugin-arg-point-memory-free -Xclang {1}/memory_flow_free.json ".format(
                plugin_dir, temp_dir)
        elif flag == "point-memory-free-2":
            new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/MemoryDataFlowFree.so -Xclang -plugin -Xclang point-memory-free -Xclang -plugin-arg-point-memory-free -Xclang {1}/overlap_func.txt -Xclang -plugin-arg-point-memory-free -Xclang 2 -Xclang -plugin-arg-point-memory-free -Xclang {1}/memory_flow_free_checked.json ".format(
                plugin_dir,temp_dir)
        else:
            new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/FreeNullCheck.so -Xclang -plugin -Xclang free-check -Xclang -plugin-arg-free-check -Xclang {1}/free_funcs.txt -Xclang -plugin-arg-free-check  -Xclang {1}/free_check.txt ".format(plugin_dir, temp_dir)

        new_cmd += cmd
        dir = command["directory"]
        pool.apply_async(dir_shell, (dir,new_cmd,))
        if i%1000 ==0 and i!=0 and flag == "print-fns":
            pool.close()
            pool.join()
            call_graph_path = temp_dir + os.sep + "call_graph.json"
            to_json(call_graph_path)
            remove_dup_caller(call_graph_path, call_graph_path)
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.close()
    pool.join()



def check_size():
    func_file = "{}/call_graph.json".format(plugin_dir)
    if not os.path.exists(func_file):
        return
    size = os.path.getsize(func_file)
    if size / 1000000000 > 4:
        to_json(func_file)
        remove_dup_caller(func_file, func_file)

def for_kernel(plugin_dir, temp_dir, project_dir, flag):
    #flag = "point-memory-free-2"
    if os.path.exists("/tmp/visited"):
        os.remove("/tmp/visited")
        f = open("/tmp/visited", "w")
        f.close()
    if os.path.exists("{0}/memory_flow_free.json".format(plugin_dir)):
        os.remove("{0}/memory_flow_free.json".format(plugin_dir))
    if os.path.exists("{0}/call_graph.json".format(plugin_dir)):
        os.remove("{0}/call_graph.json".format(plugin_dir))
    if os.path.exists("{0}/memory_flow_alloc.json".format(plugin_dir)):
        os.remove("{0}/memory_flow_alloc.json".format(plugin_dir))

    #in_dir = "/home/lyl/source_code/linux-5.9.10"
    in_dir = project_dir
    os.chdir(in_dir)
    path_list = os.listdir(in_dir)
    for path in path_list:
        p = os.path.join(in_dir, path)
        if os.path.isdir(p):
            print(p)
            walk_dir(p, flag)
            if flag == "print-fns":
                check_size()

def for_others(plugin_dir, temp_dir, project_dir, flag):
    #flag = "print-fns"
    if os.path.exists("/tmp/visited"):
        os.remove("/tmp/visited")
        f = open("/tmp/visited", "w")
        f.close()
    if os.path.exists("{0}/memory_flow_free.json".format(plugin_dir)):
        os.remove("{0}/memory_flow_free.json".format(plugin_dir))
    if os.path.exists("{0}/call_graph.json".format(plugin_dir)):
        os.remove("{0}/call_graph.json".format(plugin_dir))
    if os.path.exists("{0}/memory_flow_alloc.json".format(plugin_dir)):
        os.remove("{0}/memory_flow_alloc.json".format(plugin_dir))

    compile_database_file = project_dir + os.sep + "compilation.json"
    if not os.path.exists(compile_database_file):
        print("compile database not exist!")
    walk_compile_database(compile_database_file, flag, plugin_dir, temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSA data flow plugins.')
    parser.add_argument('plugin_dir', metavar='/home/lyl/plugins', type=str, nargs=1, default="/home/lyl/plugins",
                        help='The dir of plugins you are stored in.')
    parser.add_argument("project_dir", metavar="/xxx/linux-5.12", type=str, nargs=1,
                        help="The dir of project you want to analyze.")
    parser.add_argument("flag", type=str, nargs=1,
                        help="print-fns or point-memory or point-memory-free-1 or point-memory-free-2")
    parser.add_argument("isKernel", type=int, nargs=1, help="Whether this project is a huge project, such as kernel.")
    args = parser.parse_args()
    plugin_dir = args.plugin_dir[0]
    print("plugin_dir:", plugin_dir)
    project_dir = args.project_dir[0]
    print("project_dir:", project_dir)
    flag = args.flag[0]
    print("flag: ", flag)
    isKernel = args.isKernel[0]
    print("is Kernel: ",isKernel)
    temp_dir = plugin_dir

    if isKernel == 1:
        for_kernel(plugin_dir,temp_dir, project_dir,flag)
    else:
        for_others(plugin_dir,temp_dir, project_dir,flag)