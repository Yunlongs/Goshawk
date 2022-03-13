import os
import multiprocessing
import subprocess
from parse_call_graph import remove_dup_caller
import config


def shell(cmd):
    os.system(cmd)
    print(cmd)


def dir_shell(dir, cmd):
    os.chdir(dir)
    print(cmd)
    subprocess.call(cmd, timeout=120, shell=True)


def AddFreePluginArg(arg):
    return " -Xclang -plugin-arg-point-memory-free -Xclang " + arg


def FreePluginCmd(step):
    basic_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/MemoryDataFlowFree.so -Xclang -plugin " \
                "-Xclang point-memory-free ".format(config.plugin_dir)
    cmd = " ".join([basic_cmd, AddFreePluginArg(step), AddFreePluginArg(config.candidate_free_path),
                    AddFreePluginArg(config.seed_free_path),
                    AddFreePluginArg(config.mos_seed_path), AddFreePluginArg(config.mos_free_outpath),
                    AddFreePluginArg(config.visited_file_path)])
    return cmd + " "


def format_clang_command(plugin_dir, temp_dir, plugin, plugin_name, *plugin_args):
    arg_list = ["clang -fsyntax-only -Xclang -load -Xclang", plugin, "-Xclang -plugin -Xclang", plugin_name]
    for arg in plugin_args:
        arg_list.append("-Xclang -plugin-arg-" + plugin_name)
        arg_list.append("-Xclang")
        arg_list.append(arg)
    cmd = " ".join(arg_list).format(plugin_dir, temp_dir) + " "
    return cmd


def walking_compile_database(file, flag, plugin_dir=config.plugin_dir, temp_dir=config.temp_dir, next_setp_TU=None):
    assert flag == "extract-funcs" or flag == "point-memory-alloc" or flag == "point-memory-free-1" or flag == "point-memory-free-2" or flag == "free-check", "please choose the correct plugin name: print-fns or point-memory"
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    import json
    if next_setp_TU is None:
        with open(file, "r") as f:
            compile_database = json.load(f)
    else:
        compile_database = next_setp_TU
    for i, command in enumerate(compile_database):
        cmd = " ".join(command["command"].split(" ")[1:])
        if flag == "extract-funcs":
            new_cmd = format_clang_command(plugin_dir, temp_dir, "{0}/ExtractFunctionPrototypes.so", "extract-funcs", "{1}/call_graph.json", "{1}/indirect_call.json")
        elif flag == "point-memory-alloc":
            new_cmd = format_clang_command(plugin_dir, temp_dir, "{0}/MemoryDataFlow.so", "point-memory", "{1}/allocation_set", "{1}/memory_flow_alloc.json")
        elif flag == "point-memory-free-1":
            new_cmd = FreePluginCmd(step="1")
        elif flag == "point-memory-free-2":
            new_cmd = FreePluginCmd(step="2")
        else:
            new_cmd = "clang -fsyntax-only -Xclang -load -Xclang {0}/FreeNullCheck.so -Xclang -plugin -Xclang free-check -Xclang -plugin-arg-free-check -Xclang {1}/candidate_free.txt -Xclang -plugin-arg-free-check  -Xclang {1}/free_check.txt -Xclang -plugin-arg-free-check  -Xclang {2} ".format(
                plugin_dir, temp_dir, config.visited_file_path)

        new_cmd += cmd
        dir = command["directory"]
        pool.apply_async(dir_shell, (dir, new_cmd,))
        if i % 3000 == 0 and i != 0 and flag == "extract-funcs":
            pool.close()
            pool.join()
            call_graph_path = config.call_graph_path
            remove_dup_caller(call_graph_path, call_graph_path)
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.close()
    pool.join()


def delete_exist_file(file):
    if os.path.exists(file):
        os.remove(file)


def plugin_run(project_dir, flag, next_setp_TU=None):
    delete_exist_file(config.visited_file_path)
    os.system("touch %s" % config.visited_file_path)

    compile_database_file = project_dir + os.sep + "compilation.json"
    if not os.path.exists(compile_database_file):
        print("\ncompile database not exist! Please make sure that there is a compilation.json under the project "
              "directory!\n")
        exit(-1)
    walking_compile_database(compile_database_file, flag, next_setp_TU=next_setp_TU)


if __name__ == "__main__":
    cmd = format_clang_command(config.plugin_dir, config.temp_dir, "{0}/ExtractFunctionPrototypes.so", "extract-funcs", "{1}/call_graph.json", "{1}/indirect_call.json")
    print(cmd)
