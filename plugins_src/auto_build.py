#!/usr/bin/python3
import argparse
import shutil, os, platform
from distutils.version import StrictVersion

if os.environ.get("LLVM_PATH") is not None:
    LLVM_PATH = os.environ["LLVM_PATH"]
    clang_dir = os.path.join(LLVM_PATH, "clang")
    build_dir = os.path.join(LLVM_PATH, "build")
else:
    parser = argparse.ArgumentParser(description='Automatically put the plugins source codes into the plugins directory in clang.')
    parser.add_argument("clang_dir", metavar="/xxx/llvm-project/clang", type=str, help="The path of the clang source code directory.")
    parser.add_argument("build_dir", metavar="/xxx/llvm-project/build", type=str, help="The build path of the llvm and clang source code.")
    args = parser.parse_args()
    clang_dir = args.clang_dir
    build_dir = args.build_dir
    
curr_dir = os.getcwd()

def delete_exist_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

if not os.path.exists(clang_dir):
    print("The clang directory does not exist!")
    exit(1)

if StrictVersion(platform.python_version()) >= StrictVersion("3.8.14"):
    shutil.copytree("ExtractFunctionPrototypes", os.path.join(clang_dir, "examples/ExtractFunctionPrototypes"), dirs_exist_ok=True)
    shutil.copytree("FreeNullCheck", os.path.join(clang_dir, "examples/FreeNullCheck"), dirs_exist_ok=True)
    shutil.copytree("MemoryDataFlow", os.path.join(clang_dir, "examples/MemoryDataFlow"), dirs_exist_ok=True)
    shutil.copytree("MemoryDataFlowFree", os.path.join(clang_dir, "examples/MemoryDataFlowFree"), dirs_exist_ok=True)
    shutil.copytree("GoshawkAnalyzer", os.path.join(clang_dir, "lib/Analysis/plugins/GoshawkAnalyzer"), dirs_exist_ok=True)
else:
    delete_exist_dir(os.path.join(clang_dir, "examples/ExtractFunctionPrototypes"))
    delete_exist_dir(os.path.join(clang_dir, "examples/FreeNullCheck"))
    delete_exist_dir(os.path.join(clang_dir, "examples/MemoryDataFlow"))
    delete_exist_dir(os.path.join(clang_dir, "examples/MemoryDataFlowFree"))
    delete_exist_dir(os.path.join(clang_dir, "lib/Analysis/plugins/GoshawkAnalyzer"))
    shutil.copytree("ExtractFunctionPrototypes", os.path.join(clang_dir, "examples/ExtractFunctionPrototypes"))
    shutil.copytree("FreeNullCheck", os.path.join(clang_dir, "examples/FreeNullCheck"))
    shutil.copytree("MemoryDataFlow", os.path.join(clang_dir, "examples/MemoryDataFlow"))
    shutil.copytree("MemoryDataFlowFree", os.path.join(clang_dir, "examples/MemoryDataFlowFree"))
    shutil.copytree("GoshawkAnalyzer", os.path.join(clang_dir, "lib/Analysis/plugins/GoshawkAnalyzer"))

def parse_line(line):
    if line.strip().startswith("add_subdirectory"):
        return line.split("(")[1].split(")")[0]
    else:
        return None

def update_cmake_file(file_path, plugin_name):
    with open(file_path, "r") as f:
        lines = f.readlines()
    mods = []
    flag = 0
    with open(file_path, "w") as f:
        for line in lines:
            if line.strip().startswith("add_subdirectory"):
                mods.append(parse_line(line))
                flag = 1
            elif line.startswith("endif()"):
                if plugin_name not in mods and flag == 1:
                    f.write("  add_subdirectory(" + plugin_name + ")\n")
            f.write(line)
    print("Update the CMakeLists.txt file successfully!")

update_cmake_file(os.path.join(clang_dir, "examples/CMakeLists.txt"), "ExtractFunctionPrototypes")
update_cmake_file(os.path.join(clang_dir, "examples/CMakeLists.txt"), "FreeNullCheck")
update_cmake_file(os.path.join(clang_dir, "examples/CMakeLists.txt"), "MemoryDataFlow")
update_cmake_file(os.path.join(clang_dir, "examples/CMakeLists.txt"), "MemoryDataFlowFree")
update_cmake_file(os.path.join(clang_dir, "lib/Analysis/plugins/CMakeLists.txt"), "GoshawkAnalyzer")
os.chdir(build_dir)
os.system("ninja {} -j{}".format("ExtractFunctionPrototypes", os.cpu_count()))
os.system("ninja {} -j{}".format("FreeNullCheck", os.cpu_count()))
os.system("ninja {} -j{}".format("MemoryDataFlow", os.cpu_count()))
os.system("ninja {} -j{}".format("MemoryDataFlowFree", os.cpu_count()))
os.system("ninja {} -j{}".format("GoshawkAnalyzer", os.cpu_count()))
shutil.copy(os.path.join(build_dir, "lib/ExtractFunctionPrototypes.so"), os.path.join(curr_dir, "../plugins/ExtractFunctionPrototypes.so"))
shutil.copy(os.path.join(build_dir, "lib/FreeNullCheck.so"), os.path.join(curr_dir, "../plugins/FreeNullCheck.so"))
shutil.copy(os.path.join(build_dir, "lib/MemoryDataFlow.so"), os.path.join(curr_dir, "../plugins/MemoryDataFlow.so"))
shutil.copy(os.path.join(build_dir, "lib/MemoryDataFlowFree.so"), os.path.join(curr_dir, "../plugins/MemoryDataFlowFree.so"))
shutil.copy(os.path.join(build_dir, "lib/GoshawkAnalyzer.so"), os.path.join(curr_dir, "../plugins/GoshawkAnalyzer.so"))






