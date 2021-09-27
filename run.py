from parse_call_graph import to_json,remove_dup_caller
from auto_extract_func import for_kernel,for_others
import os,time
import argparse
from frontend_checker import belief_chain_propagation_algorithm_alloc,belief_chain_propagation_algorithm_free,run_free
from process_result import remove_dup, get_overlaped_free, add_new_memory_flow, get_can_direct_use_deallocation, \
    classify_free_data, get_new_round_free, get_added_free,get_CSA_format
from utils import cleanup_null_check,add_primitive_functions

parser = argparse.ArgumentParser(description='Process CSA data flow plugins.')
parser.add_argument("project_dir", metavar="/xxx/linux-5.12", type= str,help="The dir of project you want to analyze.")
parser.add_argument("min_call",type=int, help="Whether this project is a huge project, such as kernel.")
parser.add_argument("min_reset",type=int, help="Whether this project is a huge project, such as kernel.")


args = parser.parse_args()
plugin_dir = os.getcwd() + "/plugins"
project_dir = args.project_dir
min_call = args.min_call
min_reset = args.min_reset
isKernel = 0


def Step_0_Cleanup():
    print("Step0: Cleanup start!")
    os.system("rm temp -r")
    os.system("rm output -r")
    os.system("rm /tmp/visited")
    os.system("touch /tmp/visited")
    os.mkdir("temp")
    os.mkdir("output")
    os.mkdir("output/alloc")
    os.mkdir("output/free")
    files = os.listdir(plugin_dir)
    for file in files:
        if file.endswith(".so"):
            continue
        file_path = os.path.join(plugin_dir,file)
        os.system("rm %s" %(file_path))
    print("Step0: Cleanup finished!")
    print("-----------------------------------------------\n------------------------------------\n")


def Step_1_Extract():
    print("\n\n-----------------------------------------------\n")
    print("Step1: Extract Call Graph from source code Start!")
    print("-----------------------------------------------\n\n\n")
    flag = "print-fns"
    extract_start = time.time()
    if isKernel is True:
        for_kernel(plugin_dir,project_dir,flag)
    else:
        for_others(plugin_dir,project_dir,flag)
    call_graph_path= plugin_dir + os.sep + "call_graph.json"
    to_json(call_graph_path)
    remove_dup_caller(call_graph_path,call_graph_path)
    extract_end = time.time()
    with open("temp/extract.time", "w") as f:
        f.write(str(extract_end - extract_start) + "\n")
    os.system("mv %s %s" % (call_graph_path, "temp/call_graph.json"))
    print("-----------------------------------------------\n\n\n")
    print("Step1: Extract Call Graph from source code Finished!")
    print("-----------------------------------------------\n\n\n")

def Step_2_Allocation():
    print("\n\n-----------------------------------------------\n")
    print("Step2: Identify Allocation Functions from source code Start!")
    print("-----------------------------------------------\n\n\n")

    """
        Here, we use Siamese network to generate similarity socres (allocation) to each function prototype in project.
        Then, we adopt a threshold (config.inference_threshold) to classify these function prototypes with ["allocation", "non-allocation"].
        Those function annotated with "allocation" are considered as a candidate MM allocation function.
    """
    in_file = "temp/call_graph.json"
    memory_flow_file = "temp/memory_flow_alloc.json"
    annotation_start = time.time()
    belief_chain_propagation_algorithm_alloc(in_file, memory_flow_file, step=1)
    annotation_end = time.time()
    with open("temp/alloc_annotation.time","w") as f:
        f.write("alloc_annotation:")
        f.write(str(annotation_end-annotation_start))
        f.write("\n")
    os.remove("temp/allocation_set")
    os.remove("temp/allocation_set_call_graph")
    os.system("mv temp/allocation_set_only_funcname temp/allocation_set")

    """
        Here, we call the data flow tracking plugins to track the data flows inside the MM candidates. 
    """
    generation_start = time.time()
    os.system("cp temp/allocation_set %s/allocation_set" % (plugin_dir))
    flag = "point-memory"
    if isKernel == 1:
        for_kernel(plugin_dir,project_dir,flag)
    else:
        for_others(plugin_dir,project_dir,flag)

    os.system("mv %s/memory_flow_alloc.json %s" % (plugin_dir, memory_flow_file))
    if not os.path.exists("output/alloc"):
        os.mkdir("output/alloc")

    """
        Merge the data flows and genrerate MOS.
    """
    belief_chain_propagation_algorithm_alloc(in_file, memory_flow_file, step =2)
    generation_end = time.time()
    with open("temp/alloc_generation.time","w") as f:
        f.write("alloc_generation:")
        f.write(str(generation_end - generation_start))
        f.write("\n")
    print("\n\n-----------------------------------------------\n")
    print("Step2: Identify Allocation Functions from source code Finished!")
    print("-----------------------------------------------\n\n\n")

def get_lines(file):
    lines = 0
    with open(file,"r") as f:
        lines = len(f.readlines())
    return lines


def Step_3_Free_new(min_reset):
    checked_file = "temp/memory_flow_free_checked.json"
    memory_free_file = "temp/memory_flow_free.json"
    call_graph = "temp/call_graph.json"
    overlap_file = "temp/overlap_func.txt"

    os.system("clear")
    print("\n\n-----------------------------------------------\n")
    print("Step3: Identify Deallocation Functions from source code Start!")
    print("-----------------------------------------------\n\n\n")

    """
        Here, we use Siamese network to generate similarity socres (Deallocation) to each function prototype in project.
        Then, we adopt a threshold (config.inference_threshold) to classify these function prototypes with ["deallocation", "non-deallocation"].
        Those function annotated with "deallocation" are considered as a candidate MM deallocation function.
    """
    annotation_start = time.time()
    run_free(call_graph, step=1, min_reassignment = min_reset)
    os.system("cp temp/free_set.txt %s/free_set.txt"%(plugin_dir))
    os.system("cp temp/free_funcs.txt %s/free_funcs.txt" % (plugin_dir))
    annotation_end = time.time()
    with open("temp/dealloc_annotation.time","w") as f:
        f.write("dealloc_annotation:")
        f.write(str(annotation_end - annotation_start))
        f.write("\n")
    time.sleep(2)

    generation_start = time.time()
    flag = "free-check"
    if isKernel == 1:
        for_kernel(plugin_dir, project_dir, flag)
    else:
        for_others(plugin_dir, project_dir, flag)
    os.system("mv %s/free_check.txt temp/free_check.txt" % (plugin_dir))
    cleanup_null_check("temp/free_check.txt")
    run_free(call_graph, step=2, min_reassignment=min_reset)

    """
        Add the official MM functions as the root function to track.
    """
    official_deallocator = []
    with open("subword_dataset/official_deallocator", "r") as f:
        for line in f.readlines():
            official_deallocator.append(line.strip())
    with open("temp/seed_free.txt","a") as f:
        for deallocator in official_deallocator:
            f.write(deallocator + "\t0\n")
    os.system("clear")


    os.system("cp temp/seed_free.txt %s/seed_free.txt"%(plugin_dir))

    """
        According the MM deallocation candidates, track the data flows inside their implementations.
    """
    flag = "point-memory-free-1"
    if isKernel == 1:
        for_kernel(plugin_dir, project_dir, flag)
    else:
        for_others(plugin_dir, project_dir, flag)
    os.system("mv %s/memory_flow_free.json %s" %(plugin_dir, memory_free_file) )
    remove_dup(memory_free_file)

    # loop until all data flow are tracked (no extra MOS added).
    old_lines = get_lines(memory_free_file)
    while 1:
        get_overlaped_free(memory_free_file, call_graph)
        os.system("cp %s %s/overlap_func.txt" %(overlap_file, plugin_dir))
        flag = "point-memory-free-2"
        if isKernel == 1:
            for_kernel(plugin_dir, project_dir, flag)
        else:
            for_others(plugin_dir, project_dir, flag)

        os.system("mv %s/memory_flow_free_checked.json %s" %(plugin_dir,checked_file) )
        remove_dup(checked_file)
        add_new_memory_flow(checked_file, memory_free_file, overlap_file)
        new_lines = get_lines(memory_free_file)
        if new_lines - old_lines == 0:
            break

    classify_free_data(memory_free_file)
    add_primitive_functions()
    get_CSA_format()
    generation_end = time.time()
    with open("temp/dealloc_generation.time", "w") as f:
        f.write("dealloc_generation:")
        f.write(str(generation_end - generation_start))
        f.write("\n")

if __name__ == "__main__":
    Step_0_Cleanup()
    Step_1_Extract()
    Step_2_Allocation()
    Step_3_Free_new(min_reset)