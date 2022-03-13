from parse_call_graph import remove_dup_caller
from auto_extract_func import plugin_run
import os, time, config
import argparse
from frontend_checker import initial_candidate_alloc_functions, run_free
from process_result import deduplicate_dataflow, \
    classify_free_data, get_CSA_format
from utils import cleanup_free_null_check, add_primitive_functions

parser = argparse.ArgumentParser(description='Process CSA data flow plugins.')
parser.add_argument("project_dir", metavar="/xxx/linux-5.12", type=str, help="The dir of project you want to analyze.")

args = parser.parse_args()
project_dir = args.project_dir


def delete_exist_file(file):
    if os.path.exists(file):
        os.remove(file)


def delete_exist_dir(dir_name):
    if os.path.exists(dir_name):
        os.rmdir(dir_name)


def remake_new_dir(dir_name):
    delete_exist_dir(dir_name)
    os.mkdir(dir_name)


def Step_0_Cleanup():
    print("Step0: Cleanup start!")
    remake_new_dir("temp")
    remake_new_dir("output/alloc")
    remake_new_dir("output/free")

    print("Step0: Cleanup finished!")
    print("-----------------------------------------------\n------------------------------------\n")


def Step_1_Extract():
    print("\n\n-----------------------------------------------\n")
    print("Step1: Extract Call Graph from source code Start!")
    print("-----------------------------------------------\n\n\n")
    extract_start = time.time()

    flag = "extract-funcs"
    plugin_run(project_dir, flag)
    remove_dup_caller(config.call_graph_path, config.call_graph_path)

    extract_end = time.time()
    with open("temp/call_graph_extract.time", "w") as f:
        f.write(str(extract_end - extract_start) + "\n")
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
    annotation_start = time.time()
    initial_candidate_alloc_functions(config.call_graph_path, step=1)
    annotation_end = time.time()
    with open("temp/alloc_annotation.time", "w") as f:
        f.write("alloc annotation time:" + str(annotation_end - annotation_start) + "\n")


def get_nr_lines(file):
    with open(file, "r") as f:
        lines = len(f.readlines())
    return lines


def Step_3_Free():
    os.system("clear")
    print("\n\n-----------------------------------------------\n")
    print("Step3: Identify Deallocation Functions from source code Start!")
    print("-----------------------------------------------\n\n\n")

    """
        Here, we use Siamese network to generate similarity socres (Deallocation) to each function prototype in project.
        Then, we adopt a threshold (config.inference_threshold) to classify these function prototypes with ["deallocation", "non-deallocation"].
        Those function annotated with "deallocation" are considered as a candidate MM deallocation function.
    """
    classification_start = time.time()
    run_free(config.call_graph_path, step=1)
    classification_end = time.time()
    with open("temp/dealloc_classification.time", "w") as f:
        f.write("dealloc_classification time:" + str(classification_end - classification_start) + "\n")
    time.sleep(2)

    #with open(config.candidate_free_path, "w") as f:
    #    with open("temp/Sinkfinder_free.txt", "r") as f_r:
    #        f.write(f_r.read())

    flag = "free-check"
    plugin_run(project_dir, flag)
    cleanup_free_null_check("temp/free_check.txt")
    run_free(config.call_graph_path, step=2)

    """
        According the MM deallocation candidates, track the data flows inside their implementations.
    """
    generation_start = time.time()
    flag = "point-memory-free-1"
    plugin_run(project_dir, flag)
    deduplicate_dataflow(config.mos_free_outpath)

    # loop until all data flow are tracked (no extra MOS added).
    iteration = 1
    while 1:  # repeat until MOS information generation is converged
        old_lines = get_nr_lines(config.mos_free_outpath)
        os.system("clear")
        print("Current iteration:\t", iteration, "\nCurrent number of lines:\t", old_lines)
        # input("pause... enter any key to continue..")
        flag = "point-memory-free-2"
        os.rename(config.mos_free_outpath, config.mos_seed_path)
        plugin_run(project_dir, flag)
        deduplicate_dataflow(config.mos_free_outpath)
        new_lines = get_nr_lines(config.mos_free_outpath)
        if new_lines - old_lines == 0:
            break
        iteration += 1
        if iteration > config.max_iteration:
            break

    classify_free_data(config.mos_free_outpath)
    add_primitive_functions()
    get_CSA_format()
    generation_end = time.time()
    with open("temp/dealloc_generation.time", "w") as f:
        f.write("dealloc_generation:")
        f.write(str(generation_end - generation_start))
        f.write("\n")


if __name__ == "__main__":
    # Step_0_Cleanup()
    # Step_1_Extract()
    # Step_2_Allocation()
    Step_3_Free()
