from parse_call_graph import remove_dup_caller
from auto_extract_func import plugin_run
import os, time, config, shutil, subprocess
import argparse
from frontend_checker import run_alloc, run_free
from process_result import deduplicate_dataflow, \
    classify_free_data, get_CSA_format
from utils import cleanup_free_null_check, add_primitive_functions, retrive_next_step_TU, concat_two_file
from csa_report_clean import csa_report_cleaner

parser = argparse.ArgumentParser(description='Process CSA data flow plugins.')
parser.add_argument("project_dir", metavar="/xxx/linux-5.12", type=str, help="The dir of project you want to analyze. "
                                                                             "There should be a compilation database "
                                                                             "of this project under the directory")

args = parser.parse_args()
project_dir = args.project_dir


def delete_exist_file(file):
    if os.path.exists(file):
        os.remove(file)


def delete_exist_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def remake_new_dir(dir_name):
    delete_exist_dir(dir_name)
    os.mkdir(dir_name)


def Step_0_Cleanup():
    print("Step0: Cleanup start!")
    remake_new_dir("temp")
    if not os.path.exists("output"):
        os.mkdir("output")
    remake_new_dir("output/alloc")
    remake_new_dir("output/free")
    remake_new_dir("temp/CSA")
    delete_exist_dir("output/report_html")

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
    run_alloc(config.call_graph_path, step=1)
    annotation_end = time.time()
    with open(config.time_record_file, "w") as f:
        f.write("alloc_annotation:" + str(annotation_end - annotation_start) + "\n")

    """
            Call the data flow tracking plugins to track the data flows inside the MM candidates. 
            Merge the data flows and generate MOS.
    """
    generation_start = time.time()
    flag = "point-memory-alloc"
    plugin_run(project_dir, flag)
    run_alloc(config.call_graph_path, step=2)
    generation_end = time.time()
    with open(config.time_record_file, "w") as f:
        f.write("alloc_generation:" + str(generation_end - generation_start) + "\n")
    print("\n\n-----------------------------------------------\n")
    print("Step2: Identify Allocation Functions from source code Finished!")
    print("-----------------------------------------------\n\n\n")


def get_nr_lines(file):
    if not os.path.exists(file):
        return 0
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
    with open(config.time_record_file, "a") as f:
        f.write("dealloc_classification time:" + str(classification_end - classification_start) + "\n")
    time.sleep(2)

    flag = "free-check"
    plugin_run(project_dir, flag)
    cleanup_free_null_check(config.free_check_file)
    call_graph, call_chains = run_free(config.call_graph_path, step=2)

    """
        According the MM deallocation candidates, track the data flows inside their implementations.
    """
    generation_start = time.time()
    flag = "point-memory-free-1"
    next_step_TU = retrive_next_step_TU(project_dir, call_graph, call_chains, 0)
    plugin_run(project_dir, flag, next_step_TU)
    deduplicate_dataflow(config.mos_free_outpath)

    # loop until all data flow are tracked (no extra MOS added).
    iteration = 1
    while 1:  # repeat until MOS information generation is converged
        if iteration > config.max_iteration:
            break
        old_lines = get_nr_lines(config.mos_free_outpath)
        os.system("clear")
        flag = "point-memory-free-2"
        len_next_TUs = 0 if next_step_TU is None else len(next_step_TU)
        print("Current iteration:\t", iteration, "\nCurrent number of lines:\t", old_lines, "\n number of TUs:\t",
              len_next_TUs)
        # input("pause... enter any key to continue..")
        if os.path.exists(config.mos_free_outpath):
            os.rename(config.mos_free_outpath, config.mos_seed_path)
        next_step_TU = retrive_next_step_TU(project_dir, call_graph, call_chains, iteration)
        plugin_run(project_dir, flag, next_step_TU)
        ret_code = deduplicate_dataflow(config.mos_free_outpath)
        if ret_code == -1:
            break
        concat_two_file(config.mos_seed_path, config.mos_free_outpath)
        deduplicate_dataflow(config.mos_free_outpath, 1)
        new_lines = get_nr_lines(config.mos_free_outpath)
        if new_lines - old_lines == 0:
            break
        iteration += 1


    classify_free_data(config.mos_free_outpath)
    add_primitive_functions()
    get_CSA_format()
    generation_end = time.time()
    with open(config.time_record_file, "a") as f:
        f.write("dealloc_generation:" + str(generation_end - generation_start) + "\n")


def isCodeCheckerExist():
    retCode = subprocess.call("CodeChecker version", shell=True)
    if retCode == 0:
        return 1
    retCode = subprocess.call("codechecker version", shell=True)
    if retCode == 0:
        return 2
    return 0


def format_analyzer_command():
    retCode = isCodeCheckerExist()
    if retCode == 0:
        print("Please make sure you have installed CodeChecker, and the command is available in current environment.")
        exit(-1)

    with open("subword_dataset/static_analyzer.cfg", "r") as f:
        cfg_cmd = f.read().strip()
    analyzer_plugin = config.plugin_dir + os.sep + "GoshawkChecker.so"
    MemFuncDir = config.temp_dir + os.sep + "CSA"
    PathNumberFile = MemFuncDir + os.sep + "path_number.txt"
    ExterFile = MemFuncDir + os.sep + "extern_count.txt"

    cfg_cmd += " -Xclang -load -Xclang {0} -Xclang -analyze -Xclang -analyzer-checker=security.GoshawkChecker -Xclang -analyzer-config -Xclang " \
               "security.GoshawkChecker:MemFuncsDir={1} -Xclang -analyzer-config -Xclang security.GoshawkChecker:PathNumberFile={2} " \
               "-Xclang -analyzer-config -Xclang security.GoshawkChecker:ExternFile={3}".format(analyzer_plugin,
                                                                                                MemFuncDir,
                                                                                                PathNumberFile,
                                                                                                ExterFile)
    analyzer_cfg = config.temp_dir + os.sep + "static_analyzer.cfg"
    with open(analyzer_cfg, "w") as f:
        f.write(cfg_cmd)
    if retCode == 1:
        analyzer_cmd = "CodeChecker"
    else:
        analyzer_cmd = "codechecker"

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    compilation_json = project_dir + os.sep + "compilation.json"
    ctu_cache = config.temp_dir + os.sep + "analyze_cache"
    cmd = analyzer_cmd + " analyze --analyzers clangsa -j{0} {1} --saargs {2} -d apiModeling -d cplusplus " \
                         "-d nullability -d optin -d security -d unix -d valist -d deadcode -d core -d security.insecureAPI.rand " \
                         "--ctu --output {3}".format(cpu_count, compilation_json, analyzer_cfg, ctu_cache)

    report_path = "output/report_html"
    parser_cmd = analyzer_cmd + " parse {0} -e html -o {1}".format(ctu_cache, report_path)
    return cmd, parser_cmd


def Step_4_Analyze():
    out_alloc_dir = "output/alloc/"
    shutil.copy(out_alloc_dir + "AllocNormalFile.txt", config.temp_dir + os.sep + "CSA/AllocNormalFile.txt")
    shutil.copy(out_alloc_dir + "AllocCustomizedFile.txt", config.temp_dir + os.sep + "CSA/AllocCustomizedFile.txt")
    out_free_dir = "output/free/"
    shutil.copy(out_free_dir + "FreeNormalFile.txt", config.temp_dir + os.sep + "CSA/FreeNormalFile.txt")
    shutil.copy(out_free_dir + "FreeCustomizedFile.txt", config.temp_dir + os.sep + "CSA/FreeCustomizedFile.txt")

    cmd, parser_cmd = format_analyzer_command()
    print("\nThe bug detection phase start!\n")
    os.system("clear")
    print(cmd)
    subprocess.call(cmd, shell=True)

    print("\nParsing the detection result to html report!\n")
    os.system("clear")
    print(parser_cmd)
    subprocess.call(parser_cmd, shell=True)

    html_path = "output/report_html/index.html"
    cleaner = csa_report_cleaner(html_path)
    cleaner.clean()


if __name__ == "__main__":
    Step_0_Cleanup()
    Step_1_Extract()
    Step_2_Allocation()
    Step_3_Free()
    Step_4_Analyze()
