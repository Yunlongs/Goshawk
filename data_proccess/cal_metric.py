from parse_func import parse_func_file
import os

def write_res(acc,recall,tp,fp,fn,out_fie):
    with open(out_fie,"w") as f:
        f.write("False Postive samples:\n-------------------------------------------\n")
        f.writelines([func+"\n" for func in fp])
        f.write("\n\nFalse Negative samples:\n----------------------------------------\n")
        f.writelines([func+"\n" for func in fn])
        f.write("\n\nAccuracy:" + str(acc) + "\n")
        f.write("Recall:" + str(recall))


def cal_NLPEYE_res(dir_path,label_path):
    alloc_path = dir_path + os.sep + "alloc.nlpeye"
    free_path = dir_path + os.sep + "free.nlpeye"
    alloc_res = []
    with open(alloc_path,"r") as f:
        for line in f.readlines():
            if line.startswith("total time"):
                break
            alloc_res.append(line.split()[0])
    free_res = []
    with open(free_path,"r") as f:
        for line in f.readlines():
            if line.startswith("total time"):
                break
            free_res.append(line.split()[0])

    alloc_label = parse_func_file(label_path + os.sep + "mm_alloc_prototype")
    alloc_label = [func for func,arg in alloc_label]
    free_label = parse_func_file(label_path + os.sep + "mm_free_prototype")
    free_label = [func for func,arg in free_label]
    total_tp, total_fp, total_fn = 0,0,0
    tp, fp, fn = 0,0,0
    tp_func, fp_func, fn_func = [],[],[]


    for funcname in alloc_res:
        if funcname in alloc_label:
            tp +=1
            tp_func.append(funcname)
            alloc_label.remove(funcname)
        else:
            fp +=1
            fp_func.append(funcname)
    fn_func = alloc_label
    fn = len(alloc_label)
    total_tp += tp
    total_fp += fp
    total_fn += fn
    alloc_accuracy = tp/(tp + fp)
    alloc_recall = tp/(tp + fn)
    outfile = dir_path + os.sep + "alloc.res_lstm"
    write_res(alloc_accuracy,alloc_recall,tp_func,fp_func,fn_func,outfile)

    tp, fp, fn = 0, 0, 0
    tp_func, fp_func, fn_func = [], [], []
    for funcname in free_res:
        if funcname in free_label:
            tp += 1
            tp_func.append(funcname)
            free_label.remove(funcname)
        else:
            fp += 1
            fp_func.append(funcname)
    fn_func = free_label
    fn = len(fn_func)
    total_tp += tp
    total_fp += fp
    total_fn += fn
    free_accuracy = tp/(tp + fp)
    free_recall = tp/(tp + fn)
    outfile = dir_path + os.sep + "free.res_lstm"
    write_res(free_accuracy,free_recall,tp_func,fp_func,fn_func,outfile)

    accuracy = total_tp/(total_tp + total_fp)
    recall = total_tp /(total_tp + total_fn)

    print("Accuracy:",accuracy)
    print("Recall:",recall)

if __name__ == "__main__":
    cal_NLPEYE_res("../output/NLP_EYE/Experiment_4","../Dataset/labeled_dataset")
