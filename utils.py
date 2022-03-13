import json

def argmax(indexs):
    max = 0
    max_arg = 0
    for i,index in enumerate(indexs):
        if index > max:
            max = index
            max_arg = i
    return max_arg

def cleanup_free_null_check(file):
    func_maps = {}
    with open(file) as f:
        for line in f.readlines():
            if len(line) <=4:
                break
            func,index = line.strip().split()
            index = int(index)
            if func_maps.get(func) is None:
                func_maps[func] = [0,0,0,0,0,0,0]
            print(func)
            func_maps[func][index] +=1

    new_map = {}
    for k,v in func_maps.items():
        max_arg = argmax(v)
        max = v[max_arg]
        new_func = k + "\t" + str(max_arg)
        new_map[new_func] = max
    new_map = sorted(new_map.items(), key=lambda d: d[1], reverse=True)
    with open(file,"w") as f:
        for func,num in new_map:
            f.write(func + "\t" + str(num)+"\n")

def add_primitive_functions(out_dir = "output/free/"):
    normal_free = []
    customized_free = []
    with open("temp/seed_free.txt", "r") as f:
        for line in f.readlines():
            func,index = line.strip().split()
            index = int(index)
            if index ==0:
                normal_free.append(func)
            else:
                func_json = {}
                func_json["funcname"] = func
                func_json["param_names"] = [index,"map"]
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
    with open("temp/func_similarity","r") as f:
        for line in f.readlines():
            if "*" in line:
                count +=1
    print(count)

if __name__ == "__main__":
    #cleanup_null_check("temp/free_check.txt")
    count_candidate_deallocation()
#consume_skb	0	261