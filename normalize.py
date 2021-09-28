import json
from mysegment import Segmenter
import config


def create_id_word_map():
    """
    Create a mapping relation which maps subword to id and maps id to subword.
    :return:word_id_map,id_word_map . dict.
    """
    index = 1
    vocab = {}
    word_id_map = {}
    id_word_map = {}
    with open(config.vocab_path, "r") as f:
        line = f.readline()
        for line in f.readlines():
            k, v = line.strip().split()
            vocab[k] = v
    for word in config.conserved_words:
        word_id_map[word] = index
        id_word_map[index] = word
        index += 1
    for word in vocab.keys():
        word_id_map[word] = index
        id_word_map[index] = word
        index += 1
    return word_id_map, id_word_map


word_id_map, id_word_map = create_id_word_map()


def normalize_type_1(type):
    '''
    将类型归一化为，指针类型和非指针类型两类。
    Normalize the arg type and return type to pointer type or or double pointer type or non-pointer type.
    :param type:string.
    :return: '<ptr>' or '<dptr>' or '<noptr>'
    '''
    if type.find('**') != -1:
        return "<dptr>"
    elif type.find('*') != -1:
        return "<ptr>"
    else:
        return "<noptr>"


seg = Segmenter(config.vocab_path)


def normalize_funcname(funcname):
    '''
    Segment the function's name. Return the result that has the highest unigram language model probability.
    :param funcname: string.
    :return:
    '''
    subwords = seg.segment(funcname)
    return subwords


def normalize_funcname_multi(funcname, multi):
    """
    Segment the function's name. Return the multi results that has the highest unigram language model probability.
    :param funcname:
    :param multi: number of segment result. int.
    :return:
    """
    subwords = seg.segment(funcname, multi=multi)
    res = []
    for prob, subword in subwords:
        tmp = []
        for sub_subword in subword:
            if word_id_map.get(sub_subword) == None:
                s_seg = seg.segment(sub_subword)
                for s in s_seg:
                    tmp.append(s)
            else:
                tmp.append(sub_subword)
        res.append(tmp)
    return res


def normalize_type_2(type):
    '''
    将类型归一化为:void,int,bool 和指针类型和其他类型.
    Normal the type to (void,int,bool) or pointer type or non-pointer type.
    :param type:
    :return:‘*’,or '#',or ‘$'
    '''
    if type.find('*') != -1:
        return '*'
    elif type.find('int') != -1 or type.find('bool') != -1 or type.find('void') != -1:
        return '$'
    else:
        return '#'


def parse_params(params, normalize=True, arg_normalize = True):
    res = []
    while params.find("@") != -1:
        idx = params.index('@')
        argtype = params[:idx]
        raw_argtype = argtype
        argtype = normalize_type_1(argtype)

        params = params[idx:]
        idx = params.index(",")
        argname = params[1:idx]
        if normalize is True:
            argname = normalize_funcname(argname)
        params = params[idx + 1:]
        if arg_normalize is True:
            res.append((argtype, argname))
        else:
            res.append((raw_argtype, argtype, argname))
    return res


def normalize_on_file(filename, type, out_file):
    '''
    Nomalize function prototypes in a file.
    For example:
    ‘<ptr>’：pointer type *。'<dptr>'double pointer type **。 '<noptr>' not pointer type.
    :param filename:
    :param type: alloc or free
    :return:
    '''
    assert type == "alloc" or type == "free", "type error!"
    max_length = 0

    f_out = open(out_file, "w")
    with open(filename, "r") as f:
        for line in f.readlines():
            fullname = []
            fullname.append("<cls>")
            #print(line)
            func = json.loads(line.strip())
            fullname.append(normalize_type_1(func["return_type"]))
            if len(func["funcname"]) > 150:
                continue
            fullname += normalize_funcname(func["funcname"])
            fullname.append("(")

            params = func['params']
            for argtype, argname in parse_params(params):
                fullname.append(argtype)
                fullname += argname
                fullname.append("<dot>")
            if params != "":
                fullname.pop(-1)
            fullname.append(")")
            if len(fullname) > max_length:
                max_length = len(fullname)
            full_string = " ".join(fullname) + "\n"
            f_out.write(full_string)
    f_out.write("\nmax_length:" + str(max_length))
    f_out.close()


def normalize_one_file_multi(filename, type, out_file, multi):
    '''
        标准化函数原型,并为每个函数名进行多个概率分词。‘<ptr>’：指针类型。'<noptr>'非指针类型。
        :param filename:
        :param type: alloc or free
        :param multi: 为每个函数原型生成多少个分词结果
        :return:
        '''
    assert type == "alloc" or type == "free", "type error!"
    max_length = 0

    f_out = open(out_file, "w")
    with open(filename, "r") as f:
        for line in f.readlines():
            func = json.loads(line.strip())
            funcname = func["funcname"]
            params = func['params']
            try:
                funcname_segs = normalize_funcname_multi(funcname, multi)
            except Exception as e:
                print(e)
                print(funcname)
                print("funcname is illeagal!")
                continue

            for funcname_seg in funcname_segs:
                fullname = []
                fullname.append("<cls>")
                fullname.append(normalize_type_1(func["return_type"]))
                fullname += funcname_seg
                fullname.append("(")

                for argtype, argname in parse_params(params):
                    fullname.append(argtype)
                    fullname += argname
                    fullname.append("<dot>")
                if params != "":
                    fullname.pop(-1)
                fullname.append(")")
                if len(fullname) > max_length:
                    max_length = len(fullname)
                full_string = " ".join(fullname) + "\n"
                f_out.write(full_string)
    f_out.write("\nmax_length:" + str(max_length))
    f_out.close()


def normalize_dir(in_dir, out_dir, multi=None):
    import os
    dirs = os.listdir(in_dir)
    for file in dirs:
        in_file = in_dir + os.sep + file
        out_file = out_dir + os.sep + file
        normalize_two_files(in_file, out_file, multi)


def normalize_two_files(in_file, out_file, multi=None):
    if multi == None:
        normalize_on_file(in_file, "alloc", out_file)
    else:
        normalize_one_file_multi(in_file, "alloc", out_file, multi)

def normalize_prototype_file(in_file, out_file):
    '''
        Segment and normalize the function prototypes in the `in_file`, and the results are saved at `out_file.`
        For example,
            before: void * kmalloc_array(size_t n, size_t size, gfp_t flags)
            after:  <cls> <ptr> kmalloc array ( <noptr> n <dot> <noptr> size <dot> <noptr> flags )
    '''
    from data_proccess.get_label_data import convert_origin_to_prototype, convert_prototype_to_json
    convert_origin_to_prototype(in_file)
    convert_prototype_to_json(in_file)
    normalize_on_file(in_file,"alloc", out_file)


if __name__ == "__main__":
    normalize_prototype_file("subword_dataset/test1.func", "temp/seg.func")