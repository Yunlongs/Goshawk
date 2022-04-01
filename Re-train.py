from data_proccess.get_label_data import convert_origin_to_prototype, convert_prototype_to_json
from train import train
from normalize import normalize_two_files
import config
import argparse,os,shutil

parser = argparse.ArgumentParser(description='Retrain the NLP model for your purpose')
parser.add_argument("training_file", metavar="/xxx/crypto.txt", type= str,help="Collected training function prototype file.")
parser.add_argument("model_name", metavar="my_model",type= str, help="Name of the your retrained model")

args = parser.parse_args()
file = args.training_file
model_name = args.model_name

def delete_exist_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def remake_new_dir(dir_name):
    delete_exist_dir(dir_name)
    os.mkdir(dir_name)


def Re_train_Model(file, model_new_name):
    print("-------------------------------------\n"
          "pre-process data start!\n\n")
    remake_new_dir("temp")
    remake_new_dir("subword_dataset/training/target")
    temp_file = "temp" + os.sep + os.path.basename(file)
    shutil.copy(file, temp_file)
    convert_origin_to_prototype(temp_file)
    convert_prototype_to_json(temp_file)
    print("\n\n--------------------------------\n"
          "pre-process finished!")
    shutil.copy("subword_dataset/training/neg.train", "subword_dataset/training/target/neg.train")
    shutil.copy("subword_dataset/training/neg.valid", "subword_dataset/training/target/neg.valid")
    shutil.copy("subword_dataset/training/neg.test", "subword_dataset/training/target/neg.test")
    normalize_two_files(temp_file, "subword_dataset/training/target/target.train")
    train(model_new_name)
    src_mean_embedding_file = "subword_dataset/training/target/mean_embedding.npy"
    dst_mean_embedding_file = os.path.join(config.model_dir,model_name,"embedding", "mean_embedding.npy")
    shutil.copy(src_mean_embedding_file,dst_mean_embedding_file)


if __name__ == "__main__":
    Re_train_Model(file, model_name)