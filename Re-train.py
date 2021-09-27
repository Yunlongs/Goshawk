from data_proccess.get_label_data import convert_origin_to_prototype, convert_prototype_to_json
from train import train
from normalize import normalize_two_files
import argparse

parser = argparse.ArgumentParser(description='Retrain the NLP model for your purpose')
parser.add_argument("training corpus", metavar="/xxx/crypto.txt", type= str,help="Collected training function prototype file.")
parser.add_argument("model_name", metavar="my_model",type= str, help="Name of the your retrained model")

args = parser.parse_args()
file = args.file
model_name = args.model_name
inference = args.inference


def Re_train_Model(file, model_new_name):
    print("-------------------------------------\n"
          "pre-process data start!\n\n")
    convert_origin_to_prototype(file)
    convert_prototype_to_json(file)
    print("\n\n--------------------------------\n"
          "pre-process finished!")

    normalize_two_files(file, "subword_dataset/training/target.train")
    train(model_new_name)


if __name__ == "__main__":
    Re_train_Model(file, model_name)