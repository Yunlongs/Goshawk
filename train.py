import tensorflow as tf
from tensorflow import keras
import config
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from transformer_layer import Encoder, CustomSchedule, create_padding_mark, get_mean_pool
from dataset import dataset_generation, load_embedding_weight, vocab_size, dataset_generation_hard, generate_hard_mask, \
    funcname_to_vector, batch_extract_embedding

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Siamese_Model_Simple_RNN(keras.Model):
    def __init__(self):
        super(Siamese_Model_Simple_RNN, self).__init__()
        embedding_matrix = load_embedding_weight()
        self.embed_1 = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=config.embedding_size,
                                              weights=[embedding_matrix], input_length=config.max_seq_length,
                                              mask_zero=True, trainable=False)
        self.embed_2 = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=config.embedding_size,
                                              weights=[embedding_matrix], input_length=config.max_seq_length,
                                              mask_zero=True, trainable=False)
        self.rnn_layer = keras.layers.SimpleRNN(config.feature_size, return_state=True)

    def call(self, inputs, training=None, mask=None):
        funcname_1, funcname_2 = inputs
        x_1 = self.embed_1(funcname_1)
        out_1, state_1 = self.rnn_layer(x_1)
        if training == False:
            return state_1
        x_2 = self.embed_2(funcname_2)
        out_2, state_2 = self.rnn_layer(x_2)
        sim = -tf.keras.losses.cosine_similarity(state_1, state_2)
        return state_1, state_2, sim


class Siamese_Model_LSTM(keras.Model):
    def __init__(self):
        super(Siamese_Model_LSTM, self).__init__()
        embedding_matrix = load_embedding_weight()
        self.embed_1 = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=config.embedding_size,
                                              weights=[embedding_matrix], input_length=config.max_seq_length,
                                              mask_zero=True, trainable=False)
        self.embed_2 = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=config.embedding_size,
                                              weights=[embedding_matrix], input_length=config.max_seq_length,
                                              mask_zero=True, trainable=False)
        self.rnn_layer = keras.layers.LSTM(config.feature_size, return_state=True)

    def call(self, inputs, training=None, mask=None):
        funcname_1, funcname_2 = inputs
        x_1 = self.embed_1(funcname_1)
        mask_1 = self.embed_1.compute_mask(funcname_1)
        out_1, state_h1, state_c1 = self.rnn_layer(x_1, mask=mask_1)
        if training == False:
            return state_c1

        x_2 = self.embed_2(funcname_2)
        mask_2 = self.embed_2.compute_mask(funcname_2)
        out_2, state_h2, state_c2 = self.rnn_layer(x_2, mask=mask_2)
        sim = -tf.keras.losses.cosine_similarity(state_c1, state_c2)
        return state_c1, state_c2, sim


class Siamese_Model_2l_LSTM(keras.Model):
    def __init__(self):
        super(Siamese_Model_2l_LSTM, self).__init__()
        embedding_matrix = load_embedding_weight()
        self.embed_1 = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=config.embedding_size,
                                              weights=[embedding_matrix], input_length=config.max_seq_length,
                                              mask_zero=True, trainable=False)
        self.embed_2 = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=config.embedding_size,
                                              weights=[embedding_matrix], input_length=config.max_seq_length,
                                              mask_zero=True, trainable=False)
        self.rnn_layer_1 = keras.layers.LSTM(config.feature_size, return_sequences=True)
        self.rnn_layer_2 = keras.layers.LSTM(config.feature_size, return_state=True)

    def call(self, inputs, training=None, mask=None):
        funcname_1, funcname_2 = inputs
        x_1 = self.embed_1(funcname_1)
        x_1 = self.rnn_layer_1(x_1)
        out_1, state_h1, state_c1 = self.rnn_layer_2(x_1)
        if training == False:
            return state_c1

        x_2 = self.embed_2(funcname_2)
        x_2 = self.rnn_layer_1(x_2)
        out_2, state_h2, state_c2 = self.rnn_layer_2(x_2)
        sim = -tf.keras.losses.cosine_similarity(state_c1, state_c2)
        return state_c1, state_c2, sim


class Siamese_Model_Bi_LSTM(keras.Model):
    def __init__(self):
        super(Siamese_Model_Bi_LSTM, self).__init__()
        embedding_matrix = load_embedding_weight()
        self.embed_1 = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=config.embedding_size,
                                              weights=[embedding_matrix], input_length=config.max_seq_length,
                                              mask_zero=True, trainable=False)
        self.embed_2 = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=config.embedding_size,
                                              weights=[embedding_matrix], input_length=config.max_seq_length,
                                              mask_zero=True, trainable=False)
        self.rnn_layer = keras.layers.Bidirectional(keras.layers.LSTM(config.feature_size, return_state=True))

    def call(self, inputs, training=None, mask=None):
        funcname_1, funcname_2 = inputs
        x_1 = self.embed_1(funcname_1)
        out_1, state_forward_h1, state_forward_c1, state_back_h1, state_back_c1 = self.rnn_layer(x_1)
        state_c1 = keras.layers.concatenate([state_forward_c1, state_back_c1], axis=1)
        if training == False:
            return state_c1
        x_2 = self.embed_2(funcname_2)
        out_2, state_forward_h2, state_forward_c2, state_back_h2, state_back_c2 = self.rnn_layer(x_2)
        state_c2 = keras.layers.concatenate([state_forward_c2, state_back_c2], axis=1)
        sim = -tf.keras.losses.cosine_similarity(state_c1, state_c2)
        return state_c1, state_c2, sim


class Siamese_Transformer_encoder(keras.Model):
    def __init__(self, pooling="Mean"):
        super(Siamese_Transformer_encoder, self).__init__()
        weight = load_embedding_weight()
        assert pooling == "Mean" or pooling == "CLS"
        self.pooling = pooling
        self.encoder_layer = Encoder(n_layers=config.n_layers, d_model=config.d_model,
                                     embedding_size=config.embedding_size, embedding_matrix=weight,
                                     n_heads=config.n_heads, ddf=config.diff, input_vocab_size=vocab_size + 1,
                                     max_seq_len=config.max_seq_length, drop_rate=config.drop_rate)

    def call(self, inputs, training=None, mask=None):
        func_1, func_2 = inputs
        mask_1 = create_padding_mark(func_1)  # shape=[batch_size,1,1,seq_len]
        if training == True:
            out_1 = self.encoder_layer(func_1, training=True, mask=mask_1)
        else:
            out_1 = self.encoder_layer(func_1, training=False, mask=mask_1)

        if training == False and self.pooling == "CLS":
            cls_out_1 = tf.squeeze(out_1[:, 0:1, :], axis=1)
            return cls_out_1
        elif training == False and self.pooling == "Mean":
            mean_pool_1 = get_mean_pool(func_1, out_1)
            return mean_pool_1

        mask_2 = create_padding_mark(func_2)
        out_2 = self.encoder_layer(func_2, training=True, mask=mask_2)  # shape= [batch_size,seq_len,embedding_size]
        if self.pooling == "CLS":
            cls_out_1 = tf.squeeze(out_1[:, 0:1, :], axis=1)
            cls_out_2 = tf.squeeze(out_2[:, 0:1, :], axis=1)  # shape=[batch_size,embedding_size]
            sim = -tf.keras.losses.cosine_similarity(cls_out_1, cls_out_2)
            return cls_out_1, cls_out_2, sim
        else:
            mean_pool_1 = get_mean_pool(func_1, out_1)
            mean_pool_2 = get_mean_pool(func_2, out_2)  # shape=[batch_size,embedding_size]
            sim = -tf.keras.losses.cosine_similarity(mean_pool_1, mean_pool_2)
            return mean_pool_1, mean_pool_2, sim


def mse_loss(model, func1, func2, y):
    """
    Just compute the mean squared error loss for cosine similarity.
    :param model:
    :param func1:
    :param func2:
    :param y: true labels.
    :return:
    """
    input = (func1, func2)
    func1_embedding, func2_embedding, sim = model(input, training=True)
    if tf.reduce_max(sim) > 1 or tf.reduce_min(sim) < -1:
        sim = sim * 0.999
    m = tf.keras.losses.MeanSquaredError()
    loss_value = m(y, sim)
    return loss_value, sim, func1_embedding, func2_embedding


def constrastive_loss(model, func1, func2, y):
    """
    compute the constrastive loss for cosine similarity.
    :param model:
    :param func1:
    :param func2:
    :param y: true labels.
    :return:
    """
    input = (func1, func2)
    func1_embedding, func2_embedding, sim = model(input, training=True)
    if tf.reduce_max(sim) > 1 or tf.reduce_min(sim) < -1:
        sim = sim * 0.999
    batch_size = y.shape[0]
    y = tf.cast(tf.equal(y, 1), dtype=tf.float32)
    d = tf.constant(1, dtype=tf.float32) - sim
    tmp = y * tf.square(d)
    tmp2 = (1 - y) * tf.square(tf.maximum((config.constrastive_margin - d), 0))
    loss_value = tf.reduce_sum(tmp + tmp2) / batch_size / 2
    return loss_value, sim, func1_embedding, func2_embedding


def grad(model, func1, func2, y):
    with tf.GradientTape() as tape:
        if config.loss == "mse":
            loss_value, sim, func1_embedding, func2_embedding = mse_loss(model, func1, func2, y)
        elif config.loss == "constrastive":
            loss_value, sim, func1_embedding, func2_embedding = constrastive_loss(model, func1, func2, y)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), sim, func1_embedding, func2_embedding


def get_model(model_name):
    if model_name == "SimpleRNN":
        model = Siamese_Model_Simple_RNN()
        return model
    elif model_name == "SimpleLSTM":
        model = Siamese_Model_LSTM()
        return model
    elif model_name == "2L_LSTM":
        model = Siamese_Model_2l_LSTM()
        return model
    elif model_name == "Transformer":
        model = Siamese_Transformer_encoder(pooling=config.pooling)
        return model
    else:
        return False


def train(model_name, type="target", backbone = "Transformer"):
    train_start = time.time()
    save_prefix = config.model_dir + os.sep + model_name
    if not os.path.exists(save_prefix):
        os.mkdir(save_prefix)
        os.system("cp config.py %s" % (save_prefix))

    learning_rate = CustomSchedule(config.d_model, config.warmup_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = get_model(backbone)  ## here to choose your model

    model.build([(config.mini_batch, config.max_seq_length), (config.mini_batch, config.max_seq_length)])
    # model._set_inputs([(config.mini_batch,config.max_funcname_length),(config.mini_batch,config.max_funcname_length)])
    model.summary()
    max_f1score = 0
    train_loss = []
    train_accuracy = []
    train_recall = []
    train_f1score = []
    valid_accuracy = []
    valid_recall = []
    valid_f1score = []

    for epoch in range(1, config.epochs + 1):
        if epoch == 1:
            train_dataset = dataset_generation(type=type)
        else:
            train_dataset = dataset_generation_hard(semi_hard_positive_mask, hardest_negative_mask,
                                                    semi_hard_negative_mask, type=type)
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.BinaryAccuracy()
        epoch_auc_avg = tf.keras.metrics.AUC()
        step = 0
        for func1, func2, y in train_dataset:
            loss_value, grads, sim, _, _ = grad(model, func1, func2, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg(loss_value)
            sim = (sim + 1) / 2
            y = (y + 1) / 2
            epoch_accuracy_avg.update_state(y, sim)
            epoch_auc_avg.update_state(y, sim)

            if step % (config.step_per_epoch // 10) == 0:
                print("step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(step, epoch_loss_avg.result(),
                                                                                        epoch_accuracy_avg.result(),
                                                                                        epoch_auc_avg.result()))
            step += 1

        semi_hard_positive_mask, hardest_negative_mask, semi_hard_negative_mask, accuracy, recall, f1_score = generate_hard_mask(
            model, type=type)
        train_loss.append(epoch_loss_avg.result())
        train_accuracy.append(accuracy)
        train_recall.append(recall)
        train_f1score.append(f1_score)
        #v_accuracy, v_recall, v_f1_score = test_per_epoch(model, type)
        #valid_accuracy.append(v_accuracy)
        #valid_recall.append(v_recall)
        #valid_f1score.append(v_f1_score)

        #if v_f1_score > max_f1score:
        #    max_f1score = v_f1_score
        #    model.save(save_prefix + "maxauc_model")

        if f1_score > max_f1score:
            max_f1score = f1_score
            model.save(save_prefix + "maxauc_model")

        if epoch % 10 == 0:
            # model.save(save_prefix + "epoch_%s_model"%(epoch))
            pass

        #print("---------------------------\n epoch:%s Train_accuracy:%s Train_recall:%s Train_f1score:%s\n "
        #      "Valid_accuracy:%s Valid_recall:%s Valid_f1score:%s\n "
        #      % (epoch, accuracy, recall, f1_score, v_accuracy, v_recall, v_f1_score))
        print("---------------------------\n epoch:%s Train_accuracy:%s Train_recall:%s Train_f1score:%s\n "
              % (epoch, accuracy, recall, f1_score))
    train_end = time.time()
    with open(save_prefix + "cost_time", "w") as f:
        f.write("total cost time:" + str(train_end - train_start))
    save_result(save_prefix, train_loss, train_accuracy, train_recall, train_f1score, valid_accuracy, valid_recall,
                valid_f1score)


def save_result(save_prefix, train_loss, train_accuracy, train_recall, train_f1score, valid_accuracy, valid_recall,
                valid_f1score):
    image_prefix = save_prefix + "image/"
    if not os.path.exists(image_prefix):
        os.mkdir(image_prefix)
    text_prefix = save_prefix + "text/"
    if not os.path.exists(text_prefix):
        os.mkdir(text_prefix)
    plt.figure(figsize=(5, 4))
    plt.plot(range(config.epochs), train_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train loss curve")
    plt.savefig(image_prefix + "train_loss.png")
    np.savetxt(text_prefix + "train_loss_per_epoch.txt", train_loss)

    plt.figure(figsize=(5, 4))
    plt.plot(range(config.epochs), train_accuracy)
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.title("train accuracy curve")
    plt.savefig(image_prefix + "train_accuracy.png")
    np.savetxt(text_prefix + "train_accuracy_per_epoch.txt", train_accuracy)

    plt.figure(figsize=(5, 4))
    plt.plot(range(config.epochs), train_recall)
    plt.xlabel("epoch")
    plt.ylabel("Recall")
    plt.title("train recall curve")
    plt.savefig(image_prefix + "train_recall.png")
    np.savetxt(text_prefix + "train_recall_per_epoch.txt", train_recall)

    plt.figure(figsize=(5, 4))
    plt.plot(range(config.epochs), train_f1score)
    plt.xlabel("epoch")
    plt.ylabel("F1-Score")
    plt.title("train F1-Score curve")
    plt.savefig(image_prefix + "train_f1score.png")
    np.savetxt(text_prefix + "train_f1score_per_epoch.txt", train_f1score)

    '''
    plt.figure(figsize=(5, 4))
    plt.plot(range(config.epochs), valid_accuracy)
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.title("valid accuracy curve")
    plt.savefig(image_prefix + "valid_accuracy.png")
    np.savetxt(text_prefix + "valid_accuracy_per_epoch.txt", valid_accuracy)

    plt.figure(figsize=(5, 4))
    plt.plot(range(config.epochs), valid_recall)
    plt.xlabel("epoch")
    plt.ylabel("Recall")
    plt.title("valid recall curve")
    plt.savefig(image_prefix + "valid_recall.png")
    np.savetxt(text_prefix + "valid_recall_per_epoch.txt", valid_recall)

    plt.figure(figsize=(5, 4))
    plt.plot(range(config.epochs), valid_f1score)
    plt.xlabel("epoch")
    plt.ylabel("F1-Score")
    plt.title("valid F1-Score curve")
    plt.savefig(image_prefix + "valid_f1score.png")
    np.savetxt(text_prefix + "valid_f1score_per_epoch.txt", valid_f1score)
    '''


def test_per_epoch(model, type):
    """
    compute score for valid datasets.
    :param model:
    :param type:
    :return:
    """
    mean_embedding = np.load(config.train_data_prefix + os.sep + type + "/" + "mean_embedding.npy")
    with open(config.train_data_prefix + os.sep + type + "/" + type + ".valid") as f:
        func_true = []
        for line in f.readlines():
            if len(line) <= 1:
                break
            func = funcname_to_vector(line.strip())
            func_true.append(func)
    func_true = np.vstack(func_true)
    embedding = batch_extract_embedding(model, func_true, batch_size=config.inference_batch)
    embedding = embedding / tf.norm(embedding, axis=1, keepdims=True)
    distance = np.dot(embedding, mean_embedding.T)
    tp = np.sum(distance >= config.inference_threshold)
    fn = np.sum(distance < config.inference_threshold)

    with open(config.train_data_prefix + os.sep + type + "/" + "neg.valid") as f:
        func_neg = []
        for line in f.readlines():
            if len(line) <= 1:
                break
            func = funcname_to_vector(line.strip())
            func_neg.append(func)
    func_neg = np.vstack(func_neg)
    embedding = batch_extract_embedding(model, func_neg, batch_size=config.inference_batch)
    embedding = embedding / tf.norm(embedding, axis=1, keepdims=True)
    distance = np.dot(embedding, mean_embedding.T)
    fp = np.sum(distance >= config.inference_threshold)

    accuracy = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * accuracy * recall) / (accuracy + recall)
    return accuracy, recall, f1_score


if __name__ == "__main__":
    train("Transformer")
    #train("Transformer", 103, type="free")
