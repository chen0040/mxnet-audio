import numpy as np
from sklearn.model_selection import train_test_split
import mxnet as mx
from mxnet import nd, autograd, gluon
import os
from lru import LRU
from mxnet_audio.library.utility.audio_utils import compute_melgram
from random import shuffle


def cifar10(nb_classes):
    channel_axis = 1
    freq_axis = 2
    time_axis = 3

    activation_func = 'softrelu'

    model = gluon.nn.HybridSequential()
    with model.name_scope():
        model.add(gluon.nn.Conv2D(channels=32, kernel_size=3, padding=1))
        model.add(gluon.nn.BatchNorm(axis=channel_axis))
        model.add(gluon.nn.Activation(activation_func))
        model.add(gluon.nn.MaxPool2D(pool_size=(2, 4)))

        model.add(gluon.nn.Conv2D(channels=32, kernel_size=3, padding=1))
        model.add(gluon.nn.BatchNorm(axis=channel_axis))
        model.add(gluon.nn.Activation(activation_func))
        model.add(gluon.nn.MaxPool2D(pool_size=(2, 4)))

        model.add(gluon.nn.Dropout(rate=0.25))

        model.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=1))
        model.add(gluon.nn.BatchNorm(axis=channel_axis))
        model.add(gluon.nn.Activation(activation_func))
        model.add(gluon.nn.MaxPool2D(pool_size=(2, 4)))

        model.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=1))
        model.add(gluon.nn.BatchNorm(axis=channel_axis))
        model.add(gluon.nn.Activation(activation_func))
        model.add(gluon.nn.MaxPool2D(pool_size=(3, 5)))

        model.add(gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1))
        model.add(gluon.nn.BatchNorm(axis=channel_axis))
        model.add(gluon.nn.Activation(activation_func))
        model.add(gluon.nn.MaxPool2D(pool_size=(4, 4)))

        model.add(gluon.nn.Dropout(0.25))

        model.add(gluon.nn.Flatten())
        model.add(gluon.nn.Dense(512))
        model.add(gluon.nn.Activation(activation_func))
        model.add(gluon.nn.Dropout(0.5))
        model.add(gluon.nn.Dense(nb_classes))

    return model


class Cifar10AudioClassifier(object):
    model_name = 'cifar10'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.cache = LRU(400)
        self.input_shape = None
        self.nb_classes = None
        self.model = None
        self.config = None
        self.model_ctx = model_ctx
        self.data_ctx = data_ctx

    @staticmethod
    def create_model(nb_classes):
        return cifar10(nb_classes)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10AudioClassifier.model_name + '-config.npy')

    @staticmethod
    def get_params_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10AudioClassifier.model_name + '-net.params')

    def load_model(self, model_dir_path):
        config_file_path = Cifar10AudioClassifier.get_config_file_path(model_dir_path)
        params_file_path = Cifar10AudioClassifier.get_params_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.input_shape = self.config['input_shape']
        self.nb_classes = self.config['nb_classes']
        self.model = self.create_model(self.nb_classes)
        self.model.load_params(params_file_path, ctx=self.model_ctx)
        self.model.hybridize()

    def compute_melgram(self, audio_path):
        if audio_path in self.cache:
            return self.cache[audio_path]
        else:
            mg = compute_melgram(audio_path)
            # mg = (mg + 100) / 200  # scale the values
            self.cache[audio_path] = mg
            return mg

    def generate_batch(self, audio_paths, labels, batch_size, shuffled):
        num_batches = len(audio_paths) // batch_size
        while True:
            batch_index_list = list(range(0, num_batches))
            if shuffled:
                shuffle(batch_index_list)
            for batchIdx in batch_index_list:
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size

                X = np.zeros(shape=(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                             dtype=np.float32)
                for i in range(start, end):
                    audio_path = audio_paths[i]
                    mg = compute_melgram(audio_path)
                    X[i - start, :, :, :] = mg
                yield nd.array(X, ctx=self.data_ctx), nd.array(labels[start:end], ctx=self.data_ctx)

    @staticmethod
    def one_hot(y, nb_classes):
        result = np.zeros(shape=(len(y), nb_classes))
        for i, label in enumerate(y):
            result[i, label] = 1
        return result

    @staticmethod
    def unzip(audio_path_label_pairs):
        X = []
        Y = []

        for audio_path, label in audio_path_label_pairs:
            X.append(audio_path)
            Y.append(label)

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def checkpoint(self, model_dir_path):
        self.model.save_params(self.get_params_file_path(model_dir_path))

    def evaluate_accuracy(self, audio_path_label_pairs, batch_size=64):
        X, Y = self.unzip(audio_path_label_pairs)
        return self._evaluate_accuracy(X, Y, batch_size)

    def _evaluate_accuracy(self, X, Y, batch_size=64):
        data_loader = self.generate_batch(X, Y, batch_size, shuffled=False)

        softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()

        num_batches = len(X) // batch_size

        metric = mx.metric.Accuracy()
        loss_avg = 0.
        for i, (data, label) in enumerate(data_loader):
            data = data.as_in_context(self.model_ctx)
            label = label.as_in_context(self.model_ctx)
            output = self.model(data)
            predictions = nd.argmax(output, axis=1)
            loss = softmax_loss(output, label)
            metric.update(preds=predictions, labels=label)
            loss_avg = loss_avg * i / (i + 1) + nd.mean(loss).asscalar() / (i + 1)

            if i + 1 == num_batches:
                break
        return metric.get()[1], loss_avg

    def fit(self, audio_path_label_pairs, model_dir_path, batch_size=64, epochs=20, test_size=0.2,
            random_state=42, input_shape=(1, 96, 1366), nb_classes=10, learning_rate=.001,
            checkpoint_interval=10):

        config_file_path = Cifar10AudioClassifier.get_config_file_path(model_dir_path)

        self.input_shape = input_shape
        self.nb_classes = nb_classes

        self.config = dict()
        self.config['input_shape'] = input_shape
        self.config['nb_classes'] = nb_classes
        np.save(config_file_path, self.config)

        self.model = self.create_model(self.nb_classes)

        X, Y = self.unzip(audio_path_label_pairs)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size, shuffled=True)

        train_num_batches = len(Xtrain) // batch_size

        self.model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.model_ctx)
        self.model.hybridize()
        trainer = gluon.Trainer(self.model.collect_params(), optimizer='adam', optimizer_params={
            'learning_rate': learning_rate
        })

        softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()

        history = dict()
        loss_train = []
        loss_test = []
        acc_train = []
        acc_test = []

        for e in range(epochs):
            loss_avg = 0.
            accuracy = mx.metric.Accuracy()
            for batch_index, (data, label) in enumerate(train_gen):
                data = data.as_in_context(self.model_ctx)
                label = label.as_in_context(self.model_ctx)
                with autograd.record():
                    output = self.model(data)
                    prediction = nd.argmax(output, axis=1)
                    accuracy.update(preds=prediction, labels=label)
                    loss = softmax_loss(output, label)
                loss.backward()
                trainer.step(data.shape[0])
                loss_avg = loss_avg * batch_index / (batch_index + 1) + nd.mean(loss).asscalar() / (batch_index + 1)
                print("Epoch %s / %s, Batch %s / %s. Loss: %s, Accuracy: %s" %
                      (e + 1, epochs, batch_index + 1, train_num_batches, loss_avg, accuracy.get()[1]))
                if batch_index + 1 == train_num_batches:
                    break
            train_acc = accuracy.get()[1]
            acc_train.append(train_acc)
            loss_train.append(loss_avg)

            test_acc, test_avg_loss = self._evaluate_accuracy(Xtest, Ytest,
                                                              batch_size=batch_size)
            acc_test.append(test_acc)
            loss_test.append(test_avg_loss)

            print("Epoch %s / %s. Loss: %s. Accuracy: %s. Test Accuracy: %s." %
                  (e + 1, epochs, loss_avg, train_acc, test_acc))

            if e % checkpoint_interval == 0:
                self.checkpoint(model_dir_path)

        self.checkpoint(model_dir_path)

        history['loss_train'] = loss_train
        history['loss_test'] = loss_test
        history['acc_train'] = acc_train
        history['acc_test'] = acc_test

        np.save(model_dir_path + '/' + Cifar10AudioClassifier.model_name + '-history.npy', history)

        return history

    def encode_audio(self, audio_path):
        mg = compute_melgram(audio_path)
        mg = nd.array(np.expand_dims(mg, axis=0), ctx=self.model_ctx)
        return self.model(mg).asnumpy()[0]

    def predict_class(self, audio_path):
        predicted = self.encode_audio(audio_path)
        return np.argmax(predicted)


class Cifar10AudioSearch(Cifar10AudioClassifier):

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        super(Cifar10AudioSearch, self).__init__(model_ctx, data_ctx)
        self.database = []

    def index_audio(self, audio_path):
        self.database.append((audio_path, self.encode_audio(audio_path)))

    @staticmethod
    def distance(v1, v2, skip_exact_match=True):
        dist = np.sqrt(np.sum((v1-v2) ** 2))
        if skip_exact_match and dist == 0:
            return 10000000000
        return dist

    def query(self, audio_path, top_k=10, skip_exact_match=True):
        vec = self.encode_audio(audio_path)
        result = sorted(self.database, key=lambda x: self.distance(x[1], vec, skip_exact_match),  reverse=False)
        return result[:top_k] if len(result) >= top_k else result


