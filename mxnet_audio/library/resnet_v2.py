import numpy as np
from sklearn.model_selection import train_test_split
import mxnet as mx
from mxnet import nd, autograd, gluon
import os
from lru import LRU
from mxnet_audio.library.utility.audio_utils import compute_melgram
from random import shuffle


class Conv2DBlock(gluon.HybridBlock):
    def __init__(self, filters, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.filters = filters
        self.layer_1 = gluon.nn.BatchNorm()
        self.act_1 = gluon.nn.Activation('relu')
        self.conv_1 = gluon.nn.Conv2D(channels=filters, kernel_size=3, padding=1)
        self.layer_2 = gluon.nn.BatchNorm()
        self.act_2 = gluon.nn.Activation('relu')
        self.conv_2 = gluon.nn.Conv2D(channels=filters, kernel_size=3, padding=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = self.layer_1(x)
        x = self.act_1(y)
        y = self.conv_1(x)
        x = self.layer_2(y)
        y = self.act_2(x)
        return self.conv_2(y)


class ResNetV2(gluon.HybridBlock):

    def __init__(self, nb_classes, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        self.nb_classes = nb_classes
        self.filters = [32, 64, 128]

        self.layer1_conv2d = gluon.nn.Conv2D(channels=self.filters[0], kernel_size=3, padding=1)
        self.layer1_pool = gluon.nn.MaxPool2D(padding=1)

        self.layer1_block1 = Conv2DBlock(self.filters[0])
        self.layer1_block2 = Conv2DBlock(self.filters[0])
        self.layer1_block3 = Conv2DBlock(self.filters[0])

        self.layer2_conv2d = gluon.nn.Conv2D(channels=self.filters[1], kernel_size=3, strides=2, padding=1,
                                             activation='softrelu')
        self.layer2_block1 = Conv2DBlock(self.filters[1])
        self.layer2_block2 = Conv2DBlock(self.filters[1])
        self.layer2_block3 = Conv2DBlock(self.filters[1])

        self.layer3_conv2d = gluon.nn.Conv2D(channels=self.filters[2], kernel_size=3, strides=2, padding=1,
                                             activation='softrelu')
        self.layer3_block1 = Conv2DBlock(self.filters[2])
        self.layer3_block2 = Conv2DBlock(self.filters[2])
        self.layer3_block3 = Conv2DBlock(self.filters[2])

        self.global_avg_pool = gluon.nn.GlobalAvgPool2D()

        self.output_layer = gluon.nn.Dense(nb_classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x1 = self.layer1_conv2d(x)
        y = self.layer1_pool(x1)

        x = F.add(self.layer1_block1(y), y)
        y = F.add(self.layer1_block2(x), x)
        x = F.add(self.layer1_block3(y), y)

        y = self.layer2_conv2d(x)
        x = F.add(self.layer2_block1(y), y)
        y = F.add(self.layer2_block2(x), x)
        x = F.add(self.layer2_block3(y), y)

        y = self.layer3_conv2d(x)
        x = F.add(self.layer3_block1(y), y)
        y = F.add(self.layer3_block2(x), x)
        x = F.add(self.layer3_block3(y), y)

        x2 = self.global_avg_pool(x)
        return self.output_layer(x2)


class ResNetV2AudioClassifier(object):
    model_name = 'resnet-v2'

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
        return ResNetV2(nb_classes)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, ResNetV2AudioClassifier.model_name + '-config.npy')

    @staticmethod
    def get_params_file_path(model_dir_path):
        return os.path.join(model_dir_path, ResNetV2AudioClassifier.model_name + '-net.params')

    def load_model(self, model_dir_path):
        config_file_path = ResNetV2AudioClassifier.get_config_file_path(model_dir_path)
        params_file_path = ResNetV2AudioClassifier.get_params_file_path(model_dir_path)
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

        config_file_path = ResNetV2AudioClassifier.get_config_file_path(model_dir_path)

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

        np.save(model_dir_path + '/' + ResNetV2AudioClassifier.model_name + '-history.npy', history)

        return history

    def predict(self, audio_path):
        mg = compute_melgram(audio_path)
        mg = nd.array(np.expand_dims(mg, axis=0), ctx=self.model_ctx)
        return self.model(mg).asnumpy()[0]

    def predict_class(self, audio_path):
        predicted = self.predict(audio_path)
        return np.argmax(predicted)
