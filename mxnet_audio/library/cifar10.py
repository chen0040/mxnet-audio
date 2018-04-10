import numpy as np
from sklearn.model_selection import train_test_split
import mxnet as mx
from mxnet import nd, autograd, gluon
import scipy.misc
from lru import LRU
from mxnet_audio.library.utility.audio_utils import compute_melgram


def cifar10(input_shape, nb_classes):
    channel_axis = 3
    freq_axis = 1
    time_axis = 2

    model = gluon.nn.Sequential()
    with model.name_scope():
        model.add(Conv2D(filters=32, input_shape=input_shape, padding='same', kernel_size=(3, 3)))
        model.add(BatchNormalization(axis=channel_axis))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 4)))

        model.add(Conv2D(filters=32, padding='same', kernel_size=(3, 3)))
        model.add(BatchNormalization(axis=channel_axis))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 4)))

        model.add(Dropout(rate=0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=input_shape))
        model.add(BatchNormalization(axis=channel_axis))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 4)))

        model.add(Conv2D(filters=128, padding='same', kernel_size=(3, 3)))
        model.add(BatchNormalization(axis=channel_axis))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(3, 5)))

        model.add(Conv2D(filters=256, padding='same', kernel_size=(3, 3)))
        model.add(BatchNormalization(axis=channel_axis))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        model.add(Dropout(rate=0.25))
    
        model.add(Flatten())
        model.add(Dense(units=512))
        model.add(Activation('elu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=nb_classes))
        model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


class Cifar10AudioClassifier(object):
    model_name = 'cifar10'

    def __init__(self):
        self.cache = LRU(400)
        self.input_shape = None
        self.nb_classes = None
        self.model = None
        self.config = None

    def create_model(self):
        self.model = cifar10(input_shape=self.input_shape, nb_classes=self.nb_classes)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(self.model.summary())

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10AudioClassifier.model_name + '-config.npy')

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10AudioClassifier.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10AudioClassifier.model_name + '-weights.h5')

    def load_model(self, model_dir_path):
        config_file_path = Cifar10AudioClassifier.get_config_file_path(model_dir_path)
        weight_file_path = Cifar10AudioClassifier.get_weight_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.input_shape = self.config['input_shape']
        self.nb_classes = self.config['nb_classes']
        self.create_model()
        self.model.load_weights(weight_file_path)

    def compute_melgram(self, audio_path):
        if audio_path in self.cache:
            return self.cache[audio_path]
        else:
            mg = compute_melgram(audio_path)
            # mg = (mg + 100) / 200  # scale the values
            self.cache[audio_path] = mg
            return mg

    def generate_batch(self, audio_paths, labels, batch_size):
        num_batches = len(audio_paths) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size

                X = np.zeros(shape=(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
                for i in range(start, end):
                    audio_path = audio_paths[i]
                    mg = compute_melgram(audio_path)
                    X[i - start, :, :, :] = mg
                yield X, labels[start:end]

    def fit(self, audio_path_label_pairs, model_dir_path, batch_size=None, epochs=None, test_size=None,
            random_state=None, input_shape=None, nb_classes=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 20
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 42
        if input_shape is None:
            input_shape = (96, 1366, 1)
        if nb_classes is None:
            nb_classes = 10

        config_file_path = Cifar10AudioClassifier.get_config_file_path(model_dir_path)
        weight_file_path = Cifar10AudioClassifier.get_weight_file_path(model_dir_path)
        architecture_file_path = Cifar10AudioClassifier.get_architecture_file_path(model_dir_path)

        self.input_shape = input_shape
        self.nb_classes = nb_classes

        self.config = dict()
        self.config['input_shape'] = input_shape
        self.config['nb_classes'] = nb_classes
        np.save(config_file_path, self.config)

        self.create_model()

        with open(architecture_file_path, 'wt') as file:
            file.write(self.model.to_json())

        checkpoint = ModelCheckpoint(weight_file_path)

        X = []
        Y = []

        for audio_path, label in audio_path_label_pairs:
            X.append(audio_path)
            Y.append(label)

        X = np.array(X)
        Y = np.array(Y)

        Y = np_utils.to_categorical(Y, self.nb_classes)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        np.save(os.path.join(model_dir_path, Cifar10AudioClassifier.model_name + '-history.npy'), history.history)
        return history

    def predict(self, audio_path):
        mg = compute_melgram(audio_path)
        mg = np.expand_dims(mg, axis=0)
        return self.model.predict(mg)[0]

    def predict_class(self, audio_path):
        predicted = self.predict(audio_path)
        return np.argmax(predicted)

    def export_tensorflow_model(self, output_fld, output_model_file=None,
                                 output_graphdef_file=None,
                                 num_output=None,
                                quantize=False,
                                save_output_graphdef_file=False,
                                 output_node_prefix=None):

        K.set_learning_phase(0)

        if output_model_file is None:
            output_model_file = Cifar10AudioClassifier.model_name + '.pb'

        if output_graphdef_file is None:
            output_graphdef_file = 'model.ascii'
        if num_output is None:
            num_output = 1
        if output_node_prefix is None:
            output_node_prefix = 'output_node'

        pred = [None] * num_output
        pred_node_names = [None] * num_output
        for i in range(num_output):
            pred_node_names[i] = output_node_prefix + str(i)
            pred[i] = tf.identity(self.model.outputs[i], name=pred_node_names[i])
        print('output nodes names are: ', pred_node_names)

        sess = K.get_session()

        if save_output_graphdef_file:
            tf.train.write_graph(sess.graph.as_graph_def(), output_fld, output_graphdef_file, as_text=True)
            print('saved the graph definition in ascii format at: ', output_graphdef_file)

        from tensorflow.python.framework import graph_util
        from tensorflow.python.framework import graph_io
        from tensorflow.tools.graph_transforms import TransformGraph
        if quantize:
            transforms = ["quantize_weights", "quantize_nodes"]
            transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
            constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
        else:
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
        graph_io.write_graph(constant_graph, output_fld, output_model_file, as_text=False)
        print('saved the freezed graph (ready for inference) at: ', output_model_file)