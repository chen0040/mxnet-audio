from mxnet_audio.library.cifar10 import Cifar10AudioClassifier
from mxnet_audio.library.utility.gtzan_loader import download_gtzan_genres_if_not_found
import mxnet

def load_audio_path_label_pairs(max_allowed_pairs=None):
    download_gtzan_genres_if_not_found('./very_large_data/gtzan')
    audio_paths = []
    with open('./data/lists/test_songs_gtzan_list.txt', 'rt') as file:
        for line in file:
            audio_path = './very_large_data/' + line.strip()
            audio_paths.append(audio_path)
    pairs = []
    with open('./data/lists/test_gt_gtzan_list.txt', 'rt') as file:
        for line in file:
            label = int(line)
            if max_allowed_pairs is None or len(pairs) < max_allowed_pairs:
                pairs.append((audio_paths[len(pairs)], label))
            else:
                break
    return pairs


def main():
    audio_path_label_pairs = load_audio_path_label_pairs()
    print('loaded: ', len(audio_path_label_pairs))

    classifier = Cifar10AudioClassifier(model_ctx=mxnet.gpu(0), data_ctx=mxnet.gpu(0))
    batch_size = 8
    epochs = 100
    history = classifier.fit(audio_path_label_pairs, model_dir_path='./models',
                             batch_size=batch_size, epochs=epochs,
                             checkpoint_interval=2)


if __name__ == '__main__':
    main()
