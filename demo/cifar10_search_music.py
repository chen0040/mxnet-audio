import os
import sys


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def load_audio_path_label_pairs(max_allowed_pairs=None):
    from mxnet_audio.library.utility.gtzan_loader import download_gtzan_genres_if_not_found
    download_gtzan_genres_if_not_found(patch_path('very_large_data/gtzan'))
    audio_paths = []
    with open(patch_path('data/lists/test_songs_gtzan_list.txt'), 'rt') as file:
        for line in file:
            audio_path = patch_path('very_large_data/' + line.strip())
            audio_paths.append(audio_path)
    pairs = []
    with open(patch_path('data/lists/test_gt_gtzan_list.txt'), 'rt') as file:
        for line in file:
            label = int(line)
            if max_allowed_pairs is None or len(pairs) < max_allowed_pairs:
                pairs.append((audio_paths[len(pairs)], label))
            else:
                break
    return pairs


def main():
    sys.path.append(patch_path('..'))

    from mxnet_audio.library.cifar10 import Cifar10AudioSearch

    search_engine = Cifar10AudioSearch()
    search_engine.load_model(model_dir_path=patch_path('models'))
    for path, _ in load_audio_path_label_pairs():
        search_engine.index_audio(path)

    query_audio = './data/audio_samples/example.mp3'
    search_result = search_engine.query(query_audio, top_k=10)

    for idx, similar_audio in enumerate(search_result):
        print('result #%s: %s' % (idx+1, similar_audio))


if __name__ == '__main__':
    main()
