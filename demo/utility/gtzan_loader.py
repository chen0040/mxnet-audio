from mxnet_audio.library.utility.audio_utils import compute_melgram
from mxnet_audio.library.utility.gtzan_loader import download_gtzan_genres_if_not_found
import numpy as np

def load_audio_path_label_pairs(max_allowed_pairs=None):
    download_gtzan_genres_if_not_found('../very_large_data/gtzan')
    audio_paths = []
    with open('../data/lists/test_songs_gtzan_list.txt', 'rt') as file:
        for line in file:
            audio_path = '../very_large_data/' + line.strip()
            audio_paths.append(audio_path)
    pairs = []
    with open('../data/lists/test_gt_gtzan_list.txt', 'rt') as file:
        for line in file:
            label = int(line)
            if max_allowed_pairs is None or len(pairs) < max_allowed_pairs:
                pairs.append((audio_paths[len(pairs)], label))
            else:
                break
    return pairs


def main():
    pairs = load_audio_path_label_pairs()
    for index, (audio_path, _) in enumerate(pairs):
        print('{} / {} ...'.format(index+1, len(pairs)))
        mg = compute_melgram(audio_path)
        print('max: ', np.max(mg))
        print('min: ', np.min(mg))



if __name__ == '__main__':
    main()
