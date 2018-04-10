import urllib.request
import os
import tarfile

from mxnet_audio.library.utility.download_utils import reporthook

gtzan_labels = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal',
                7: 'pop', 8: 'reggae', 9: 'rock'}


def download_gtzan_music_speech(data_dir_path):
    zip_file_path = data_dir_path + '/music_speech.tar.gz'

    if not os.path.exists(zip_file_path):
        url_link = 'http://opihi.cs.uvic.ca/sound/music_speech.tar.gz'
        print('gz model file does not exist, downloading from internet')
        urllib.request.urlretrieve(url=url_link, filename=zip_file_path,
                                   reporthook=reporthook)

    tar = tarfile.open(zip_file_path, "r:gz")
    tar.extractall(data_dir_path)
    tar.close()


def download_gtzan_genres_if_not_found(data_dir_path):
    flag_file = os.path.join(data_dir_path, 'genres_done.txt')
    if os.path.exists(flag_file):
        return

    zip_file_path = data_dir_path + '/genres.tar.gz'

    if not os.path.exists(zip_file_path):
        url_link = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
        print('gz model file does not exist, downloading from internet')
        urllib.request.urlretrieve(url=url_link, filename=zip_file_path,
                                   reporthook=reporthook)

    tar = tarfile.open(zip_file_path, "r:gz")
    tar.extractall(data_dir_path)
    tar.close()

    with open(flag_file, 'wt') as file:
        file.write('done')


