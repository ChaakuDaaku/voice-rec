import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os


data_path = os.path.dirname('wavs/')
data = sorted(os.listdir(data_path))[11789:]
target_path = os.path.dirname('data/test/')

for audio in data:
    audio_path = os.path.join(data_path, audio)
    sample_rate, samples = wavfile.read(audio_path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    plt.pcolormesh(times, frequencies, spectrogram)
    fig = plt.figure()
    fig.patch.set_visible(False)
    ax = fig.add_subplot(111)
    plt.axis('off')
    plt.imshow(spectrogram)
    plt.set_cmap('gray')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(target_path, os.path.splitext(audio)[0]+'.png'), bbox_inches=extent)
    plt.close()