from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from dtw import *
import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt

dyn_time_warp = False
cross_corr = True

# use pyAudioAnalysis to read audio files
[Fs, x] = audioBasicIO.read_audio_file("data/mitnoise.wav")
# ShortTermFeatures.spectrogram(x, Fs, 0.050*Fs, 0.025*Fs, 1, 1)

# use pyAudioAnalysis to extract relevant features
F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

# mfcc in feature vector an Stelle 8 bis 20
print("features: ", len(f_names), f_names[8:21])
print("Feature Vectors: ", F.shape)

# for i in range(len(F)):
#     print(F[i][8:21])

[Fs, x] = audioBasicIO.read_audio_file("data/m16s1_2.wav")
# ShortTermFeatures.spectrogram(x, Fs, 0.050*Fs, 0.025*Fs, 1, 1)
F_2, f_names_2 = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

# mfcc in feature vector an Stelle 8 bis 20
print(f_names[8:21])
print("Feature Vectors: ", F_2.shape)

# for i in range(len(F)):
#     print(F[i][8:21])

# use MFCC features only (index 8:21)
query = F_2[8:21, :]
template = F[8:21, :]

print("Query dimensions: ", query.shape, type(query))
print("template dimensions: ", template.shape, type(query))

# dynamic time warping (requires feature vectors of same dimensions)
# limitation: can only be used on audio with exact same length
if (dyn_time_warp):
    alignment = dtw(query, template, keep_internals=True)
    print("alignement: ", type(alignment))
    print(alignment)
    alignment.plot(type="threeway")

if (cross_corr):
    # cross correlation (can be used on audio with different length)
    cc = scipy.signal.correlate2d(query, template)
    print("cross correlated array: ", cc.shape, type(cc))
    # find index of match
    y, x = np.unravel_index(np.argmax(cc), cc.shape)
    print(cc[y, x], y, x)
    plt.plot(cc[y, :])
    plt.show()
