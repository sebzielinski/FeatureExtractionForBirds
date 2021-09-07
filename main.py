from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from dtw import *
import numpy as np

import matplotlib.pyplot as plt
[Fs, x] = audioBasicIO.read_audio_file("data/m12s1_1.wav")
# ShortTermFeatures.spectrogram(x, Fs, 0.050*Fs, 0.025*Fs, 1, 1)
F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

# mfcc in feature vector an Stelle 8 bis 20
print("features: ", len(f_names), f_names[8:21])
print("Feature Vectors: ", F.shape)

# for i in range(len(F)):
#     print(F[i][8:21])

[Fs, x] = audioBasicIO.read_audio_file("data/m12s1_2.wav")
# ShortTermFeatures.spectrogram(x, Fs, 0.050*Fs, 0.025*Fs, 1, 1)
F_2, f_names_2 = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

# mfcc in feature vector an Stelle 8 bis 20
print(f_names[8:21])
print("Feature Vectors: ", F_2.shape)

# for i in range(len(F)):
#     print(F[i][8:21])

query = F_2[8:21, :]
template = F[8:21, 0:247]

print("Query dimensions: ", query.shape, type(query))
print("template dimensions: ", template.shape, type(query))

# dynamic time warping (requires feature vectors of same dimensions)
alignment = dtw(query, template, keep_internals=True)
alignment.plot(type="threeway")

# cross correlation
cc = np.correlate(query.flatten(), template.flatten())
print("cross correlated array: ", cc.shape)
print(cc)