import librosa
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from dtw import *
import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt
from librosa import *

dyn_time_warp = False
cross_corr = True
librsa = True

if (librsa):
    # use librosa to read files and get MFCCs
    y, sr = librosa.load("data/Sylvia atricapilla spain_2_1.wav", sr=44100)
    mfcc_template = librosa.feature.mfcc(y, sr, n_mfcc=13)
    print("librosa mfccs: ", mfcc_template.shape)

    y, sr = librosa.load("data/Sylvia atricapilla spain_2_2.wav", sr=44100)
    mfcc_query = librosa.feature.mfcc(y, sr, n_mfcc=13)

else:
    # use pyAudioAnalysis to read audio files
    [Fs, x] = audioBasicIO.read_audio_file("data/mitnoise.wav")
    # ShortTermFeatures.spectrogram(x, Fs, 0.050*Fs, 0.025*Fs, 1, 1)

    # use pyAudioAnalysis to extract relevant features
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs, deltas=False)
    # plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
    # plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

    # mfcc in feature vector an Stelle 8 bis 20
    print("features: ", len(f_names), f_names[8:21])
    print("Feature Vectors: ", F.shape)

    # for i in range(len(F)):
    #     print(F[i][8:21])

    [Fs, x] = audioBasicIO.read_audio_file("data/Sylvia atricapilla spain_1_1.wav")
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



# dynamic time warping (requires feature vectors of same dimensions)
# limitation: can only be used on audio with exact same length
if (dyn_time_warp):
    alignment = dtw(query, template, keep_internals=True)
    print("alignement: ", type(alignment))
    print(alignment)
    alignment.plot(type="threeway")

if (cross_corr):

    if (librsa):
        score = 0
        for i in range(13):
            query = mfcc_query[i, :] / np.linalg.norm(mfcc_query[i, :])
            # query = mfcc_query[i, :]
            template = mfcc_template[i, :] / np.linalg.norm(mfcc_template[i, :])
            # template = mfcc_template[i, :]
            print("Query dimensions: ", query.shape, type(query), query.dtype)
            print("template dimensions: ", template.shape, type(template), template.dtype)
            # cross correlation (can be used on audio with different length)
            # normalize before cross correlation
            cc = scipy.signal.correlate(query, template, mode='valid')
            print("cross correlated array: ", cc.shape, type(cc))

            # use pearson correlation coefficient
            # problem: vectors have to have same length
            # solution: pad smaller vector with zeros
            ml = max(len(query), len(template))
            query = np.concatenate([query, np.zeros(ml - len(query))])
            template = np.concatenate([template, np.zeros(ml - len(template))])
            pearson_cc, bla = scipy.stats.pearsonr(query, template)
            print("Pearson correlation coefficient: ", pearson_cc)
            # find index of match
            # y, x = np.unravel_index(np.argmax(cc), cc.shape)
            # print(cc[y, x], y, x)
            # for i in range(25):
            #     plt.plot(cc[i, :])
            # plt.plot(cc[y, :])
            # score_loc = cc[np.argmax(cc)]
            # score_loc = np.sum(cc)
            # eucl_dist = np.linalg.norm(mfcc_query[i, :] - mfcc_template[i, :])
            # cc /= np.linalg.norm(cc)
            # score_loc = cc[np.argmax(cc)] / np.median(cc)
            # print(score_loc)
            plt.title("mfcc " + str(i + 1))
            plt.plot(cc)
            # plt.show()
            # score += score_loc
            score += pearson_cc

        print("Score: ", score)
    else:
        for i in range(13):
            query = F_2[i+8, :]
            template = F[i+8, :]
            print("Query dimensions: ", query.shape, type(query))
            print("template dimensions: ", template.shape, type(query))
            # cross correlation (can be used on audio with different length)
            cc = scipy.signal.correlate(query, template)
            print("cross correlated array: ", cc.shape, type(cc))
            # find index of match
            # y, x = np.unravel_index(np.argmax(cc), cc.shape)
            # print(cc[y, x], y, x)
            # for i in range(25):
            #     plt.plot(cc[i, :])
            # plt.plot(cc[y, :])
            plt.title("mfcc" + str(i+1))
            plt.plot(cc)
            plt.show()

