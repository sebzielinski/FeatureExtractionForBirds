import time
import argparse
import os
import scipy
import numpy as np
import numpy.ma as ma
import librosa


import matplotlib.pyplot as plt
from sklearn import metrics

from utils import calculateMFCCs, calculateScore

cross_corr = True
librsa = True

directory = "data/labeled"
print("calculating labels and scores for given bird song audio files...")
num_file = 1

labels = []
scores = []
mfccs = {}
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--debug",
                    action="store_true",
                    dest="debug_plot"
                    )
parser.add_argument("-c", "--cross_correlation",
                    action="store_true",
                    dest="cc"
                    )
parser.add_argument("-v", "--verbose",
                    action="store_true",
                    dest="verbose"
                    )
parser.add_argument("-n", "--num_mfcc",
                    default=15,
                    type=int,
                    nargs='?',
                    dest="num_mfccs"
                    )
parser.add_argument("-s", "--sample_rate",
                    default=44100,
                    type=int,
                    nargs='?',
                    dest="sample_rate"
                    )
parser.add_argument("-D", "--dtw",
                    action="store_true",
                    dest="dtw"
                    )

args = parser.parse_args()
num_mfccs = args.num_mfccs
print(f"amount of mfccs to use: {num_mfccs}")
print(f"sample rate: {args.sample_rate}")

# timing
start_time = time.perf_counter()

for birdsong in os.scandir(directory):

    print("num_file: ", num_file)
    num_file += 1

    if birdsong.is_file():
        # print(birdsong.path)
        filename = birdsong.path.split("/")[2].split(".")[0]
        # print(filename)
        filename = filename.split("_")
        individual = filename[0][1]
        recording = filename[1]
        # print(individual)
        # print(recording)
        # print("Template: individual: ", individual, ", recording: ", recording)

        file_template = birdsong.path
        # file_template = directory + "/m" + individual + "_" + recording + ".wav"
        # print(file_template)

        for birdsong_query in os.scandir(directory):
            if birdsong_query.is_file():
                # if not (birdsong.path == birdsong_query.path):
                # print("template: ", birdsong.path, ", query: ", birdsong_query.path)
                filename_query = birdsong_query.path.split("/")[2].split(".")[0]
                # print(filename)
                filename_query = filename_query.split("_")
                individual_query = filename_query[0][1]
                recording_query = filename_query[1]

                # fill in correct labels. 1 if same individual, 0 if different individual
                if (individual == individual_query):
                    labels.append(1)
                else:
                    labels.append(0)

                if (args.cc):
                    # load audios
                    y_1, sr_1 = librosa.load(birdsong.path, sr=args.sample_rate)
                    y_2, sr_2 = librosa.load(birdsong_query.path, sr=args.sample_rate)
                    
                    # cross correlation to match recordings due to imperfect segmentation
                    # shift recording according to highest peak in cc-array
                    cc = scipy.signal.correlate(y_1, y_2, mode='full', method='fft')
                    
                    # get index of peak 
                    peak_index = np.where(cc == np.amax(cc))
                    
                    
                    # shift the shorter audio, otherwise problems may occur
                    if (len(y_1) <= len(y_2)):
                        # get lag array
                        lag_array = scipy.signal.correlation_lags(len(y_1), len(y_2), mode='full')
                        lag = lag_array[np.argmax(cc)]
                        
                        # shift the array according to the lag
                        shifted = scipy.ndimage.interpolation.shift(y_2, lag, mode="constant")
                        y_2 = shifted
                    else:
                        # get lag array
                        lag_array = scipy.signal.correlation_lags(len(y_2), len(y_1), mode='full')
                        lag = lag_array[np.argmax(cc)]
                        
                        # shift the array according to the lag
                        shifted = scipy.ndimage.interpolation.shift(y_1, lag, mode="constant")
                        y_1 = shifted
                        
                    if(args.debug_plot):
                        # show template and query
                        ax1 = plt.subplot(221)
                        ax1.set_title(f"template: {birdsong.path}")
                        ax1.plot(y_1)
                        ax2 = plt.subplot(222)
                        ax2.set_title(f"query: {birdsong_query.path}")
                        ax2.plot(y_2)
                        
                        # show cc array
                        ax3 = plt.subplot(223)
                        ax3.set_title(f"shifted")
                        ax3.plot(shifted)
                        plt.tight_layout()
                        plt.show()
                    
                    
                # TODO Figure out caching with shifted audios
                """ 
                calculate correlation coefficient for every template/ query pair
                to obtain label and score vectors 
                save calculated mfccs for speedup
                """
                # if not args.cc:
                #     if not birdsong.path in mfccs:
                #         mfcc_template = calculateMFCCs(birdsong.path, sr=args.sample_rate)
                #         mfccs[birdsong.path] = mfcc_template
                #     else:
                #         if args.verbose:
                #             print("mfcc already calculated for template: ", birdsong.path)
                #         mfcc_template = mfccs[birdsong.path]
                #     if not birdsong_query.path in mfccs:
                #         mfcc_query = calculateMFCCs(birdsong_query.path, sr=args.sample_rate)
                #         mfccs[birdsong_query.path] = mfcc_query
                #     else: 
                #         if args.verbose:
                #             print("mfcc already calculated for query: ", birdsong_query.path)
                #         mfcc_query = mfccs[birdsong_query.path]
                        
                mfcc_template = librosa.feature.mfcc(y_1, sr_1, n_mfcc=num_mfccs)
                mfcc_query = librosa.feature.mfcc(y_2, sr_2, n_mfcc=num_mfccs)
                
                # mfcc_query has apparently NaN and infs values due to shifting
                # reason: query audio is way longer, therefore it can happen that shifted audio is just Null
                # solution: shift the shorter audio
                # mfcc_query = ma.masked_invalid(mfcc_query)
                try:
                    score = calculateScore(num_mfccs, mfcc_query, mfcc_template)
                    scores.append(score)
                except ValueError:
                    print(f"lag: {lag}")
                    print(f"lag (s): {lag/sr_2}")
                    ax1 = plt.subplot(221)
                    ax1.set_title(f"template: {birdsong.path}")
                    ax1.plot(y_1)
                    ax2 = plt.subplot(222)
                    ax2.set_title(f"query: {birdsong_query.path}")
                    ax2.plot(y_2)
                    
                    # show cc array
                    ax3 = plt.subplot(223)
                    ax3.set_title(f"shifted")
                    ax3.plot(shifted)
                    plt.tight_layout()
                    plt.show()

end_time = time.perf_counter()
print(f"calculation took {end_time - start_time:0.4f} seconds")

fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
roc_auc = metrics.auc(fpr, tpr)
# print(fpr)
# print(tpr)
print(f"Thresholds: \n {thresholds} ")
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
# print(labels)
# plt.plot(labels)
# plt.plot(scores)
plt.show()


print("done.\n")
