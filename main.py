import librosa
import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import time
import argparse
import os

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
parser.add_argument("-n", "--num_mfcc",
                    default=15,
                    type=int,
                    nargs=1,
                    dest="num_mfccs")

args = parser.parse_args()
num_mfccs = args.num_mfccs[0]
print(f"amount of mfccs to use: {num_mfccs}")

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


                """ 
                calculate correlation coefficient for every template/ query pair
                to obtain label and score vectors 
                save calculated mfccs for speedup
                """
                if not birdsong.path in mfccs:
                    # use librosa to read files and get MFCCs
                    y_1, sr_1 = librosa.load(birdsong.path, sr=44100)
                    mfcc_template = librosa.feature.mfcc(y_1, sr_1, n_mfcc=num_mfccs)
                    mfccs = {birdsong.path: mfcc_template}
                else: 
                    # print("mfcc already calculated for: ", birdsong.path)
                    y_1, sr_1 = librosa.load(birdsong.path, sr=44100)
                    mfcc_template = mfccs[birdsong.path]
                if not birdsong_query.path in mfccs:
                    y_2, sr_2 = librosa.load(birdsong_query.path, sr=44100)
                    mfcc_query = librosa.feature.mfcc(y_2, sr_2, n_mfcc=num_mfccs)
                    mfccs = {birdsong.path: mfcc_query}
                else: 
                    # print("mfcc already calculated for: ", birdsong_query.path)
                    y_2, sr_2 = librosa.load(birdsong_query.path, sr=44100)
                    mfcc_query = mfccs[birdsong_query.path]


                # TODO cross correlation
                if (args.cc):
                    # cross correlation to match recordings due to imperfect segmentation
                    # shift recording according to highest peak in cc-array
                    cc = scipy.signal.correlate(y_1, y_2, mode='same', method='fft')
                    
                    # get index of peak 
                    peak_index = np.where(cc == np.amax(cc))
                    
                    # get lag array
                    lag_array = scipy.signal.correlation_lags(len(y_1), len(y_2))
                    lag = lag_array[np.argmax(cc)]
                    
                    if(args.debug_plot):
                        # show template and query
                        ax1 = plt.subplot(211)
                        ax1.set_title("template")
                        ax1.plot(y_1)
                        ax2 = plt.subplot(212)
                        ax2.set_title("query")
                        ax2.plot(y_2)
                        plt.show()
                        
                        # show cc array
                        plt.title(f"lag: {lag/sr_1}")
                        plt.plot(cc)
                        plt.show()

                score = 0
                # calculate score for every MFCC-vector
                for i in range(num_mfccs):
                    # first: cross correlation (can be used on audio with different length)
                    # use this to find position of best match between query and template
                    # normalize before cross correlation
                    query = mfcc_query[i, :] / np.linalg.norm(mfcc_query[i, :])
                    template = mfcc_template[i, :] / np.linalg.norm(mfcc_template[i, :])
                    # use pearson correlation coefficient
                    # problem: vectors have to have same length
                    # solution: pad smaller vector with zeros
                    ml = max(len(query), len(template))
                    query = np.concatenate([query, np.zeros(ml - len(query))])
                    template = np.concatenate([template, np.zeros(ml - len(template))])
                    pearson_cc, bla = scipy.stats.pearsonr(query, template)
                    # print("Pearson correlation coefficient: ", pearson_cc)
                    # take absolute value of cc (high negative values can indicate correlation)
                    score += abs(pearson_cc)
                # print("Score: ", score)
                score /= num_mfccs
                scores.append(score)

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
