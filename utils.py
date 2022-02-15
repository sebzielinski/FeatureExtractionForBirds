import re
import librosa
import numpy as np
import scipy

# TODO add all needed utility functionality here

def calculateMFCCs(path, sr=44100, n_mfcc=15):
    """ calculates MFCCs and returns them
    
    @param:     path    The path to the audio file
    @param:     sr      desired sample rate
    @returns:   mfccs   the calculated mfcc feature vector
    """
    
    y_1, sr_1 = librosa.load(path, sr=sr)
    mfccs = librosa.feature.mfcc(y_1, sr_1, n_mfcc=n_mfcc)
    
    return mfccs


def calculateScore(n_mfccs, mfcc_query, mfcc_template):
    """ Calculates a score to determine, how similar two mfcc vectors are
        Uses Pearson Correlation Coefficient that produces Score between -1 and 1
            0  --> no correlation
            1  --> high correlation
            -1 --> high inverse correlation
    
    @param   n_mfccs    number of mfccs of the feature vector
    @param   query      query feature vector
    @param   template   template feature vector
    @returns score      the calculated similarity score
    """
    
    score = 0
    # calculate score for every MFCC-vector
    for i in range(n_mfccs):
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
    score /= n_mfccs
    
    return score


def calculateCrossCorrelation(query, template):
    # cross correlation to match recordings due to imperfect segmentation
    # shift recording according to highest peak in cc-array
    
    cc = scipy.signal.correlate(query, template, mode='full', method='fft')
    
    # get index of peak 
    peak_index = np.where(cc == np.amax(cc))
    
    # get lag array
    lag_array = scipy.signal.correlation_lags(len(query), len(template), mode='full')
    lag = lag_array[np.argmax(cc)]
    
    return cc, lag

