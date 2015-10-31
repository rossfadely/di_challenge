from time import time
from matplotlib import use; use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

def convert_data(df, outbase='../data/ny_discharge_2013', shuffle=True,
                 seed=1234):
    """
    Prepare the data for prediction tasks and save to file.
    """
    # hand selected columns to use as features.
    cols = ['Facility Id', 'Age Group', 'Type of Admission',
            'CCS Diagnosis Code', 'APR DRG Code',
            'APR MDC Code', 'APR Severity of Illness Code',
            'Source of Payment 1', 'Abortion Edit Indicator',
            'Emergency Department Indicator']

    # data converted to ints
    outfile = outbase + '_prepped.dat'
    try:
        print 'Loading prepped file.'
        data = np.loadtxt(outfile)
    except:
        data = np.zeros((df.shape[0], len(cols)), np.int)
        for i in range(len(cols)):
            print i
            arr = np.array(df[cols[i]], np.str)
            u = np.unique(arr)
            for j in range(u.size):
                ind = arr == u[j]
                data[ind, i] = j + 1
        print 'Saving converted array.'
        np.savetxt(outfile, data, fmt='%d', delimiter=' ')

    outfile = outbase + '_targets.dat'
    try:
        print 'Loading target file.'
        targets = np.loadtxt(outfile)
    except:
        targets = np.zeros((df.shape[0], 2))
        arr = np.array(df['Length of Stay'], np.str)
        ind = arr == '120 +'
        arr[ind] = '120'
        arr = arr.astype(np.float)
        targets[:, 0] = arr
        for i in range(targets.shape[0]):
            if i % 20000 == 0:
                print i
            targets[i, 1] = np.float(df['Total Charges'][i][1:])
        print 'Saving target array'
        np.savetxt(outfile, targets, fmt='%0.2f', delimiter=' ')
        
    if shuffle:
        np.random.seed(seed)
        ind = np.random.permutation(data.shape[0])
        data = data[ind]
        targets = targets[ind]

    return data, targets

def encode_data(data):
    """
    Encode the categorical data as one hot.
    """
    print 'Encoding data.'
    from time import time
    enc = OneHotEncoder()
    t=time()
    enc.fit(data)
    x = enc.transform(data).toarray()
    return x

def split(x, y, test_frac=0.2):
    """
    Return a split of the data into train/test.
    """
    print 'Train/test split.'
    Ntst = np.int(np.round(y.size * test_frac))
    xtrn = x[Ntst:]
    ytrn = y[Ntst:]
    xtst = x[:Ntst]
    ytst = y[:Ntst]
    return xtrn, ytrn, xtst, ytst

def run_rf(xtrn, ytrn, xtst, ytst, filename, nest=100):
    """
    Run a random forest on the data, predict on the test set and record.
    """
    print 'Running RF.'
    t0 = time()
    rgr = RandomForestRegressor(n_estimators=nest)
    rgr.fit(xtrn, ytrn)
    print 'Done in %0.1f sec' % (time() - t0)
    preds = rgr.predict(xtst)
    results = np.zeros((preds.size, 2))
    results[:, 0] = ytst
    results[:, 1] = preds
    np.savetxt(filename, results, fmt='%0.2f')

def generate_plots(results_file):
    """
    Plot the results from the RF Regression.
    """
    data = np.loadtxt(results_file)

    if 'stay' in results_file:
        data[:, 0] += np.random.rand(data.shape[0])
        test = 'stay'
        units = '(days)'
        lim = (0, 20)
    else:
        test = 'cost'
        units = '(dollars)'
        lim = (1000, 50000)
    fs = 5
    f = pl.figure(figsize=(fs, fs))
    pl.plot(data[:, 0], data[:, 1], 'k.', alpha=0.2)
    pl.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    pl.plot(lim, lim, 'r')
    pl.xlabel('True %s %s' % (test, units))
    pl.ylabel('Predicted %s %s' % (test, units))
    pl.xlim(lim)
    pl.ylim(lim)
    f.savefig('../plots/%s.png' % test)

if __name__ == '__main__':

    if False:
        filename = '../data/ny_discharge_2013.csv'
        df = pd.read_csv(filename)

    results_file = '../data/est_cost_300.txt'
    if True:
        if 'stay' in results_file:
            ind = 0
        else:
            ind = 1
        try:
            data, targets = convert_data(df)
        except:
            data, targets = convert_data(None)
        encoded = encode_data(data)
        Ntot = 300000
        encoded = encoded[:Ntot]
        targets = targets[:Ntot]
        xtrn, ytrn, xtst, ytst = split(encoded, targets[:, ind])
        run_rf(xtrn, ytrn, xtst, ytst, results_file, nest=20)

    if False:
        generate_plots(results_file)
