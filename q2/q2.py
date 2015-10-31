import numpy as np
import pandas as pd

from sklearn.mixture import GMM

def part_1(df):
    """
    Compute the fraction of complaints from second most popular agency.

    prints 'Answer to part 1 is 0.1719314121'
    """
    hist = pd.Series(df['Agency']).value_counts()
    fraction = 1. * hist[1] / hist.sum()
    print 'Answer to part 1 is %0.10f\n' % fraction

def part_2(df):
    """
    Compute the distance in degrees between the 10% and 90% quantiles of
    Latitude.

    print 'Answer to part 2 is 0.2357908310'
    """
    diff = df['Latitude'].quantile(0.9) - df['Latitude'].quantile(0.1)
    print 'Answer to part 2 is %0.10f\n' % diff

def part_3(df):
    """
    Compute the difference betweent the number of calls between the
    most and least popular whole hours, removing bad time stamps.

    prints 'Answer to part 3 is 0.0616877918'
    returns cleaned list of datetimes
    """
    # convert to 24hr clock, this takes a bit of time.
    datetimes = pd.to_datetime(df['Created Date'])

    # decied what to throw away.
    hist = pd.Series(datetimes).value_counts()
    print 'There are %d timestamps.' % hist.size
    print 'There are %d timestamps which are completely unique.' % \
        (hist[hist==1].size)
    print 'Requiring that the stamps are completely unique for accurate'
    print 'times.'
    datetimes = hist[hist==1].keys()

    # compute difference in total counts for most/least popular
    hours, counts = np.unique(datetimes.hour, return_counts=True)
    counts = np.sort(counts)
    diff = counts[-1] - counts[0]
    fraction = 1. * diff / counts.sum()
    print 'Answer to part 3 is %0.10f\n' % (fraction)
    return datetimes

def part_4(df):
    """
    Compute the most surprising complaint type for a borough.

    prints 'Answer to part 4 is 18.2636539395'
    """
    # unconditional probabilities
    counts = pd.Series(df['Complaint Type']).value_counts()
    unc_probs = counts / counts.sum()
    complaints = unc_probs.keys()

    boroughs = pd.Series(df['Borough']).value_counts().keys()
    boroughs = boroughs[boroughs != 'Unspecified'] # remove non-specified
    max_ratio = 0.0

    # for each borough, go through complains and record max ratio
    ratios = np.zeros(boroughs.size)
    for i, b in enumerate(boroughs):
        cdf = df.query('Borough == "%s"' % b)
        counts = pd.Series(cdf['Complaint Type']).value_counts()
        probs = counts / counts.sum()
        cond_complaints = probs.keys()
        for j in range(complaints.size):
            ind = complaints[j] == cond_complaints
            if np.any(ind):
                ratios[i] = np.maximum(ratios[i], probs[ind] / unc_probs[j])

    print 'Answer to part 4 is %0.10f\n' % np.max(ratios)

def part_5(df):
    """
    Compute the area in square km of the 1 sigma ellipse for a 2D gaussian fit 
    to latitudes and longitudes.

    prints 'Answer to part 5 is 225.8509156285\n'
    """
    obs = np.zeros((df.shape[0], 2))
    obs[:, 0] = df['Latitude']
    obs[:, 1] = df['Longitude']
    ind = (~np.isnan(obs[:, 0])) | (~np.isnan(obs[:, 1]))
    obs = obs[ind]

    # fit a 2D gaussian using scikit-learn's GMM with k=1
    gmm = GMM(covariance_type='full')
    gmm.fit(obs)
    cov = gmm.covars_[0] # note this is a 3d object because of GMM.

    # compute area
    u, s, v = np.linalg.svd(cov)
    rot_cov = np.dot(u, np.dot(cov, u.T))
    varx, vary = np.diag(rot_cov)
    # for now, have to estimate conversion
    # using http://www.nhc.noaa.gov/gccalc.shtml
    sigx = np.sqrt(varx)
    sigy = np.sqrt(vary)
    sigx *= 84. # estimated km / deg at 40.7127N, 74.0059W
    sigy *= 111. # estimated km / deg at 40.7127N, 74.0059W
    area = np.pi * sigx * sigy # area = pi * siga * sigb when rot.
    print 'Answer to part 5 is %0.10f\n' % area

def part_6(datetimes):
    """
    Compute the STD between consecutive calls, using the scrubbed list
    of timestamps.
    
    This is a slow, bad implementation.  Return to fix if have time.

    prints 'Answer to part 6 is 15.9620746400'
    """
    ordered = datetimes.order()
    diffs = np.zeros(ordered.size - 1)
    for i in range(diffs.size):
        diffs[i] = (ordered[i + 1] - ordered[i]).seconds
    print 'Answer to part 6 is %0.10f\n' % np.std(diffs)


if __name__ == '__main__':

    filename = '../data/nyc311calls.csv.gz'

    df = pd.read_csv(filename, compression='gzip',
                     usecols=[1, 3, 5, 24, 50, 51])
    part_1(df)
    part_2(df)
    datetimes = part_3(df)
    part_4(df)
    part_5(df)
    part_6(datetimes)
