from datetime import datetime
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from datetime import timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Initialize a Spark configuration
conf = SparkConf().setAppName("")
sc = SparkContext(conf=conf)

# Create a SparkSession
spark = SparkSession(sc)

# Define the tickers for the factors you want to fetch from Yahoo Finance
factor_tickers = ['CL=F', '^IRX', '^GSPC', '^IXIC']

# Define the start and end dates for the data
start_date = '2018-01-23'
end_date = '2023-01-23'

# Fetch data from Yahoo Finance for the specified tickers and date range
factor_data = yf.download(factor_tickers, start=start_date, end=end_date)

# Extract the adjusted closing prices for the factors
factors = factor_data['Adj Close']

# Define the names for the factors (for reference)
factor_names = ['Crude Oil', 'Treasury Bonds', 'GSPC', 'IXIC']

# Convert the factor data into a list of tuples (date, value) for each factor
factor_tuples = [(factors.index[i], factors[factor].iloc[i]) for i in range(len(factors)) for factor in factor_names]

# Split the factor data into separate lists for each factor
factors1 = factor_tuples[:len(factors)]  # Crude Oil and Treasury Bonds
factors2 = factor_tuples[len(factors):]  # GSPC and IXIC

# Print the first 5 entries of the first factor (Crude Oil)
print('First 5 entries of the first factor (Crude Oil):')
print(factors1[:5])

# Print the first 5 entries of the second factor (GSPC)
print('First 5 entries of the second factor (GSPC):')
print(factors2[:5])

print('Boudary of factor1:')
print(factors2[0][0])
print(factors2[0][-1])
print('size:', len(factors2[0]))
print('\n')
print('Boundary of factor2:')
print(factors2[1][0])
print(factors2[1][-1])
print('size:', len(factors2[1]))

from os import listdir
from os.path import isfile, join
from datetime import datetime

stock_folder = base_folder + 'stocks'

def process_stock_file(fname):
    try:
        return readYahooHistory(fname)
    except Exception as e:
        raise e
        return None



# select path of all stock data files in "stock_folder"
files = [join(stock_folder, f) for f in listdir(stock_folder) if isfile(join(stock_folder, f))]


# assume that we invest only the first 35 stocks (for faster computation)
files = files[:35]

# read each line in each file, convert it into the format: (date, value)
rawStocks = [ process_stock_file(f) for f in files ]

# select only instruments which have more than 5 years of history
# Note: the number of business days in a year is 260
number_of_years = 5
# now = datetime.datetime.now()
# year = now.year

rawStocks = list(filter(lambda instrument: (((instrument[-1][0] - instrument[0][0]).days)/260) >= number_of_years , rawStocks))

# For testing, print the first 5 entry of the first stock
print('first 5 entry of the first stock:', '\n', rawStocks[0][:5])

print('number of stocks having more than five years of history:', '\n',len(rawStocks))

start = datetime(year=2018, month=1, day=23)
end = datetime(year=2023, month=1, day=23)

def trimToRegion(history, start, end):
    def isInTimeRegion(entry):
        (date, value) = entry
        return date >= start and date <= end

    # only select entries which are in the time region
    trimmed = list(filter( lambda entry: (isInTimeRegion(entry) == True), history))

    # if the data has incorrect time boundaries, add time boundaries
    if trimmed[0][0] != start:
        trimmed.insert(0, (start, trimmed[0][1]))
    if trimmed[-1][0] != end:
        trimmed.append((end, trimmed[-1][1]))
    return trimmed

# test our function
trimmedStock0  = trimToRegion(rawStocks[0], start, end)
# the first 5 records of stock 0
print('the first 5 records of stock 0:','\n', trimmedStock0[:5])
# the last 5 records of stock 0
print('the last 5 records of stock 0:','\n', trimmedStock0[-5:])

assert(trimmedStock0[0][0] == start), "the first record must contain the price in the first day of time interval"
assert(trimmedStock0[-1][0] == end), "the last record must contain the price in the last day of time interval"


def fillInHistory(history, start, end):
     curr = history
     filled = []
     idx = 0
     curDate = start
     numEntries = len(history)
     while curDate < end:

         # if the next entry is in the same day
         # or the next entry is at the weekend
         # but the curDate has already skipped it and moved to the next monday
         # (only in that case, curr[idx + 1][0] < curDate )
         # then move to the next entry
         while idx + 1 < numEntries and curr[idx + 1][0] <= curDate:
             idx += 1

         # only add the last value of instrument in a single day
         # check curDate is weekday or not
         # 0: Monday -> 5: Saturday, 6: Sunday
         if curDate.weekday() < 5:
             filled.append((curDate, curr[idx][1]))
             # move to the next business day
             curDate += timedelta(days=1)

         # skip the weekends
         if curDate.weekday() >= 5:
             # if curDate is Sat, skip 2 days, otherwise, skip 1 day
             curDate += timedelta(days=(7 - curDate.weekday()))

     return filled

# trim into a specific time region
# and fill up the missing values
stocks = list(map(lambda stock: \
            fillInHistory(
                trimToRegion(stock, start, end),
            start, end),
        rawStocks))



# merge two factors, trim each factor into a time region
# and fill up the missing values
allfactors = factors1 + factors2
factors = list(map(lambda factor: \
            fillInHistory(
                trimToRegion(factor, start, end),
            start, end),
            allfactors
            ))

# test our code
print("the first 5 records of stock 0:", stocks[0][:5], "\n")
print("the last 5 records of stock 0:", stocks[0][-5:], "\n")
print("the first 5 records of factor 0:", factors[0][:5], "\n")
print("the first 5 records of factor 0:", factors[0][-5:], "\n")

def buildWindow(seq, k=2):
    "Returns a sliding window (of width k) over data from iterable data structures"
    "   s -> (s0,s1,...s[k-1]), (s1,s2,...,sk), ...                   "
    it = iter(seq)
    result = tuple(islice(it, k))
    if len(result) == k:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def calculateReturn(window):
    # return the change of value after two weeks
    return window[-1][1] - window[0][1]

def twoWeekReturns(history):
    # we use 10 instead of 14 to define the window
    # because financial data does not include weekends
    return [calculateReturn(entry) for entry in buildWindow(history, 10)]

stocksReturns = list(map(twoWeekReturns, stocks))
factorsReturns = list(map(twoWeekReturns, factors))

# test our functions
print("the first 5 returns of stock 0:", stocksReturns[0][:5])
print("the last 5 returns of stock 0:", stocksReturns[0][-5:])


def transpose(matrix):
     nparray = np.array(matrix)
     transpose = np.transpose(nparray)
     result = transpose.tolist()
     return result


 # test function
assert (transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[1, 4, 7], [2, 5, 8],
                                                          [3, 6, 9]]), "Function transpose runs incorrectly"
import math


def featurize(factorReturns):
     squaredReturns = [i ** 2 if i >= 0 else -i ** 2 for i in factorReturns]
     squareRootedReturns = [math.sqrt(i) if i >= 0 else -math.sqrt(-i) for i in factorReturns]
     # concat new features
     return squaredReturns + squareRootedReturns + factorReturns


 # test our function

assert (featurize([4, -9, 25]) == [16, -81, 625, 2, -3, 5, 4, -9, 25]), "Function runs incorrectly"

def estimateParams(y, x):
    return sm.OLS(y, x).fit().params

# transpose factorsReturns
factorMat = transpose(factorsReturns)

# featurize each row of factorMat
factorFeatures = list(map(featurize, factorMat))

# OLS require parameter is a numpy array
factor_columns = np.array(factorFeatures)

#add a constant - the intercept term for each instrument i.
factor_columns = sm.add_constant(factor_columns, prepend=True)

# estimate weights
weights = [estimateParams(stockReturns,factor_columns) for stockReturns in stocksReturns]

print("weights:", weights)

print('\n','weights size:')
np.array(weights).shape

np.array(factorsReturns).shape

from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.nonparametric.kde import KDEUnivariate
import matplotlib.pyplot as plt
import scipy


def plotDistribution(samples):
     vmin = min(samples)
     vmax = max(samples)
     stddev = np.std(samples)

     domain = np.arange(vmin, vmax, (vmax - vmin) / 100)

     # a simple heuristic to select bandwidth
     bandwidth = 1.06 * stddev * pow(len(samples), -.2)

     # estimate density
     kde = KDEUnivariate(samples)
     kde.fit(bw=bandwidth)
     density = kde.evaluate(domain)

     # plot
     plt.plot(domain, density)
     plt.show()


plotDistribution(factorsReturns[0])
plotDistribution(factorsReturns[1])
plotDistribution(factorsReturns[2])
plotDistribution(factorsReturns[3])

correlation = np.corrcoef(factorsReturns)

print('Correlation of the 4 factors:','\n', (correlation))

##Trying to compare the 2 factors

plt.figure(figsize=(30,10))
f = [x*2.5 for x in factorsReturns[2]]    #The correlation c23 let us guess that GSPC and IXIC are linearly linked. We chose 2.5 as the estimation best suits the curve of IXIC.
plt.plot(factorsReturns[2],  'green')
plt.plot(factorsReturns[3], 'steelblue')
plt.plot(f, 'indianred')
plt.title("Estimation of IXIC by GSPC", fontweight='bold')
plt.legend(['real GSCP', "real IXIC", "Estimation of GSPC",])
plt.xlabel("2 weeks window")
plt.ylabel("Variations over a window")


# ax = plt.gca()
# ax.set_facecolor('xkcd:white')

plt.show()

#Representation of the data

plt.figure(figsize=(30,10))

plt.plot([x*10 for x in factorsReturns[0]],  'green') #Here we try to align the 3 remaining factors to see if they overloap.
plt.plot([x*400 for x in factorsReturns[1]], 'steelblue')
plt.plot([x for x in factorsReturns[2]], 'indianred')
plt.title("Representative factors", fontweight='bold')
plt.legend(["Crude Oil", "Treasury Bonds", "GSCP"])
plt.xlabel("2 weeks window")
plt.ylabel("Variations over a window")


# ax = plt.gca()
# ax.set_facecolor('xkcd:white')

plt.show()

factorCov = np.cov(factorsReturns)
factorMeans = [sum(factorsReturns[i])/len(factorsReturns[i]) for i in range(4)]
sample = np.random.multivariate_normal(factorMeans,factorCov)
print('factorCov:', '\n', factorCov)
print('factorMeans:', '\n', factorMeans)
print('sample:', '\n', sample)

 # We redefine the function given previously to plot two model curve.

def plotDistribution2(samples, samples2, i):
     vmin = min(samples)
     vmax = max(samples)
     stddev = np.std(samples)

     domain = np.arange(vmin, vmax, (vmax - vmin) / 100)

     # a simple heuristic to select bandwidth
     bandwidth = 1.06 * stddev * pow(len(samples), -.2)

     # estimate density
     kde = KDEUnivariate(samples)
     kde.fit(bw=bandwidth)
     density = kde.evaluate(domain)

     vmin2 = min(samples2)
     vmax2 = max(samples2)
     stddev2 = np.std(samples2)

     domain2 = np.arange(vmin2, vmax2, (vmax2 - vmin2) / 100)

     # a simple heuristic to select bandwidth
     bandwidth2 = 1.06 * stddev2 * pow(len(samples2), -.2)

     # estimate density
     kde2 = KDEUnivariate(samples2)
     kde2.fit(bw=bandwidth2)
     density2 = kde2.evaluate(domain2)

     # plot
     plt.plot(domain, density)
     plt.plot(domain2, density2)

     factors = ["Crude Oil", "Treasury Bonds", "GSPC", "IXIC"]
     plt.title(factors[i], fontweight='bold')
     plt.legend(["Estimation", "Real data"])
     plt.xlabel("2 weeks window")
     plt.ylabel("Variations over a window")
     plt.show()


 # Comparision of the reality and the normal distribution


for i in range(4):
     normalEstimates = [np.random.multivariate_normal(factorMeans, factorCov)[i] for k in range(len(factorsReturns[i]))]
     plotDistribution2(normalEstimates, factorsReturns[i], i)

def fivePercentVaR(trials):
    numTrials = trials.count()
    topLosses = trials.takeOrdered(max(round(numTrials/20.0), 1))
    return topLosses[-1]

# an extension of VaR
def fivePercentCVaR(trials):
    numTrials = trials.count()
    topLosses = trials.takeOrdered(max(round(numTrials/20.0), 1))
    return sum(topLosses)/len(topLosses)

def bootstrappedConfidenceInterval(
      trials, computeStatisticFunction,
      numResamples, pValue):
    stats = []
    for i in range(0, numResamples):
        resample = trials.sample(True, 1.0)
        stats.append(computeStatisticFunction(resample))
    sorted(stats)
    lowerIndex = int(numResamples * pValue / 2 - 1)
    upperIndex = int(np.ceil(numResamples * (1 - pValue / 2)))
    return (stats[lowerIndex], stats[upperIndex])


#RUN SILMULATION
def simulateTrialReturns(numTrials, factorMeans, factorCov, weights):
    trialReturns = []
    for i in range(0, numTrials):
        # generate sample of factors' returns
        trialFactorReturns =  np.random.multivariate_normal(factorMeans, factorCov)

        # featurize the factors' returns
        trialFeatures = featurize(trialFactorReturns.tolist())

        # insert weight for intercept term
        trialFeatures.insert(0,1)

        trialTotalReturn = 0

        # calculate the return of each instrument
        # then calulate the total of return for this trial features
        trialTotalReturn = sum(np.dot(weights,trialFeatures))


        trialReturns.append(trialTotalReturn)
    return trialReturns



parallelism = 4
numTrials = 10000
trial_indexes = list(range(0, parallelism))
seedRDD = sc.parallelize(trial_indexes, parallelism)
bFactorWeights = sc.broadcast(weights)

trials = seedRDD.flatMap(lambda idx: \
                simulateTrialReturns(
                    max(int(numTrials/parallelism), 1),
                    factorMeans, factorCov,
                    bFactorWeights.value
                ))
trials.cache()


valueAtRisk = fivePercentVaR(trials)
conditionalValueAtRisk = fivePercentCVaR(trials)

print ("Value at Risk(VaR) 5%:", valueAtRisk)
print ("Conditional Value at Risk(CVaR) 5%:", conditionalValueAtRisk)

print('shape of the random trial')
plt.hist(trials.collect())

from scipy import stats
import math


def countFailures(stocksReturns, valueAtRisk):
     failures = 0
     # iterate over time intervals
     for i in range(0, len(stocksReturns[0])):
         # calculate the losses in each time interval
         loss = -sum([stockReturn[i] for stockReturn in stocksReturns])

         # if the loss exceeds VaR
         if loss > -valueAtRisk:
             failures += 1
     return failures

print('We find a number of failures of:' , countFailures(stocksReturns, valueAtRisk))


def kupiecTestStatistic(total, failures, confidenceLevel):
     failureRatio = (failures / total)
     logNumer = (total - failures) * np.log(1 - confidenceLevel) + failures * np.log(confidenceLevel)
     logDenom = (total - failures) * np.log(1 - failureRatio) + failures * np.log(failureRatio)

     return -2 * (logNumer - logDenom)


 # test the function
assert (round(kupiecTestStatistic(250, 36, 0.1), 2) == 4.80), "function kupiecTestStatistic runs incorrectly"

def kupiecTestPValue(stocksReturns, valueAtRisk, confidenceLevel):
    failures = countFailures(stocksReturns, valueAtRisk)
    N = len(stocksReturns)
    print("num failures:", failures)
    total = len(stocksReturns[0])
    print("failure % : " , failures * 100/total)
    testStatistic = kupiecTestStatistic(total, failures, confidenceLevel)
    #return 1 - stats.chi2.cdf(testStatistic, 1.0)
    return stats.chi2.sf(testStatistic, 1.0)

varConfidenceInterval = bootstrappedConfidenceInterval(trials, fivePercentVaR, 100, 0.05)
cvarConfidenceInterval = bootstrappedConfidenceInterval(trials, fivePercentCVaR, 100, .05)
print("VaR confidence interval: " , varConfidenceInterval)
print("CVaR confidence interval: " , cvarConfidenceInterval)
print("Kupiec test p-value: " , kupiecTestPValue(stocksReturns, valueAtRisk, 0.05))

number_of_years = 5


def TotalInvest(start, end, number_of_years, factorMeans, factorCov):
     # select path of all stock data files in "stock_folder"
     files = [join(stock_folder, f) for f in listdir(stock_folder) if isfile(join(stock_folder, f))]

     # assume that we invest only the first 35 stocks (for faster computation)
     files = files[:150]

     # read each line in each file, convert it into the format: (date, value)
     rawStocks = [process_stock_file(f) for f in files]

     rawStocks = list(
         filter(lambda instrument: (((instrument[-1][0] - instrument[0][0]).days) / 260) >= number_of_years, rawStocks))

     # trim into a specific time region
     # and fill up the missing values
     stocks = list(map(lambda stock: \
                           fillInHistory(
                               trimToRegion(stock, start, end),
                               start, end),
                       rawStocks))
     print(len(stocks), 'kept stocks after time trim')

     stocksReturns = list(map(twoWeekReturns, stocks))
     print("5 first returns of stock 0 : ", stocksReturns[0][:5])

     weights = [estimateParams(stockReturns, factor_columns) for stockReturns in stocksReturns]
     print("number of weights : ", len(weights))

     parallelism = 4
     numTrials = 10000
     trial_indexes = list(range(0, parallelism))
     seedRDD = sc.parallelize(trial_indexes, parallelism)
     bFactorWeights = sc.broadcast(weights)

     trials = seedRDD.flatMap(lambda idx: \
                                  simulateTrialReturns(
                                      max(int(numTrials / parallelism), 1),
                                      factorMeans, factorCov,
                                      bFactorWeights.value
                                  ))
     trials.cache()

     valueAtRisk = fivePercentVaR(trials)
     conditionalValueAtRisk = fivePercentCVaR(trials)

     print("Value at Risk(VaR) 5%:", valueAtRisk)
     print("Conditional Value at Risk(CVaR) 5%:", conditionalValueAtRisk)
     print(countFailures(stocksReturns, valueAtRisk))
     varConfidenceInterval = bootstrappedConfidenceInterval(trials, fivePercentVaR, 100, 0.05)
     cvarConfidenceInterval = bootstrappedConfidenceInterval(trials, fivePercentCVaR, 100, .05)
     print("VaR confidence interval: ", varConfidenceInterval)
     print("CVaR confidence interval: ", cvarConfidenceInterval)
     print("Kupiec test p-value: ", kupiecTestPValue(stocksReturns, valueAtRisk, 0.05))

TotalInvest(start, end, number_of_years, factorMeans, factorCov)

#this code is the generous contribution of stackoverflow.

import matplotlib.pyplot as plt

import warnings
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (8.0, 6.0)
matplotlib.style.use('ggplot')

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Propbability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# Load data from statsmodels datasets
data = pd.Series(factorsReturns[0])

# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fir_paramms = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(' All Fitted Distributions')


# Make PDF
pdf = make_pdf(best_dist, best_fir_paramms)

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fir_paramms)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title('best fit distribution for crude oil \n' + dist_str)

best_crude_oil = best_dist
params_crude_oil = best_crude_oil.fit(factorsReturns[0])
best_estimate_crude_oil = best_crude_oil.rvs(*params_crude_oil, size=1295)

plotDistribution2(best_estimate_crude_oil,factorsReturns[0],0)

# Load data from statsmodels datasets
data = pd.Series(factorsReturns[1])

# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fir_paramms = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(' All Fitted Distributions')


# Make PDF
pdf = make_pdf(best_dist, best_fir_paramms)

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fir_paramms)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title('best fit distribution for treasury bonds \n' + dist_str)

best_bonds = best_dist
params_bonds = best_bonds.fit(factorsReturns[1])
best_estimate_bonds = best_bonds.rvs(*params_bonds, size=1295)

plotDistribution2(best_estimate_bonds,factorsReturns[1],1)

# Load data from statsmodels datasets

data = pd.Series(factorsReturns[3])

# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fir_paramms = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(' All Fitted Distributions')


# Make PDF
pdf = make_pdf(best_dist, best_fir_paramms)

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fir_paramms)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title('best fit distribution for IXIC \n' + dist_str)

best_ixic = best_dist
params_ixic = best_ixic.fit(factorsReturns[3])
best_estimate_ixic = best_ixic.rvs(*params_ixic, size=1295)

plotDistribution2(best_estimate_ixic,factorsReturns[3],3)

# Load data from statsmodels datasets

data = pd.Series(factorsReturns[2])

# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fir_paramms = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(' All Fitted Distributions')


# Make PDF
pdf = make_pdf(best_dist, best_fir_paramms)

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fir_paramms)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title('best fit distribution for GSPC \n' + dist_str)

best_gspc = best_dist
params_gspc = best_gspc.fit(factorsReturns[2])
best_estimate_gspc = best_gspc.rvs(*params_gspc, size=1295)

plotDistribution2(best_estimate_gspc,factorsReturns[2],2)
def simulateTrialReturns2(numTrials, factorMeans, factorCov, weights):
    trialReturns = []
    trialFactorReturns = np.array([0,0,0,0])
    for i in range(0, numTrials):
        # generate sample of factors' returns
        #trialFactorReturns =  np.random.multivariate_normal(factorMeans, factorCov)
        #here we sample each factor from their corresponding "best dist"
        trialFactorReturns[0] =best_crude_oil.rvs(*params_crude_oil)
        trialFactorReturns[1] =  best_bonds.rvs(*params_bonds)
        trialFactorReturns[2] =best_gspc.rvs(*params_gspc)
        trialFactorReturns[3] =  best_ixic.rvs(*params_ixic)

        # featurize the factors' returns
        trialFeatures = featurize(trialFactorReturns.tolist())

        # insert weight for intercept term
        trialFeatures.insert(0,1)

        trialTotalReturn = 0

        # calculate the return of each instrument
        # then calulate the total of return for this trial features
        trialTotalReturn = sum(np.dot(weights,trialFeatures))


        trialReturns.append(trialTotalReturn)
    return trialReturns


number_of_years = 5


def TotalInvest2(start, end, number_of_years, factorMeans, factorCov):
     # select path of all stock data files in "stock_folder"
     files = [join(stock_folder, f) for f in listdir(stock_folder) if isfile(join(stock_folder, f))]

     # assume that we invest only the first 35 stocks (for faster computation)
     files = files[:150]

     # read each line in each file, convert it into the format: (date, value)
     rawStocks = [process_stock_file(f) for f in files]

     rawStocks = list(
         filter(lambda instrument: (((instrument[-1][0] - instrument[0][0]).days) / 260) >= number_of_years, rawStocks))

     # trim into a specific time region
     # and fill up the missing values
     stocks = list(map(lambda stock: \
                           fillInHistory(
                               trimToRegion(stock, start, end),
                               start, end),
                       rawStocks))
     print(len(stocks), 'kept stocks after time trim')

     stocksReturns = list(map(twoWeekReturns, stocks))
     print("5 first returns of stock 0 : ", stocksReturns[0][:5])

     weights = [estimateParams(stockReturns, factor_columns) for stockReturns in stocksReturns]
     print("number of weights : ", len(weights))

     parallelism = 4
     numTrials = 10000
     trial_indexes = list(range(0, parallelism))
     seedRDD = sc.parallelize(trial_indexes, parallelism)
     bFactorWeights = sc.broadcast(weights)

     trials = seedRDD.flatMap(lambda idx: \
                                  simulateTrialReturns2(
                                      max(int(numTrials / parallelism), 1),
                                      factorMeans, factorCov,
                                      bFactorWeights.value
                                  ))
     trials.cache()

     valueAtRisk = fivePercentVaR(trials)
     conditionalValueAtRisk = fivePercentCVaR(trials)

     print("Value at Risk(VaR) 5%:", valueAtRisk)
     print("Conditional Value at Risk(CVaR) 5%:", conditionalValueAtRisk)
     print(countFailures(stocksReturns, valueAtRisk))
     varConfidenceInterval = bootstrappedConfidenceInterval(trials, fivePercentVaR, 100, 0.05)
     cvarConfidenceInterval = bootstrappedConfidenceInterval(trials, fivePercentCVaR, 100, .05)
     print("VaR confidence interval: ", varConfidenceInterval)
     print("CVaR confidence interval: ", cvarConfidenceInterval)
     print("Kupiec test p-value: ", kupiecTestPValue(stocksReturns, valueAtRisk, 0.05))

TotalInvest2(start, end, number_of_years, factorMeans, factorCov)

sc.stop()


