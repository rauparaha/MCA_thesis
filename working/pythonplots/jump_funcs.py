## ======================================== ##
## JUMP_FUNC.PY                             ##
##                                          ##
## Function to generate variable paths and  ##
## plots from maple results contained in    ##
## resdata.csv and generated from the Maple ##
## code in ../maplesolver/                  ##
##                                          ##
## James Zuccollo                           ##
## 24/03/2011                               ##
## ======================================== ##


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

## ==============================
## DEFINE THE NECESSARY FUNCTIONS
## ==============================


#Define the regulator's strategy function
def RegStratFunc(p0, tau0):
    tau1 = ar + gr1 * p0 + gr2 * tau0
    return tau1


#Define the monopolist's strategy function
def MonStratFunc(p0, tau0):
    p1 = am + gm1 * p0 + gm2 * tau0
    return p1


#Define the demand function
def DemandFunc(p0, tau0):
    x1 = p0 - (beta + 1) * MonStratFunc(p0, tau0) + beta *\
         MonStratFunc(MonStratFunc(p0, tau0), RegStratFunc(p0, tau0))
    return x1


#Define the consumer surplus function
def CS(p0, tau0):
    ## CS1 = 0.5 * (1 -\
    ##              (beta * MonStratFunc(MonStratFunc(p0, tau0),\
    ##                                   RegStratFunc(p0, tau0))) ** 2\
    ##              + MonStratFunc(p0, tau0) ** 2) * (beta + 0.5)\
    ##              - p0 * MonStratFunc(p0, tau0)
    CS1 = 0.5 * MonStratFunc(p0, tau0)**2 - \
          0.5 * beta**2 * MonStratFunc(MonStratFunc(p0, tau0), RegStratFunc(p0, tau0))**2\
          - MonStratFunc(p0, tau0) * p0 + beta * MonStratFunc(p0, tau0)**2 + 0.5
    return CS1


#Define the pollution function
def Pollution(p0, tau0):
    Psi1 = kappa * DemandFunc(p0, tau0)
    return Psi1


#Define production costs
def ProdCost(p0, tau0):
    PrdCst1 = 0.5 * rho * DemandFunc(p0, tau0) ** 2
    return PrdCst1


#Define the monopolist's profit function
def Profit(p0, tau0):
    Prft1 = MonStratFunc(p0, tau0) * DemandFunc(p0, tau0)\
           - ProdCost(p0, tau0) - RegStratFunc(p0, tau0)\
           * Pollution(p0, tau0)
    return Prft1


#Define the welfare function/regulator's objective
def Welfare(p0, tau0):
    Welf1 = Profit(p0, tau0) + CS(p0, tau0) - Pollution(p0, tau0)\
           - (1 - alpha) * RegStratFunc(p0, tau0) * Pollution(p0, tau0)\
           - theta * (RegStratFunc(p0, tau0) - tau0) ** 2
    return Welf1

# Set the welfare parameter values using global variables
# and set to base levels

global alpha, theta, rho, kappa, beta,\
       delta, ar, grOne, grTwo, am, gmOne, gmTwo

# =======================
#   Read data from CSV
# ======================

# Read all data into a numpy array
resdata = np.genfromtxt('resdata.csv', dtype=float, delimiter=',')

# Generate price and tax rate series for arbitrary starting point

# Pick an arbitrary steady-state starting point
pbar = 1.4
taubar = 0.0

# number of periods to plot
periods = 15

# Initialise numpy arrays to take the paths
PricePath = np.zeros((len(resdata), periods))
TaxPath = np.zeros((len(resdata), periods))
PricePath.transpose()[0] = pbar
TaxPath.transpose()[0] = taubar

DemandPath = np.zeros((len(resdata), periods))
ProfitPath = np.zeros((len(resdata), periods))
CSPath = np.zeros((len(resdata), periods))
PollutionPath = np.zeros((len(resdata), periods))
WelfarePath = np.zeros((len(resdata), periods))

# Fill the array with price and taxation data
for sim  in np.arange(24):
    alpha = resdata[sim][0]
    theta = resdata[sim][1]
    rho = resdata[sim][2]
    kappa = resdata[sim][3]
    beta = resdata[sim][4]
    delta = resdata[sim][5]
    ar = resdata[sim][6]
    gr1 = resdata[sim][7]
    gr2 = resdata[sim][8]
    am = resdata[sim][9]
    gm1 = resdata[sim][10]
    gm2 = resdata[sim][11]
    for period in np.arange(1, periods):
        PricePath[sim][period] = MonStratFunc(PricePath[sim][period-1],\
                                              TaxPath[sim][period-1])
        TaxPath[sim][period] = RegStratFunc(PricePath[sim][period-1],\
                                            TaxPath[sim][period-1])
        DemandPath[sim][period] = DemandFunc(PricePath[sim][period-1],\
                                             TaxPath[sim][period-1])
        ProfitPath[sim][period] = Profit(PricePath[sim][period-1],\
                                         TaxPath[sim][period-1])
        CSPath[sim][period] = CS(PricePath[sim][period-1],\
                                 TaxPath[sim][period-1])
        PollutionPath[sim][period] = Pollution(PricePath[sim][period-1],\
                                               TaxPath[sim][period-1])
        WelfarePath[sim][period] = Welfare(PricePath[sim][period-1],\
                                           TaxPath[sim][period-1])



# ============================
#      Plot the series
# ============================

# Create a colour and marker map for the scatterplot
params = ['alpha', 'theta', 'rho', 'kappa', 'beta', 'delta']
cols = [ '#3D2B1F','#002FA7', '#B7410E', '#08457E', '#592720',\
         '#FFC0CB', '#008080']
marks = ['s', 'o', '^', '>', 'v', '<', 'd', 'p', 'h', '8', '+', 'x']
colmap = dict(zip(params, cols))
markmap = dict(zip(params, marks))

# Set backend params [play with this when generating figs for typesetting]
##fig_width_pt = 360.0  # Get this from LaTeX using \showthe\columnwidth
##inches_per_pt = 1.0 / 72.27               # Convert pt to inch
##golden_mean = (np.sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
##fig_width = fig_width_pt * inches_per_pt  # width in inches
##fig_height = fig_width * golden_mean      # height in inches
##fig_size =  [fig_width, fig_height]
##param = {'backend' : 'ps',
##           'axes.labelsize' : 7,
##           'text.fontsize' : 9,
##           'legend.fontsize' : 7,
##           'xtick.labelsize' : 6,
##           'ytick.labelsize' : 6,
##           'text.usetex' : True}
##           'figure.figsize' : fig_size}
##mpl.rcParams.update(param)
mpl.rcdefaults()
mpl.rcParamsDefault.update()
#mpl.rc('text', usetex = True)

# ============
# SCATTERPLOTS
# ============

# Scatterplots of steady state price and tax rate across various
# parameter values. Not welfare etc because demand is zero in the
# steady state.


# Create a figure.


## ==== BETA/DELTA ====

BetaDeltaFig = plt.figure()
BetaDeltaPlot = BetaDeltaFig.add_subplot(111)
#BetaDeltaPlot.set_title("Steady states across parameter values")
BetaDeltaPlot.set_ylabel('Pollution tax')
BetaDeltaPlot.set_xlabel("Monopolist's price")
BetaDeltaPlot.grid(True, linestyle='-', color='0.2')
BetaDeltaPlot.scatter(resdata.transpose()[12][5:9],\
                      resdata.transpose()[13][5:9], s = 50,\
                      marker = marks[1], color = cols[1], label = 'Beta')
BetaDeltaPlot.scatter(resdata.transpose()[12][9:14], \
                      resdata.transpose()[13][9:14], marker = marks[2], s = 50,\
                      color = cols[2], label ='Delta')
BetaDeltaPlot.legend(loc=2)
BetaDeltaFig.savefig('SS_beta_delta.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

## ==== KAPPA/ALPHA ====

KappaAlphaFig = plt.figure()
KappaAlphaPlot = KappaAlphaFig.add_subplot(111)
#KappaAlphaPlot.set_title("Steady states across parameter values")
KappaAlphaPlot.set_ylabel('Pollution tax')
KappaAlphaPlot.set_xlabel("Monopolist's price")
KappaAlphaPlot.grid(True, linestyle='-', color='0.2')
KappaAlphaPlot.scatter(resdata.transpose()[12][19:24],\
                      resdata.transpose()[13][19:24], s = 50,\
                      marker = marks[3], color = cols[3], label = 'Kappa')
KappaAlphaPlot.scatter(resdata.transpose()[12][0:3],\
                      resdata.transpose()[13][0:3], marker = marks[4], s = 50,\
                      color = cols[4], label ='Alpha')
KappaAlphaPlot.legend(loc=2)
KappaAlphaFig.savefig('SS_kappa_alpha.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

## ==== THETA/RHO ====

ThetaRhoFig = plt.figure()
ThetaRhoPlot = ThetaRhoFig.add_subplot(111)
#ThetaRhoPlot.set_title("Steady states across parameter values")
ThetaRhoPlot.set_ylabel('Pollution tax')
ThetaRhoPlot.set_xlabel("Monopolist's price")
ThetaRhoPlot.grid(True, linestyle='-', color='0.2')
ThetaRhoPlot.scatter(resdata.transpose()[12][2:5],\
                      resdata.transpose()[13][2:5],s = 100,\
                      marker = marks[5], color = cols[5], label = 'Theta')
ThetaRhoPlot.scatter(resdata.transpose()[12][14:19],\
                      resdata.transpose()[13][14:19], marker = marks[6], s = 100,\
                      color = cols[6], label ='Rho')
ThetaRhoPlot.legend(loc=2)
ThetaRhoFig.savefig('SS_theta_rho.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Save the generated Scatter Plot to a PNG file.



# =================
# TIME SERIES PLOTS
# =================

# Create an index for periods
TimeIndex = np.arange(periods-2)

# Plot the variables in subplots

# ====== BASE CASE ======

# Create a figure.
conv_BC_priceFig = plt.figure()
conv_BC_pricePlot = conv_BC_priceFig.add_subplot(111)
conv_BC_pricePlot.set_ylabel('Price')
conv_BC_pricePlot.plot(TimeIndex, PricePath[2][2:])
conv_BC_priceFig.savefig('conv_BC_price.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_BC_taxFig = plt.figure()
conv_BC_taxPlot = conv_BC_taxFig.add_subplot(111)
conv_BC_taxPlot.set_ylabel('Tax')
conv_BC_taxPlot.plot(TimeIndex, TaxPath[2][2:])
conv_BC_taxFig.savefig('conv_BC_tax.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_BC_demandFig = plt.figure()
conv_BC_demandPlot = conv_BC_demandFig.add_subplot(111)
conv_BC_demandPlot.set_ylabel('Demand')
conv_BC_demandPlot.plot(TimeIndex, DemandPath[2][2:])
conv_BC_demandFig.savefig('conv_BC_demand.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_BC_profitFig = plt.figure()
conv_BC_profitPlot = conv_BC_profitFig.add_subplot(111)
conv_BC_profitPlot.set_ylabel('Profit')
conv_BC_profitPlot.plot(TimeIndex, ProfitPath[2][2:])
conv_BC_profitFig.savefig('conv_BC_profit.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_BC_PollutionFig = plt.figure()
conv_BC_PollutionPlot = conv_BC_PollutionFig.add_subplot(111)
conv_BC_PollutionPlot.set_ylabel('Pollution')
conv_BC_PollutionPlot.plot(TimeIndex, PollutionPath[2][2:])
conv_BC_PollutionFig.savefig('conv_BC_pollution.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_BC_welfareFig = plt.figure()
conv_BC_welfarePlot = conv_BC_welfareFig.add_subplot(111)
conv_BC_welfarePlot.set_ylabel('Welfare')
conv_BC_welfarePlot.plot(TimeIndex, WelfarePath[2][2:])
conv_BC_welfareFig.savefig('conv_BC_welfare.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# ====== BETA ========

# Create a figure.
conv_Beta_priceFig = plt.figure()
conv_Beta_pricePlot = conv_Beta_priceFig.add_subplot(111)
conv_Beta_pricePlot.set_ylabel('Price')
for i in range(4):
    conv_Beta_pricePlot.plot(TimeIndex, PricePath[i+5][2:])
conv_Beta_priceFig.savefig('conv_Beta_price.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_Beta_taxFig = plt.figure()
conv_Beta_taxPlot = conv_Beta_taxFig.add_subplot(111)
conv_Beta_taxPlot.set_ylabel('Tax')
for i in range(4):
    conv_Beta_taxPlot.plot(TimeIndex, TaxPath[i+5][2:])
conv_Beta_taxFig.savefig('conv_Beta_tax.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_Beta_demandFig = plt.figure()
conv_Beta_demandPlot = conv_Beta_demandFig.add_subplot(111)
conv_Beta_demandPlot.set_ylabel('Demand')
for i in range(4):
    conv_Beta_demandPlot.plot(TimeIndex, DemandPath[i+5][2:])
conv_Beta_demandFig.savefig('conv_Beta_demand.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_Beta_profitFig = plt.figure()
conv_Beta_profitPlot = conv_Beta_profitFig.add_subplot(111)
conv_Beta_profitPlot.set_ylabel('Profit')
for i in range(4):
    conv_Beta_profitPlot.plot(TimeIndex, ProfitPath[i+5][2:])
conv_Beta_profitFig.savefig('conv_Beta_profit.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_Beta_pollutionFig = plt.figure()
conv_Beta_pollutionPlot = conv_Beta_pollutionFig.add_subplot(111)
conv_Beta_pollutionPlot.set_ylabel('Pollution')
for i in range(4):
    conv_Beta_pollutionPlot.plot(TimeIndex, PollutionPath[i+5][2:])
conv_Beta_pollutionFig.savefig('conv_Beta_pollution.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_Beta_welfareFig = plt.figure()
conv_Beta_welfarePlot = conv_Beta_welfareFig.add_subplot(111)
conv_Beta_welfarePlot.set_ylabel('Welfare')
for i in range(4):
    conv_Beta_welfarePlot.plot(TimeIndex, WelfarePath[i+5][2:])
conv_Beta_welfareFig.savefig('conv_Beta_welfare.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')



# ====== DELTA ========

# Create a figure.
conv_Delta_priceFig = plt.figure()
conv_Delta_pricePlot = conv_Delta_priceFig.add_subplot(111)
conv_Delta_pricePlot.set_ylabel('Price')
for i in range(5):
    conv_Delta_pricePlot.plot(TimeIndex, PricePath[i+9][2:])
conv_Delta_priceFig.savefig('conv_Delta_price.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_Delta_taxFig = plt.figure()
conv_Delta_taxPlot = conv_Delta_taxFig.add_subplot(111)
conv_Delta_taxPlot.set_ylabel('Tax')
for i in range(5):
    conv_Delta_taxPlot.plot(TimeIndex, TaxPath[i+9][2:])
conv_Delta_taxFig.savefig('conv_Delta_tax.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_Delta_demandFig = plt.figure()
conv_Delta_demandPlot = conv_Delta_demandFig.add_subplot(111)
conv_Delta_demandPlot.set_ylabel('Demand')
for i in range(5):
    conv_Delta_demandPlot.plot(TimeIndex, DemandPath[i+9][2:])
conv_Delta_demandFig.savefig('conv_Delta_demand.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_Delta_profitFig = plt.figure()
conv_Delta_profitPlot = conv_Delta_profitFig.add_subplot(111)
conv_Delta_profitPlot.set_ylabel('Profit')
for i in range(5):
    conv_Delta_profitPlot.plot(TimeIndex, ProfitPath[i+9][2:])
conv_Delta_profitFig.savefig('conv_Delta_profit.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_Delta_pollutionFig = plt.figure()
conv_Delta_pollutionPlot = conv_Delta_pollutionFig.add_subplot(111)
conv_Delta_pollutionPlot.set_ylabel('Pollution')
for i in range(5):
    conv_Delta_pollutionPlot.plot(TimeIndex, PollutionPath[i+9][2:])
conv_Delta_pollutionFig.savefig('conv_Delta_pollution.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')

# Create a figure.
conv_Delta_welfareFig = plt.figure()
conv_Delta_welfarePlot = conv_Delta_welfareFig.add_subplot(111)
conv_Delta_welfarePlot.set_ylabel('Welfare')
for i in range(5):
    conv_Delta_welfarePlot.plot(TimeIndex, WelfarePath[i+9][2:])
conv_Delta_welfareFig.savefig('conv_Delta_welfare.pdf', dpi = 500,\
                   bbox_inches = 'tight', papertype = 'a4')
