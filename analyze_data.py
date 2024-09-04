import numpy as np
import matplotlib.pyplot as plt
import ctem
import csv
import pandas as pd
import os
import scipy.stats as stats
import matplotlib
from scipy.interpolate import interp1d
from matplotlib.patches import Rectangle
from scipy.optimize import minimize, curve_fit, fsolve

###### This section contains routines to find bin boundary ages for both the NPF and RPF (see Supporting Info)

def npf(t):
    return 5.44e-14*(np.exp(6.93*t)-1)+8.17e-4*t

def rpf(t):
    return 7.26e-41*(np.exp(22.6*t)-1)+(9.49e-4*t)+(1.88e-4*t**2)

def func(t,n1):
    return rpf(t) - n1

def test(age): #use NPF to get a N1. What RPF age does that N1 correspond to?
    n1 = npf(age)
    return fsolve(func,3.9,args=(n1,))

def reverse(t,n1):
    return npf(t) - n1

def convert(age): #RPF to NPF
    n1 = rpf(age)
    return fsolve(reverse,3.9,args=(n1,))

#### End section

def calculate_required_simulations(data,confidence_level=0.95,margin_of_error=0.05):
    '''Calculates number of simulations needed with given significance parameters'''

    required_simulations = {}

    for item in data.columns:
        # Calculate the sample mean (mu) and standard deviation (sigma)
        mu = data[item].mean()
        sigma = data[item].std(ddof=1)

        # Calculate the z-score for the given confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # Calculate the required sample size
        required_n = ((z_score * sigma) / margin_of_error) ** 2

        # Store the result, rounding up to ensure a whole number of simulations
        required_simulations[item] = {
            'mean': mu,
            'standard_deviation': sigma,
            'required_simulations': np.ceil(required_n)
        }
        
        key, largest_value = max(required_simulations.items(), key=lambda item: item[1]['required_simulations'])

    return key, largest_value

def calculate_combined_relative_probabilities(df1, df2):

    combined_df = df1 + df2

    # Calculate the combined relative probabilities
    combined_relative_probs = combined_df.sum() / combined_df.sum().sum()

    # Calculate the contribution of small and large craters to the combined relative probabilities
    small_crater_contribution = (df1.sum() / combined_df.sum()) * combined_relative_probs
    large_crater_contribution = (df2.sum() / combined_df.sum()) * combined_relative_probs

    return combined_relative_probs, small_crater_contribution, large_crater_contribution

def convert_npf_to_rpf(alrp):
    '''converts NPF ages of local melt to RPF ages based on the bin structure of the project.
        This assumes a calculated percentage of each NPF-derived age bin will be the RPF-derived age bin.'''

    rlrp = alrp.copy()
    rlrp['4.09454166'] = 0.0
    rlrp['4.16637572'] = 0.0
    rlrp['4.23820979'] = 0.0
    rlrp['4.31004385'] = 0.0
    rlrp['4.02270759'] = alrp['4.31004385'] + alrp['4.23820979'] + alrp['4.16637572'] + alrp['4.09454166'] + alrp['4.02270759'] + (0.792436067*alrp['3.95087353'])
    rlrp['3.95087353'] = (0.541002694*alrp['3.8072054']) + alrp['3.87903947'] + (0.20756393*alrp['3.95087353'])
    rlrp['3.87903947'] = (0.297849238*alrp['3.66353727']) + alrp['3.73537134'] + (0.45899731*alrp['3.8072054'])
    rlrp['3.8072054'] = (0.010298736*alrp['3.59170321']) + (0.70215076*alrp['3.66353727'])
    rlrp['3.73537134'] = (0.25783632*alrp['3.59170321'])
    rlrp['3.66353727'] = (0.14255494*alrp['3.59170321'])
    rlrp['3.59170321'] = (0.12105302*alrp['3.59170321'])
    rlrp['3.51986915'] = (0.12065516*alrp['3.59170321'])
    rlrp['3.44803508'] = (0.12511279*alrp['3.59170321'])
    rlrp['3.37620102'] = (0.13106415*alrp['3.59170321'])
    rlrp['3.30436695'] = (0.046463196*alrp['3.51986915']) + (0.09142488*alrp['3.59170321'])
    rlrp['3.23253289'] = (0.14555324*alrp['3.51986915'])
    rlrp['3.16069882'] = (0.15417436*alrp['3.51986915'])
    rlrp['3.08886476'] = (0.15570956*alrp['3.51986915']) 

    return rlrp

def change_iridum_age(mts,la,qa,age='3.8072054'): #age must be a string of either: 3.87903947, '3.8072054', '3.73537134', '3.66353727', or '3.59170321'.
    '''Changes the age of the melt from the Iridum sub-basin for the exercises in the Discussion.'''

    #Example inputs:

    #a14melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    #a14la
    #a14qa.copy()


    qa['3.95087353'] += mts['Imbrium']
    qa['3.87903947'] -= mts['Imbrium']
    qa['3.95087353'] += mts['Orientale']
    qa['3.87903947'] -= mts['Orientale']
    qa['3.51986915'] -= mts['Archimedes']
    qa['3.66353727'] += mts['Archimedes']
    qa['3.44803508'] -= mts['Iridum']
    qa[age] += mts['Iridum']
    qa['3.59170321'] = qa['3.59170321'] + qa['3.51986915'] + qa['3.44803508'] + qa['3.37620102'] + qa['3.30436695'] + qa['3.23253289'] + qa['3.16069882'] + qa['3.08886476'] + qa['3.0170307']
    for time in ['3.0170307', '3.08886476', '3.16069882', '3.23253289', '3.30436695', '3.37620102', '3.44803508', '3.51986915']:
        qa[time] = 0.0
    crp, lrp, qrp = calculate_combined_relative_probabilities(la.drop('Unnamed: 0',axis=1),qa.drop('Unnamed: 0',axis=1))

    return crp, lrp, qrp

def generate_figure_1():
    '''Generates figure of NPF and RPF, along with datapoints'''

    t = np.linspace(0, 4.6, 100000)
    oldnpf = 5.44e-14*(np.exp(6.93*t)-1)+8.38e-4*t
    npf = 5.44e-14*(np.exp(6.93*t)-1)+8.17e-4*t
    rpf = 7.26e-41*(np.exp(22.6*t)-1)+(9.49e-4*t)+(1.888e-4*t**2)
    ce5_a = 2.03
    ce5_n1 = 0.0017

    #Datapoints

    # A11O, A11Y, A17, L16, A15, L24, A12, North Ray, Cone

    n1 = np.array([0.009, 0.0064, 100e-4, 0.0033, 0.0032, 0.003, 0.0036])#, 0.000044, 0.000021])
    n1_error = np.array([18e-4, 20e-4, 30e-4, 10e-4, 11e-4, 10e-4, 11e-4])#, 0.11e-4, 0.21e-4])
    ages = np.array([3.80, 3.58, 3.75, 3.41, 3.3, 3.22, 3.15])#, 0.053, 0.025])
    age_error = np.array([0.02, 0.01, 0.01, 0.04, 0.02, 0.02, 0.015])#, 0.008, 0.012])
    ages_sc = np.array([0.053, 0.025])
    ages_error_sc = np.array([0.008, 0.012])
    n1_sc = np.array([0.000044, 0.000021])
    n1_error_sc = np.array([0.11e-4, 0.21e-4])
    ce5_a = 2.03
    ce5_n1 = 0.0017
    ages_r = np.array([3.80, 3.75, 3.41, 3.3, 3.22, 3.15])
    n1_r = np.array([8140/1e6,5660/1e6,5280/1e6,5500/1e6,4660/1e6,5910/1e6])
    n1_r_e = np.array([800/1e6,820/1e6,640/1e6,1300/1e6,580/1e6,960/1e6])
    age_error_r = np.array([0.02, 0.01, 0.04, 0.02, 0.02, 0.015])

    #Nectaris, Serenitatis, Crisium, Imbrium

    ages_basins = [3.92, 3.89, 3.89, 3.85]
    ages_error_basins = [0.03, 0.02, 0.01, 0.02]
    n1_basins = [1200e-4,  570e-4, 0.065, 370e-4]
    n1_error_basins = [400e-4, 0, 0, 70e-4]

    ages_sc = np.array([0.85, 0.053, 0.025])
    ages_error_sc = np.array([0.20, 0.008, 0.012])
    n1_sc = np.array([0.0013, 0.000044, 0.000021]) #Copernicus, North Ray, Cone
    n1_error_sc = np.array([0.0003, 0.11e-4, 0.21e-4])

    f1 = plt.figure(1)
    a1 = f1.add_subplot(111)
    a1.plot(t, npf, color='r', label='Neukum Production Function (NPF)')
    a1.plot(t, rpf, color='b', label='Robbins Production Function (RPF)')
    a1.scatter(ages, n1, color='r', label='Neukum mare counts')
    a1.errorbar(ages, n1, xerr=age_error, yerr=n1_error, color='r', ls='None')
    a1.scatter(ages_r,n1_r,color='b',label='Robbins mare counts')
    a1.errorbar(ages_r, n1_r, xerr=age_error_r, yerr=n1_r_e, color='b', ls='None')
    a1.scatter(ce5_a, ce5_n1, color='k', label="Chang'e 5 (Li et al., 2021)")
    a1.scatter(ages_basins, n1_basins, color='r', marker='^', label='Basin constraining points')
    a1.scatter(ages_sc, n1_sc, color='purple', marker='s', label='Small crater constraining points')
    a1.errorbar(ages_sc, n1_sc, xerr=ages_error_sc, yerr=n1_error_sc, color='indigo', ls='None')
    a1.errorbar(ages_basins, n1_basins, xerr=ages_error_basins, yerr=n1_error_basins, color='r', ls='None')
    #a1.scatter(4.05, 370e-4, color='k')#, label='Basins')
    a1.set_yscale('log')
    a1.set_xlim(4.5,0)
    a1.set_ylim(1e-5,1)
    r = plt.Rectangle((3.92,1e-5),(4.5-3.92),(1-1e-5),alpha=0.5,fill=True,color='grey',label='Unconstrained')
    r2 = plt.Rectangle((3.80,1e-5),(3.92-3.80),(1-1e-5),alpha=0.5,fill=True,color='silver',label='Poorly constrained')
    a1.add_patch(r2)
    a1.add_patch(r)
    a1.legend()
    a1.set_xlabel('Age (Ga)')
    a1.set_ylabel('Crater Density [N(1)] ($km^{-2}$)')
    a1.grid()
    plt.rcParams['figure.figsize'] = [10, 10]
    matplotlib.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.savefig('figure_1.pdf')

    return




def generate_figure_2():
    '''Generates figure of histogram of Imbrian-aged samples from the compendium of Michael et al. (2018)'''

    f1 = plt.figure(1)
    a1 = f1.add_subplot(111)
    a1.bar(bin_centers, hist14, bottom=0, width=0.07183406, color='#fef0d9', edgecolor='k', label='Apollo 14')
    a1.bar(bin_centers, hist15, bottom=hist14, width=0.07183406, color='#fdcc8a', edgecolor='k', label='Apollo 15')
    a1.bar(bin_centers, hist16, bottom=hist14+hist15, width=0.07183406, color='#fc8d59', edgecolor='k', label='Apollo 16')
    a1.bar(bin_centers, hist17, bottom=hist14+hist15+hist16, width=0.07183406, color='#d7301f', edgecolor='k', label='Apollo 17')
    a1.set_xlim(3.95,3.0)
    a1.set_xlabel('Age (Ga)')
    a1.set_ylabel('Number of samples')
    a1.legend()
    plt.rcParams['figure.figsize'] = [7, 5]
    matplotlib.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.savefig('figure_2.pdf')

    return



def generate_figure_7():
    '''generates a plot of local melt for all Apollo sites under both the NPF and RPF'''

    r14lrp = convert_npf_to_rpf(a14lrp)
    r15lrp = convert_npf_to_rpf(a15lrp)
    r16lrp = convert_npf_to_rpf(a16lrp)
    r17lrp = convert_npf_to_rpf(a17lrp)

    #plot

    f1, a1 = plt.subplots(nrows=4,ncols=1,figsize=(7,25))
    a1[3].plot(newtimes,a17lrp/a17qrp[6],color='r',label='NPF')
    a1[3].plot(newtimes,r17lrp/a17qrp[6],color='b',label='RPF')
    a1[3].set_xlim(3.95,3.0)
    a1[3].set_ylim(0,0.6)
    a1[3].set_xlabel('Age (Ga)')
    a1[3].set_ylabel('Relative probability / RP(Imbrium)')
    a1[3].set_title('(d) Apollo 17')
    a1[3].legend()
    a1[2].plot(newtimes,a16lrp/a16qrp[6],color='r',label='NPF')
    a1[2].plot(newtimes,r16lrp/a16qrp[6],color='b',label='RPF')
    a1[2].set_xlim(3.95,3.0)
    a1[2].set_ylim(0,0.8)
    a1[2].set_ylabel('Relative probability / RP(Imbrium)')
    a1[2].set_title('(c) Apollo 16')
    a1[2].legend()
    a1[1].plot(newtimes,a15lrp/a15qrp[6],color='r',label='NPF')
    a1[1].plot(newtimes,r15lrp/a15qrp[6],color='b',label='RPF')
    a1[1].set_xlim(3.95,3.0)
    a1[1].set_ylim(0,0.06)
    a1[1].set_ylabel('Relative probability / RP(Imbrium)')
    a1[1].set_title('(b) Apollo 15')
    a1[1].legend()
    a1[0].plot(newtimes,a14lrp/a14qrp[6],color='r',label='NPF')
    a1[0].plot(newtimes,r14lrp/a14qrp[6],color='b',label='RPF')
    a1[0].set_xlim(3.95,3.0)
    a1[0].set_ylim(0,0.25)
    a1[0].set_ylabel('Relative probability / RP(Imbrium)')
    a1[0].set_title('(a) Apollo 14')
    a1[0].legend()




    matplotlib.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.savefig('figure_7.pdf')

    return

def generate_figure_8():
    '''Generates plot of NPF and RPF including local and sub-basin melt, with and without max rays'''

    #These are the simulations that are classified as outliers in Figure 9

    outliers1 = np.array([12,15,34,38,41,49,50,52,59,61,70,82,84])
    #outliers2 = np.array([1,2,4,17,23,26,27,31,37,40,45,50,57,71,78,81])
    outliers2 = np.array([1,2,4,17,23,26,27,31,37,40,45,50,57,71,78,81,6,29,32,34,49,54,59,62,66,76,80]) #These are not outliers, but they're runs that have the most Iridum melt
    outliers3 = np.array([12,14,18,21,26,40,48,55,63,67,74,80])
    outliers4 = np.array([0,11,33,41,45,46,68,70,74,84])

    o2 = outliers2 + 85
    o3 = outliers3 + 170
    o4 = outliers4 + 255

    combined_outliers = np.concatenate((outliers1,o2,o3,o4))

    #Generate "modified" RP plots (ending in 'm') that remove the simulations above

    a14qam = a14qa.drop(outliers1)
    a14lam = a14la.drop(outliers1)
    a14mts1 = a14melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a14mtsm = a14mts1.drop(outliers1)
    a14crpm, a14lrpm, a14qrpm = calculate_combined_relative_probabilities(a14lam.drop('Unnamed: 0',axis=1),a14qam.drop('Unnamed: 0',axis=1))

    a15qam = a15qa.drop(outliers2)
    a15lam = a15la.drop(outliers2)
    a15mts1 = a15melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a15mtsm = a15mts1.drop(outliers2)
    a15crpm, a15lrpm, a15qrpm = calculate_combined_relative_probabilities(a15lam.drop('Unnamed: 0',axis=1),a15qam.drop('Unnamed: 0',axis=1))

    a16qam = a16qa.drop(outliers3)
    a16lam = a16la.drop(outliers3)
    a16mts1 = a16melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a16mtsm = a16mts1.drop(outliers3)
    a16crpm, a16lrpm, a16qrpm = calculate_combined_relative_probabilities(a16lam.drop('Unnamed: 0',axis=1),a16qam.drop('Unnamed: 0',axis=1))

    a17qam = a17qa.drop(outliers4)
    a17lam = a17la.drop(outliers4)
    a17mts1 = a17melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a17mtsm = a17mts1.drop(outliers4)
    a17crpm, a17lrpm, a17qrpm = calculate_combined_relative_probabilities(a17lam.drop('Unnamed: 0',axis=1),a17qam.drop('Unnamed: 0',axis=1))

    cqm = combined_qmc.drop(combined_outliers)
    clm = combined_local.drop(combined_outliers)
    ccrpm, cclrpm, ccqrpm = calculate_combined_relative_probabilities(clm,cqm)

    r14la = convert_npf_to_rpf(a14la)
    r15la = convert_npf_to_rpf(a15la)
    r16la = convert_npf_to_rpf(a16la)
    r17la = convert_npf_to_rpf(a17la)
    r14lam = convert_npf_to_rpf(a14lam)
    r15lam = convert_npf_to_rpf(a15lam)
    r16lam = convert_npf_to_rpf(a16lam)
    r17lam = convert_npf_to_rpf(a17lam)

    r14crp, r14lrp, r14qrp = calculate_combined_relative_probabilities(r14la.drop('Unnamed: 0',axis=1),a14qa.drop('Unnamed: 0',axis=1))
    r15crp, r15lrp, r15qrp = calculate_combined_relative_probabilities(r15la.drop('Unnamed: 0',axis=1),a15qa.drop('Unnamed: 0',axis=1))
    r16crp, r16lrp, r16qrp = calculate_combined_relative_probabilities(r16la.drop('Unnamed: 0',axis=1),a16qa.drop('Unnamed: 0',axis=1))
    r17crp, r17lrp, r17qrp = calculate_combined_relative_probabilities(r17la.drop('Unnamed: 0',axis=1),a17qa.drop('Unnamed: 0',axis=1))

    r14crpm, r14lrpm, r14qrpm = calculate_combined_relative_probabilities(r14lam.drop('Unnamed: 0',axis=1),a14qam.drop('Unnamed: 0',axis=1))
    r15crpm, r15lrpm, r15qrpm = calculate_combined_relative_probabilities(r15lam.drop('Unnamed: 0',axis=1),a15qam.drop('Unnamed: 0',axis=1))
    r16crpm, r16lrpm, r16qrpm = calculate_combined_relative_probabilities(r16lam.drop('Unnamed: 0',axis=1),a16qam.drop('Unnamed: 0',axis=1))
    r17crpm, r17lrpm, r17qrpm = calculate_combined_relative_probabilities(r17lam.drop('Unnamed: 0',axis=1),a17qam.drop('Unnamed: 0',axis=1))


    #Plot Figure 8

    f1, a1 = plt.subplots(nrows=4,ncols=2,figsize=(15,15))
    a1[3,0].plot(newtimes,a17lrp/a17qrp[6],color='b',label='Melt from small craters')
    a1[3,0].plot(newtimes,a17qrp/a17qrp[6],color='#af4d43',label='Melt from basins and sub-basins')
    a1[3,0].plot(newtimes,a17crp/a17qrp[6],color='k',label='Total melt')
    a1[3,0].plot(newtimes,a17lrpm/a17qrpm[6],color='b',ls='--')
    a1[3,0].plot(newtimes,a17qrpm/a17qrpm[6],color='#af4d43',ls='--')
    a1[3,0].plot(newtimes,a17crpm/a17qrpm[6],color='k',ls='--')
    a1[3,0].set_xlim(3.8,3.0)
    a1[3,0].set_xlabel('Age (Ga)')
    a1[3,0].set_ylabel('RP / RP(Imbrium)')
    # a1[3,0].set_title('Apollo 17')
    a1[3,0].legend()
    a1[2,0].plot(newtimes,a16lrp/a16qrp[6],color='b',label='Melt from small craters')
    a1[2,0].plot(newtimes,a16qrp/a16qrp[6],color='#af4d43',label='Melt from basins and sub-basins')
    a1[2,0].plot(newtimes,a16crp/a16qrp[6],color='k',label='Total melt')
    a1[2,0].plot(newtimes,a16lrpm/a16qrpm[6],color='b',ls='--')
    a1[2,0].plot(newtimes,a16qrpm/a16qrpm[6],color='#af4d43',ls='--')
    a1[2,0].plot(newtimes,a16crpm/a16qrpm[6],color='k',ls='--')
    a1[2,0].set_xlim(3.8,3.0)
    # a1[2,0].set_xlabel('Age (Ga)')
    a1[2,0].set_ylabel('RP / RP(Imbrium)')
    # a1[2,0].set_title('Apollo 16')
    a1[2,0].legend()
    a1[1,0].plot(newtimes,a15lrp/a15qrp[6],color='b',label='Melt from small craters')
    a1[1,0].plot(newtimes,a15qrp/a15qrp[6],color='#af4d43',label='Melt from basins and sub-basins')
    a1[1,0].plot(newtimes,a15crp/a15qrp[6],color='k',label='Total melt')
    a1[1,0].plot(newtimes,a15lrpm/a15qrpm[6],color='b',ls='--')
    a1[1,0].plot(newtimes,a15qrpm/a15qrpm[6],color='#af4d43',ls='--')
    a1[1,0].plot(newtimes,a15crpm/a15qrpm[6],color='k',ls='--')
    a1[1,0].set_xlim(3.8,3.0)
    # a1[1,0].set_xlabel('Age (Ga)')
    a1[1,0].set_ylabel('RP / RP(Imbrium)')
    # a1[1,0].set_title('Apollo 15')
    a1[1,0].legend()
    a1[0,0].plot(newtimes,a14lrp/a14qrp[6],color='b',label='Melt from small craters')
    a1[0,0].plot(newtimes,a14qrp/a14qrp[6],color='#af4d43',label='Melt from basins and sub-basins')
    a1[0,0].plot(newtimes,a14crp/a14qrp[6],color='k',label='Total melt')
    a1[0,0].plot(newtimes,a14lrpm/a14qrpm[6],color='b',ls='--')
    a1[0,0].plot(newtimes,a14qrpm/a14qrpm[6],color='#af4d43',ls='--')
    a1[0,0].plot(newtimes,a14crpm/a14qrpm[6],color='k',ls='--')
    a1[0,0].set_xlim(3.8,3.0)
    # a1[0,0].set_xlabel('Age (Ga)')
    a1[0,0].set_ylabel('RP / RP(Imbrium)')
    a1[0,0].set_title('NPF')
    a1[0,0].legend()



    a1[3,1].plot(newtimes,r17lrp/a17qrp[6],color='b',label='Melt from small craters')
    a1[3,1].plot(newtimes,a17qrp/a17qrp[6],color='#af4d43',label='Melt from basins and sub-basins')
    a1[3,1].plot(newtimes,(r17lrp+a17qrp)/a17qrp[6],color='k',label='Total melt')
    a1[3,1].plot(newtimes,r17lrpm/r17qrpm[6],color='b',ls='--')
    a1[3,1].plot(newtimes,a17qrpm/a17qrpm[6],color='#af4d43',ls='--')
    a1[3,1].plot(newtimes,(r17lrpm+a17qrpm)/a17qrpm[6],color='k',ls='--')
    a1[3,1].set_xlim(3.8,3.0)
    a1[3,1].set_xlabel('Age (Ga)')
    a1[3,1].set_ylabel('RP / RP(Imbrium)')
    # a1[3,1].set_title('Apollo 17')
    a1[3,1].legend()
    a1[2,1].plot(newtimes,r16lrp/a16qrp[6],color='b',label='Melt from small craters')
    a1[2,1].plot(newtimes,a16qrp/a16qrp[6],color='#af4d43',label='Melt from basins and sub-basins')
    a1[2,1].plot(newtimes,(r16lrp+a16qrp)/a16qrp[6],color='k',label='Total melt')
    a1[2,1].plot(newtimes,r16lrpm/r16qrpm[6],color='b',ls='--')
    a1[2,1].plot(newtimes,a16qrpm/r16qrpm[6],color='#af4d43',ls='--')
    a1[2,1].plot(newtimes,(r16lrpm+a16qrpm)/a16qrpm[6],color='k',ls='--')
    a1[2,1].set_xlim(3.8,3.0)
    # a1[2,1].set_xlabel('Age (Ga)')
    a1[2,1].set_ylabel('RP / RP(Imbrium)')
    # a1[2,1].set_title('Apollo 16')
    a1[2,1].legend()
    a1[1,1].plot(newtimes,r15lrp/a15qrp[6],color='b',label='Melt from small craters')
    a1[1,1].plot(newtimes,a15qrp/a15qrp[6],color='#af4d43',label='Melt from basins and sub-basins')
    a1[1,1].plot(newtimes,(r15lrp+a15qrp)/a15qrp[6],color='k',label='Total melt')
    a1[1,1].plot(newtimes,r15lrpm/r15qrpm[6],color='b',ls='--')
    a1[1,1].plot(newtimes,a15qrpm/r15qrpm[6],color='#af4d43',ls='--')
    a1[1,1].plot(newtimes,(r15lrpm+a15qrpm)/a15qrpm[6],color='k',ls='--')
    a1[1,1].set_xlim(3.8,3.0)
    # a1[1,1].set_xlabel('Age (Ga)')
    a1[1,1].set_ylabel('RP / RP(Imbrium)')
    # a1[1,1].set_title('Apollo 15')
    a1[1,1].legend()
    a1[0,1].plot(newtimes,r14lrp/a14qrp[6],color='b',label='Melt from small craters')
    a1[0,1].plot(newtimes,a14qrp/a14qrp[6],color='#af4d43',label='Melt from basins and sub-basins')
    a1[0,1].plot(newtimes,(r14lrp+a14qrp)/a14qrp[6],color='k',label='Total melt')
    a1[0,1].plot(newtimes,r14lrpm/r14qrpm[6],color='b',ls='--')
    a1[0,1].plot(newtimes,a14qrpm/r14qrpm[6],color='#af4d43',ls='--')
    a1[0,1].plot(newtimes,(r14lrpm+a14qrpm)/a14qrpm[6],color='k',ls='--')
    a1[0,1].set_xlim(3.8,3.0)
    #a1[0,1].set_xlabel('Age (Ga)')
    a1[0,1].set_ylabel('RP / RP(Imbrium)')
    a1[0,1].set_title('RPF')
    a1[0,1].legend()

    a1[0,1].text(2.95,0.2,'A14',fontsize=20)
    a1[1,1].text(2.95,0.125,'A15',fontsize=20)
    a1[2,1].text(2.95,0.6,'A16',fontsize=20)
    a1[3,1].text(2.95,0.35,'A17',fontsize=20)

    for i in range(4):
        for j in range(2):
            #a1[i,j].set_yticks([])
            a1[0,j].set_xticks([])
            a1[1,j].set_xticks([])
            a1[2,j].set_xticks([])
    #         a1[i,j].set_ylim(0,0.3)

    for i in range(2):
        a1[0,i].set_ylim(0,0.5)
        a1[1,i].set_ylim(0,0.25)
        a1[2,i].set_ylim(0,1)
        a1[3,i].set_ylim(0,0.9)

    matplotlib.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.savefig('figure_8.pdf')

    return

def generate_figure_9():
    '''Generates box plot of Imbrian melt by crater name'''

    a14ml = a14melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a15ml = a15melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a16ml = a16melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a17ml = a17melts.drop('Unnamed: 0',axis=1) / (pix*pix)

    a14_D = a14ml.loc[:,'Lansberg':'Mairan']
    a15_D = a15ml.loc[:,'Lansberg':'Mairan']
    a16_D = a16ml.loc[:,'Lansberg':'Mairan']
    a17_D = a17ml.loc[:,'Lansberg':'Mairan']

    d14 = a14la.loc[:,"3.8072054":"3.08886476"]
    d15 = a15la.loc[:,"3.8072054":"3.08886476"]
    d16 = a16la.loc[:,"3.8072054":"3.08886476"]
    d17 = a17la.loc[:,"3.8072054":"3.08886476"]
    a14l_D = d14.sum(axis=1)
    a15l_D = d15.sum(axis=1)
    a16l_D = d16.sum(axis=1)
    a17l_D = d17.sum(axis=1)
    a14_D['Local'] = a14l_D
    a15_D['Local'] = a15l_D
    a16_D['Local'] = a16l_D
    a17_D['Local'] = a17l_D

    a14_D_N = a14_D.apply(lambda x: x / x.sum(), axis=1)
    a15_D_N = a15_D.apply(lambda x: x / x.sum(), axis=1)
    a16_D_N = a16_D.apply(lambda x: x / x.sum(), axis=1)
    a17_D_N = a17_D.apply(lambda x: x / x.sum(), axis=1)

    a14mfs_D = []
    a14names_D = []
    for crater in a14_D_N:
        if np.percentile(a14_D_N[crater],75) > 0.03 and np.max(a14_D_N[crater]) < 1.01:
            a14mfs_D.append(a14_D_N[crater])
            a14names_D.append(crater)

    a15mfs_D = []
    a15names_D = []
    for crater in a15_D_N:
        if np.percentile(a15_D_N[crater],75) > 0.03 and np.max(a15_D_N[crater]) < 1.01:
            a15mfs_D.append(a15_D_N[crater])
            a15names_D.append(crater)
    a16mfs_D = []
    a16names_D = []
    for crater in a16_D_N:
        if np.percentile(a16_D_N[crater],75) > 0.03 and np.max(a16_D_N[crater]) < 1.01:
            a16mfs_D.append(a16_D_N[crater])
            a16names_D.append(crater)

    a17mfs_D = []
    a17names_D = []
    for crater in a17_D_N:
        if np.percentile(a17_D_N[crater],75) > 0.03 and np.max(a17_D_N[crater]) < 1.01:
            a17mfs_D.append(a17_D_N[crater])
            a17names_D.append(crater)


    #Plot Figure 9

    f3, a3 = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
    a3[0,0].boxplot(a14mfs_D)
    a3[0,0].set_title('(a) Apollo 14')
    a3[0,0].set_xticklabels(a14names_D)
    a3[0,1].boxplot(a15mfs_D)
    a3[0,1].set_title('(b) Apollo 15')
    a3[0,1].set_xticklabels(a15names_D)
    a3[1,0].boxplot(a16mfs_D)
    a3[1,0].set_title('(c) Apollo 16')
    a3[1,0].set_xticklabels(a16names_D)
    a3[1,1].boxplot(a17mfs_D)
    a3[1,1].set_title('(d) Apollo 17')
    a3[1,1].set_xticklabels(a17names_D)
    for i in range(2):
        for j in range(2):
            a3[i,j].set_ylabel('Fraction of total melt with this age')
    matplotlib.rcParams.update({'font.size': 16})        
    plt.tight_layout()
    plt.savefig('figure_9.pdf')

    return

def generate_figure_10():
    '''Generates figure for A15 with different ages for Iridum'''

    a15crp387, a15lrp387, a15qrp387 = change_iridum_age(a15melts.drop('Unnamed: 0',axis=1) / (pix*pix),a15la,a15qa.copy(),age='3.87903947')
    a15crp380, a15lrp380, a15qrp380 = change_iridum_age(a15melts.drop('Unnamed: 0',axis=1) / (pix*pix),a15la,a15qa.copy())
    a15crp373, a15lrp373, a15qrp373 = change_iridum_age(a15melts.drop('Unnamed: 0',axis=1) / (pix*pix),a15la,a15qa.copy(),age='3.73537134')
    a15crp366, a15lrp366, a15qrp366 = change_iridum_age(a15melts.drop('Unnamed: 0',axis=1) / (pix*pix),a15la,a15qa.copy(),age='3.66353727')

    #Plot figure 10

    f1 = plt.figure(1)
    f1.clf()
    a1 = f1.add_subplot(111)
    a1.plot(newtimes,a15qrp366/a15crp366[5],color='k',label='Iridum = 3.66 Ga')
    a1.plot(newtimes,a15qrp373/a15crp373[5],color='k',ls='--',label ='Iridum = 3.73 Ga')
    a1.plot(newtimes,a15qrp380/a15crp380[5],color='k',ls=':',label='Iridum = 3.80 Ga')
    a1.plot(newtimes,a15qrp387/a15crp387[5],color='k',ls='-.',label='Iridum = 3.87 Ga')
    a1.set_xlim(3.97,3.0)
    a1.set_ylim(0,1)
    a1.set_yticks([])
    a1.set_ylabel('Relative probability')
    a1.set_xlabel('Age (Ga)')
    a1.legend()
    plt.tight_layout()
    plt.rcParams['figure.figsize'] = [7,5]
    plt.savefig('figure_10.pdf')

    return

def generate_figure_11():
    '''Generates relative probability curve with Iridum being 3.80 Ga'''

    a14crp380, a14lrp380, a14qrp380 = change_iridum_age(a14melts.drop('Unnamed: 0',axis=1) / (pix*pix),a14la,a14qa.copy())
    a15crp380, a15lrp380, a15qrp380 = change_iridum_age(a15melts.drop('Unnamed: 0',axis=1) / (pix*pix),a15la,a15qa.copy())
    a16crp380, a16lrp380, a16qrp380 = change_iridum_age(a16melts.drop('Unnamed: 0',axis=1) / (pix*pix),a16la,a16qa.copy())
    a17crp380, a17lrp380, a17qrp380 = change_iridum_age(a17melts.drop('Unnamed: 0',axis=1) / (pix*pix),a17la,a17qa.copy())

    r14la = convert_npf_to_rpf(a14la)
    r15la = convert_npf_to_rpf(a15la)
    r16la = convert_npf_to_rpf(a16la)
    r17la = convert_npf_to_rpf(a17la)

    r14crp380, r14lrp380, r14qrp380 = change_iridum_age(a14melts.drop('Unnamed: 0',axis=1) / (pix*pix),r14la,a14qa.copy())
    r15crp380, r15lrp380, r15qrp380 = change_iridum_age(a15melts.drop('Unnamed: 0',axis=1) / (pix*pix),r15la,a15qa.copy())
    r16crp380, r16lrp380, r16qrp380 = change_iridum_age(a16melts.drop('Unnamed: 0',axis=1) / (pix*pix),r16la,a16qa.copy())
    r17crp380, r17lrp380, r17qrp380 = change_iridum_age(a17melts.drop('Unnamed: 0',axis=1) / (pix*pix),r17la,a17qa.copy())

    outliers1 = np.array([12,15,34,38,41,49,50,52,59,61,70,82,84])
    outliers2 = np.array([1,2,4,17,23,26,27,31,37,40,45,50,57,71,78,81]) #For this site, there were no statistical outliers, so for an example I included all runs where Iridum > 0.6 total melt in the Imbrian
    outliers3 = np.array([12,14,18,21,26,40,48,55,63,67,74,80])
    outliers4 = np.array([0,11,33,41,45,46,68,70,74,84])

    o2 = outliers2 + 85
    o3 = outliers3 + 170
    o4 = outliers4 + 255

    combined_outliers = np.concatenate((outliers1,o2,o3,o4))

    #Generate "modified" RP plots (ending in 'm') that remove the simulations above

    a14qam = a14qa.drop(outliers1)
    a14lam = a14la.drop(outliers1)
    a14mts1 = a14melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a14mtsm = a14mts1.drop(outliers1)

    a15qam = a15qa.drop(outliers2)
    a15lam = a15la.drop(outliers2)
    a15mts1 = a15melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a15mtsm = a15mts1.drop(outliers2)
    a15crpm, a15lrpm, a15qrpm = calculate_combined_relative_probabilities(a15lam.drop('Unnamed: 0',axis=1),a15qam.drop('Unnamed: 0',axis=1))

    a16qam = a16qa.drop(outliers3)
    a16lam = a16la.drop(outliers3)
    a16mts1 = a16melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a16mtsm = a16mts1.drop(outliers3)
    a16crpm, a16lrpm, a16qrpm = calculate_combined_relative_probabilities(a16lam.drop('Unnamed: 0',axis=1),a16qam.drop('Unnamed: 0',axis=1))

    a17qam = a17qa.drop(outliers4)
    a17lam = a17la.drop(outliers4)
    a17mts1 = a17melts.drop('Unnamed: 0',axis=1) / (pix*pix)
    a17mtsm = a17mts1.drop(outliers4)
    a17crpm, a17lrpm, a17qrpm = calculate_combined_relative_probabilities(a17lam.drop('Unnamed: 0',axis=1),a17qam.drop('Unnamed: 0',axis=1))

    a14crp380m, a14lrp380m, a14qrp380m = change_iridum_age(a14mtsm,a14lam,a14qam.copy())
    a15crp380m, a15lrp380m, a15qrp380m = change_iridum_age(a15mtsm,a15lam,a15qam.copy())
    a16crp380m, a16lrp380m, a16qrp380m = change_iridum_age(a16mtsm,a16lam,a16qam.copy())
    a17crp380m, a17lrp380m, a17qrp380m = change_iridum_age(a17mtsm,a17lam,a17qam.copy())

    r14lam = convert_npf_to_rpf(a14lam)
    r15lam = convert_npf_to_rpf(a15lam)
    r16lam = convert_npf_to_rpf(a16lam)
    r17lam = convert_npf_to_rpf(a17lam)

    r14crp380m, r14lrp380m, r14qrp380m = change_iridum_age(a14mtsm,r14lam,a14qa.copy())
    r15crp380m, r15lrp380m, r15qrp380m = change_iridum_age(a15mtsm,r15lam,a15qa.copy())
    r16crp380m, r16lrp380m, r16qrp380m = change_iridum_age(a16mtsm,r16lam,a16qa.copy())
    r17crp380m, r17lrp380m, r17qrp380m = change_iridum_age(a17mtsm,r17lam,a17qa.copy())

    #Plot Figure 11

    f1, a1 = plt.subplots(nrows=4,ncols=2,figsize=(15,15))
    a1[3,0].plot(newtimes,a17lrp380/a17qrp380[5],color='b',label='Melt from small craters')
    a1[3,0].plot(newtimes,a17qrp380/a17qrp380[5],color='#af4d43',label='Melt from basins and sub-basins')
    a1[3,0].plot(newtimes,a17crp380/a17qrp380[5],color='k',label='Total melt')
    a1[3,0].plot(newtimes,a17lrp380m/a17qrp380m[5],color='b',ls='--')
    a1[3,0].plot(newtimes,a17qrp380m/a17qrp380m[5],color='#af4d43',ls='--')
    a1[3,0].plot(newtimes,a17crp380m/a17qrp380m[5],color='k',ls='--')
    a1[3,0].set_xlim(3.89,3.0)
    a1[3,0].set_xlabel('Age (Ga)')
    a1[3,0].set_ylabel('RP / RP(Imbrium)')
    # a1[3,0].set_title('Apollo 17')
    a1[3,0].legend()
    a1[2,0].plot(newtimes,a16lrp380/a16qrp380[5],color='b',label='Melt from small craters')
    a1[2,0].plot(newtimes,a16qrp380/a16qrp380[5],color='#af4d43',label='Melt from basins and sub-basins')
    a1[2,0].plot(newtimes,a16crp380/a16qrp380[5],color='k',label='Total melt')
    a1[2,0].plot(newtimes,a16lrp380m/a16qrp380m[5],color='b',ls='--')
    a1[2,0].plot(newtimes,a16qrp380m/a16qrp380m[5],color='#af4d43',ls='--')
    a1[2,0].plot(newtimes,a16crp380m/a16qrp380m[5],color='k',ls='--')
    a1[2,0].set_xlim(3.89,3.0)
    # a1[2,0].set_xlabel('Age (Ga)')
    a1[2,0].set_ylabel('RP / RP(Imbrium)')
    # a1[2,0].set_title('Apollo 16')
    a1[2,0].legend()
    a1[1,0].plot(newtimes,a15lrp380/a15qrp380[5],color='b',label='Melt from small craters')
    a1[1,0].plot(newtimes,a15qrp380/a15qrp380[5],color='#af4d43',label='Melt from basins and sub-basins')
    a1[1,0].plot(newtimes,a15crp380/a15qrp380[5],color='k',label='Total melt')
    a1[1,0].plot(newtimes,a15lrp380m/a15qrp380m[5],color='b',ls='--')
    a1[1,0].plot(newtimes,a15qrp380m/a15qrp380m[5],color='#af4d43',ls='--')
    a1[1,0].plot(newtimes,a15crp380m/a15qrp380m[5],color='k',ls='--')
    a1[1,0].set_xlim(3.89,3.0)
    # a1[1,0].set_xlabel('Age (Ga)')
    a1[1,0].set_ylabel('RP / RP(Imbrium)')
    # a1[1,0].set_title('Apollo 15')
    a1[1,0].legend()
    a1[0,0].plot(newtimes,a14lrp380/a14qrp380[5],color='b',label='Melt from small craters')
    a1[0,0].plot(newtimes,a14qrp380/a14qrp380[5],color='#af4d43',label='Melt from basins and sub-basins')
    a1[0,0].plot(newtimes,a14crp380/a14qrp380[5],color='k',label='Total melt')
    a1[0,0].plot(newtimes,a14lrp380m/a14qrp380m[5],color='b',ls='--')
    a1[0,0].plot(newtimes,a14qrp380m/a14qrp380m[5],color='#af4d43',ls='--')
    a1[0,0].plot(newtimes,a14crp380m/a14qrp380m[5],color='k',ls='--')
    a1[0,0].set_xlim(3.89,3.0)
    # a1[0,0].set_xlabel('Age (Ga)')
    a1[0,0].set_ylabel('RP / RP(Imbrium)')
    a1[0,0].set_title('NPF')
    a1[0,0].legend()


    a1[3,1].plot(newtimes,r17lrp380/a17qrp380[5],color='b',label='Melt from small craters')
    a1[3,1].plot(newtimes,a17qrp380/a17qrp380[5],color='#af4d43',label='Melt from basins and sub-basins')
    a1[3,1].plot(newtimes,(r17lrp380+a17qrp380)/a17qrp380[5],color='k',label='Total melt')
    a1[3,1].plot(newtimes,r17lrp380m/r17qrp380m[5],color='b',ls='--')
    a1[3,1].plot(newtimes,a17qrp380m/a17qrp380m[5],color='#af4d43',ls='--')
    a1[3,1].plot(newtimes,(r17lrp380m+a17qrp380m)/a17qrp380m[5],color='k',ls='--')
    a1[3,1].set_xlim(3.89,3.0)
    a1[3,1].set_xlabel('Age (Ga)')
    a1[3,1].set_ylabel('RP / RP(Imbrium)')
    # a1[3,1].set_title('Apollo 17')
    a1[3,1].legend()
    a1[2,1].plot(newtimes,r16lrp380/a16qrp380[5],color='b',label='Melt from small craters')
    a1[2,1].plot(newtimes,a16qrp380/a16qrp380[5],color='#af4d43',label='Melt from basins and sub-basins')
    a1[2,1].plot(newtimes,(r16lrp380+a16qrp380)/a16qrp380[5],color='k',label='Total melt')
    a1[2,1].plot(newtimes,r16lrp380m/r16qrp380m[5],color='b',ls='--')
    a1[2,1].plot(newtimes,a16qrp380m/r16qrp380m[5],color='#af4d43',ls='--')
    a1[2,1].plot(newtimes,(r16lrp380m+a16qrp380m)/a16qrp380m[5],color='k',ls='--')
    a1[2,1].set_xlim(3.89,3.0)
    # a1[2,1].set_xlabel('Age (Ga)')
    a1[2,1].set_ylabel('RP / RP(Imbrium)')
    # a1[2,1].set_title('Apollo 16')
    a1[2,1].legend()
    a1[1,1].plot(newtimes,r15lrp380/a15qrp380[5],color='b',label='Melt from small craters')
    a1[1,1].plot(newtimes,a15qrp380/a15qrp380[5],color='#af4d43',label='Melt from basins and sub-basins')
    a1[1,1].plot(newtimes,(r15lrp380+a15qrp380)/a15qrp380[5],color='k',label='Total melt')
    a1[1,1].plot(newtimes,r15lrp380m/r15qrp380m[5],color='b',ls='--')
    a1[1,1].plot(newtimes,a15qrp380m/r15qrp380m[5],color='#af4d43',ls='--')
    a1[1,1].plot(newtimes,(r15lrp380m+a15qrp380m)/a15qrp380m[5],color='k',ls='--')
    a1[1,1].set_xlim(3.89,3.0)
    # a1[1,1].set_xlabel('Age (Ga)')
    a1[1,1].set_ylabel('RP / RP(Imbrium)')
    # a1[1,1].set_title('Apollo 15')
    a1[1,1].legend()
    a1[0,1].plot(newtimes,r14lrp380/a14qrp380[5],color='b',label='Melt from small craters')
    a1[0,1].plot(newtimes,a14qrp380/a14qrp380[5],color='#af4d43',label='Melt from basins and sub-basins')
    a1[0,1].plot(newtimes,(r14lrp380+a14qrp380)/a14qrp380[5],color='k',label='Total melt')
    a1[0,1].plot(newtimes,r14lrp380m/r14qrp380m[5],color='b',ls='--')
    a1[0,1].plot(newtimes,a14qrp380m/r14qrp380m[5],color='#af4d43',ls='--')
    a1[0,1].plot(newtimes,(r14lrp380m+a14qrp380m)/a14qrp380m[5],color='k',ls='--')
    a1[0,1].set_xlim(3.89,3.0)
    #a1[0,1].set_xlabel('Age (Ga)')
    a1[0,1].set_ylabel('RP / RP(Imbrium)')
    a1[0,1].set_title('RPF')
    a1[0,1].legend()

    a1[0,1].text(2.95,0.2,'A14',fontsize=20)
    a1[1,1].text(2.95,0.125,'A15',fontsize=20)
    a1[2,1].text(2.95,0.6,'A16',fontsize=20)
    a1[3,1].text(2.95,0.35,'A17',fontsize=20)

    for i in range(4):
        for j in range(2):
            #a1[i,j].set_yticks([])
            a1[0,j].set_xticks([])
            a1[1,j].set_xticks([])
            a1[2,j].set_xticks([])
    #         a1[i,j].set_ylim(0,0.3)

    for i in range(2):
        a1[0,i].set_ylim(0,0.4)
        a1[1,i].set_ylim(0,0.25)
        a1[2,i].set_ylim(0,1)
        a1[3,i].set_ylim(0,0.8)

    matplotlib.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.savefig('figure_11.pdf')

    return

    # Generate F.13 and 14

def generate_scaled_plots(outliers=False): #Generates Figure 13 if outliers=False and Figure 14 if outliers=True. Also generates Figure 12.

    #Iridum 3.87Ga in F.13 (even though the suffix says 380)

    a14crp380, a14lrp380, a14qrp380 = change_iridum_age(a14melts.drop('Unnamed: 0',axis=1) / (pix*pix),a14la,a14qa.copy(),age='3.87903947')
    a15crp380, a15lrp380, a15qrp380 = change_iridum_age(a15melts.drop('Unnamed: 0',axis=1) / (pix*pix),a15la,a15qa.copy(),age='3.87903947')
    a16crp380, a16lrp380, a16qrp380 = change_iridum_age(a16melts.drop('Unnamed: 0',axis=1) / (pix*pix),a16la,a16qa.copy(),age='3.87903947')
    a17crp380, a17lrp380, a17qrp380 = change_iridum_age(a17melts.drop('Unnamed: 0',axis=1) / (pix*pix),a17la,a17qa.copy(),age='3.87903947')

    r14la = convert_npf_to_rpf(a14la)
    r15la = convert_npf_to_rpf(a15la)
    r16la = convert_npf_to_rpf(a16la)
    r17la = convert_npf_to_rpf(a17la)

    r14crp380, r14lrp380, r14qrp380 = change_iridum_age(a14melts.drop('Unnamed: 0',axis=1) / (pix*pix),r14la,a14qa.copy(),age='3.87903947')
    r15crp380, r15lrp380, r15qrp380 = change_iridum_age(a15melts.drop('Unnamed: 0',axis=1) / (pix*pix),r15la,a15qa.copy(),age='3.87903947')
    r16crp380, r16lrp380, r16qrp380 = change_iridum_age(a16melts.drop('Unnamed: 0',axis=1) / (pix*pix),r16la,a16qa.copy(),age='3.87903947')
    r17crp380, r17lrp380, r17qrp380 = change_iridum_age(a17melts.drop('Unnamed: 0',axis=1) / (pix*pix),r17la,a17qa.copy(),age='3.87903947')

    #Move samples associated with Imbrium to 3.95Ga bin

    #See supporting information for why this number of samples is associated with Imbrium

    hist14[53] += (hist14[52] - 3)
    hist14[52] = 3
    hist15[53] += (hist15[52] - 1)
    hist15[52] = 1
    hist16[53] += (hist16[52] - 11)
    hist16[52] = 11
    hist17[53] += (hist17[52] - 1)
    hist17[52] = 1

    # Plot Figure 12

    f1 = plt.figure(1,figsize=(5,5))
    f1.clf()
    a1 = f1.add_subplot(111)
    a1.bar(bin_centers, hist14, bottom=0, width=0.07183406, color='#fef0d9', edgecolor='k', label='Apollo 14')
    a1.bar(bin_centers, hist15, bottom=hist14, width=0.07183406, color='#fdcc8a', edgecolor='k', label='Apollo 15')
    a1.bar(bin_centers, hist16, bottom=hist14+hist15, width=0.07183406, color='#fc8d59', edgecolor='k', label='Apollo 16')
    a1.bar(bin_centers, hist17, bottom=hist14+hist15+hist16, width=0.07183406, color='#d7301f', edgecolor='k', label='Apollo 17')
    a1.set_xlim(3.87,3.0)
    a1.set_ylim(0,23)
    a1.set_xlabel('Age (Ga)')
    a1.set_ylabel('Number of samples')
    a1.legend()
    #plt.rcParams['figure.figsize'] = [7, 5]
    matplotlib.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.savefig('figure_12.pdf')

    if not outliers:
        #Find scale factor between relative probability and number of Imbrian-aged samples

        f14i = np.sum(hist14[0:53]) / np.sum(a14crp380[-54:-41])
        f15i = np.sum(hist15[0:53]) / np.sum(a15crp380[-54:-41])
        f16i = np.sum(hist16[0:53]) / np.sum(a16crp380[-54:-41])
        f17i = np.sum(hist17[0:53]) / np.sum(a17crp380[-54:-41])

        f14ir = np.sum(hist14[0:53]) / np.sum(r14crp380[-54:-42]) + ((0.3085113*r14crp380[-55]))
        f15ir = np.sum(hist15[0:53]) / np.sum(r15crp380[-54:-42]) + ((0.3085113*r15crp380[-55]))
        f16ir = np.sum(hist16[0:53]) / np.sum(r16crp380[-54:-42]) + ((0.3085113*r16crp380[-55]))
        f17ir = np.sum(hist17[0:53]) / np.sum(r17crp380[-54:-42]) + ((0.3085113*r17crp380[-55]))

        #Apply this scale factor

        sa14lrpi = a14lrp * f14i
        sa14qrpi = a14qrp * f14i
        sa14crpi = a14crp * f14i
        sa15lrpi = a15lrp * f15i
        sa15qrpi = a15qrp * f15i
        sa15crpi = a15crp * f15i
        sa16lrpi = a16lrp * f16i
        sa16qrpi = a16qrp * f16i
        sa16crpi = a16crp * f16i
        sa17lrpi = a17lrp * f17i
        sa17qrpi = a17qrp * f17i
        sa17crpi = a17crp * f17i

        sa14lrp380i = a14lrp380 * f14i
        sa14qrp380i = a14qrp380 * f14i
        sa14crp380i = a14crp380 * f14i
        sa15lrp380i = a15lrp380 * f15i
        sa15qrp380i = a15qrp380 * f15i
        sa15crp380i = a15crp380 * f15i
        sa16lrp380i = a16lrp380 * f16i
        sa16qrp380i = a16qrp380 * f16i
        sa16crp380i = a16crp380 * f16i
        sa17lrp380i = a17lrp380 * f17i
        sa17qrp380i = a17qrp380 * f17i
        sa17crp380i = a17crp380 * f17i

        ra14lrp380i = r14lrp380 * f14ir
        ra14qrp380i = r14qrp380 * f14ir
        ra14crp380i = r14crp380 * f14ir
        ra15lrp380i = r15lrp380 * f15ir
        ra15qrp380i = r15qrp380 * f15ir
        ra15crp380i = r15crp380 * f15ir
        ra16lrp380i = r16lrp380 * f16ir
        ra16qrp380i = r16qrp380 * f16ir
        ra16crp380i = r16crp380 * f16ir
        ra17lrp380i = r17lrp380 * f17ir
        ra17qrp380i = r17qrp380 * f17ir
        ra17crp380i = r17crp380 * f17ir

        salrpi = sa14lrpi + sa15lrpi + sa16lrpi + sa17lrpi
        saqrpi = sa14qrpi + sa15qrpi + sa16qrpi + sa17qrpi

        salrp380i = sa14lrp380i + sa15lrp380i + sa16lrp380i + sa17lrp380i
        saqrp380i = sa14qrp380i + sa15qrp380i + sa16qrp380i + sa17qrp380i

        ralrp380i = ra14lrp380i + ra15lrp380i + ra16lrp380i + ra17lrp380i
        raqrp380i = ra14qrp380i + ra15qrp380i + ra16qrp380i + ra17qrp380i

        #Plot Figure 13

        f1, a1 = plt.subplots(nrows=4,ncols=2,figsize=(15,15))

        a1[0,0].bar(bin_centers+offset,hist14,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 14 Melts')
        a1[0,0].bar(bin_centers2-offset,sa14lrp380i[:-1],yerr=np.sqrt(sa14lrp380i[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[0,0].bar(bin_centers2-offset,sa14qrp380i[:-1],yerr=np.sqrt(sa14qrp380i[:-1]),color='#af4d43',edgecolor='k',bottom=sa14lrp380i[:-1],width=0.07183406/3,label='sub-basin melt')
        a1[0,1].bar(bin_centers+offset,hist14,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 14 Melts')
        a1[0,1].bar(bin_centers2-offset,ra14lrp380i[:-1],yerr=np.sqrt(ra14lrp380i[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[0,1].bar(bin_centers2-offset,ra14qrp380i[:-1],yerr=np.sqrt(ra14qrp380i[:-1]),color='#af4d43',edgecolor='k',bottom=ra14lrp380i[:-1],width=0.07183406/3,label='sub-basin melt')

        a1[1,0].bar(bin_centers+offset,hist15,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 15 Melts')
        a1[1,0].bar(bin_centers2-offset,sa15lrp380i[:-1],yerr=np.sqrt(sa15lrp380i[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[1,0].bar(bin_centers2-offset,sa15qrp380i[:-1],yerr=np.sqrt(sa15qrp380i[:-1]),color='#af4d43',edgecolor='k',bottom=sa15lrp380i[:-1],width=0.07183406/3,label='sub-basin melt')
        a1[1,1].bar(bin_centers+offset,hist15,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 15 Melts')
        a1[1,1].bar(bin_centers2-offset,ra15lrp380i[:-1],yerr=np.sqrt(ra15lrp380i[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[1,1].bar(bin_centers2-offset,ra15qrp380i[:-1],yerr=np.sqrt(ra15qrp380i[:-1]),color='#af4d43',edgecolor='k',bottom=ra15lrp380i[:-1],width=0.07183406/3,label='sub-basin melt')

        a1[2,0].bar(bin_centers+offset,hist16,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 16 Melts')
        a1[2,0].bar(bin_centers2-offset,sa16lrp380i[:-1],yerr=np.sqrt(sa16lrp380i[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[2,0].bar(bin_centers2-offset,sa16qrp380i[:-1],yerr=np.sqrt(sa16qrp380i[:-1]),color='#af4d43',edgecolor='k',bottom=sa16lrp380i[:-1],width=0.07183406/3,label='sub-basin melt')
        a1[2,1].bar(bin_centers+offset,hist16,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 16 Melts')
        a1[2,1].bar(bin_centers2-offset,ra16lrp380i[:-1],yerr=np.sqrt(ra16lrp380i[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[2,1].bar(bin_centers2-offset,ra16qrp380i[:-1],yerr=np.sqrt(ra16qrp380i[:-1]),color='#af4d43',edgecolor='k',bottom=ra16lrp380i[:-1],width=0.07183406/3,label='sub-basin melt')

        a1[3,0].bar(bin_centers+offset,hist17,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 17 Melts')
        a1[3,0].bar(bin_centers2-offset,sa17lrp380i[:-1],yerr=np.sqrt(sa17lrp380i[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[3,0].bar(bin_centers2-offset,sa17qrp380i[:-1],yerr=np.sqrt(sa17qrp380i[:-1]),color='#af4d43',edgecolor='k',bottom=sa17lrp380i[:-1],width=0.07183406/3,label='sub-basin melt')
        a1[3,1].bar(bin_centers+offset,hist17,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 17 Melts')
        a1[3,1].bar(bin_centers2-offset,ra17lrp380i[:-1],yerr=np.sqrt(ra17lrp380i[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[3,1].bar(bin_centers2-offset,ra17qrp380i[:-1],yerr=np.sqrt(ra17qrp380i[:-1]),color='#af4d43',edgecolor='k',bottom=ra17lrp380i[:-1],width=0.07183406/3,label='sub-basin melt')


        a1[0,0].set_ylim(0,8)
        a1[0,1].set_ylim(0,8)
        a1[1,0].set_ylim(0,4)
        a1[1,1].set_ylim(0,4)
        a1[2,0].set_ylim(0,16)
        a1[2,1].set_ylim(0,16)
        a1[3,0].set_ylim(0,5)
        a1[3,1].set_ylim(0,5)

        a1[0,0].set_title('NPF')
        a1[0,1].set_title('RPF')

        a1[3,0].set_xlabel('Age (Ga)')
        a1[3,1].set_xlabel('Age (Ga)')

        for i in range(4):
            for j in range(2):
                a1[i,j].set_xlim(3.87,3.0)
                #a1[i,j].set_xlabel('Age (Ga)')
                a1[i,j].set_ylabel('Number of samples')
                a1[i,j].legend()
                
        # a1[0,0].text(3.8,7.15,'(a)',fontsize=16)
        # a1[0,1].text(3.8,7.15,'(b)',fontsize=16)
        # a1[1,0].text(3.8,2.65,'(c)',fontsize=16)
        # a1[1,1].text(3.8,2.65,'(d)',fontsize=16)
        # a1[2,0].text(3.8,15.15,'(e)',fontsize=16)
        # a1[2,1].text(3.8,15.15,'(f)',fontsize=16)
        # a1[3,0].text(3.8,2.65,'(g)',fontsize=16)
        # a1[3,1].text(3.8,2.65,'(h)',fontsize=16)
                
        matplotlib.rcParams.update({'font.size': 16})
        plt.tight_layout()
                
        plt.savefig('figure_13.pdf')

        return


    else:

        outliers1 = np.array([12,15,34,38,41,49,50,52,59,61,70,82,84])
        outliers2 = np.array([1,2,4,17,23,26,27,31,37,40,45,50,57,71,78,81])
        outliers3 = np.array([12,14,18,21,26,40,48,55,63,67,74,80])
        outliers4 = np.array([0,11,33,41,45,46,68,70,74,84])

        o2 = outliers2 + 85
        o3 = outliers3 + 170
        o4 = outliers4 + 255

        combined_outliers = np.concatenate((outliers1,o2,o3,o4))

        #Generate "modified" RP plots (ending in 'm') that remove the simulations above

        a14qam = a14qa.drop(outliers1)
        a14lam = a14la.drop(outliers1)
        a14mts1 = a14melts.drop('Unnamed: 0',axis=1) / (pix*pix)
        a14mtsm = a14mts1.drop(outliers1)

        a15qam = a15qa.drop(outliers2)
        a15lam = a15la.drop(outliers2)
        a15mts1 = a15melts.drop('Unnamed: 0',axis=1) / (pix*pix)
        a15mtsm = a15mts1.drop(outliers2)
        a15crpm, a15lrpm, a15qrpm = calculate_combined_relative_probabilities(a15lam.drop('Unnamed: 0',axis=1),a15qam.drop('Unnamed: 0',axis=1))

        a16qam = a16qa.drop(outliers3)
        a16lam = a16la.drop(outliers3)
        a16mts1 = a16melts.drop('Unnamed: 0',axis=1) / (pix*pix)
        a16mtsm = a16mts1.drop(outliers3)
        a16crpm, a16lrpm, a16qrpm = calculate_combined_relative_probabilities(a16lam.drop('Unnamed: 0',axis=1),a16qam.drop('Unnamed: 0',axis=1))

        a17qam = a17qa.drop(outliers4)
        a17lam = a17la.drop(outliers4)
        a17mts1 = a17melts.drop('Unnamed: 0',axis=1) / (pix*pix)
        a17mtsm = a17mts1.drop(outliers4)
        a17crpm, a17lrpm, a17qrpm = calculate_combined_relative_probabilities(a17lam.drop('Unnamed: 0',axis=1),a17qam.drop('Unnamed: 0',axis=1))

        a14crp380m, a14lrp380m, a14qrp380m = change_iridum_age(a14mtsm,a14lam,a14qam.copy(),age='3.87903947')
        a15crp380m, a15lrp380m, a15qrp380m = change_iridum_age(a15mtsm,a15lam,a15qam.copy(),age='3.87903947')
        a16crp380m, a16lrp380m, a16qrp380m = change_iridum_age(a16mtsm,a16lam,a16qam.copy(),age='3.87903947')
        a17crp380m, a17lrp380m, a17qrp380m = change_iridum_age(a17mtsm,a17lam,a17qam.copy(),age='3.87903947')

        #find scale factor

        f14m = np.sum(hist14[0:53]) / np.sum(a14crp380m[-54:-41])
        f15m = np.sum(hist15[0:53]) / np.sum(a15crp380m[-54:-41])
        f16m = np.sum(hist16[0:53]) / np.sum(a16crp380m[-54:-41])
        f17m = np.sum(hist17[0:53]) / np.sum(a17crp380m[-54:-41])

        sa14lrp380m = a14lrp380m * f14m
        sa14qrp380m = a14qrp380m * f14m
        sa14crp380m = a14crp380m * f14m
        sa15lrp380m = a15lrp380m * f15m
        sa15qrp380m = a15qrp380m * f15m
        sa15crp380m = a15crp380m * f15m
        sa16lrp380m = a16lrp380m * f16m
        sa16qrp380m = a16qrp380m * f16m
        sa16crp380m = a16crp380m * f16m
        sa17lrp380m = a17lrp380m * f17m
        sa17qrp380m = a17qrp380m * f17m
        sa17crp380m = a17crp380m * f17m

        salrp380m = sa14lrp380m + sa15lrp380m + sa16lrp380m + sa17lrp380m
        saqrp380m = sa14qrp380m + sa15qrp380m + sa16qrp380m + sa17qrp380m

        r14lam = convert_npf_to_rpf(a14lam)
        r15lam = convert_npf_to_rpf(a15lam)
        r16lam = convert_npf_to_rpf(a16lam)
        r17lam = convert_npf_to_rpf(a17lam)

        r14crp380m, r14lrp380m, r14qrp380m = change_iridum_age(a14mtsm,r14lam,a14qa.copy(),age='3.87903947')
        r15crp380m, r15lrp380m, r15qrp380m = change_iridum_age(a15mtsm,r15lam,a15qa.copy(),age='3.87903947')
        r16crp380m, r16lrp380m, r16qrp380m = change_iridum_age(a16mtsm,r16lam,a16qa.copy(),age='3.87903947')
        r17crp380m, r17lrp380m, r17qrp380m = change_iridum_age(a17mtsm,r17lam,a17qa.copy(),age='3.87903947')

        f14irm = np.sum(hist14[0:53]) / np.sum(r14crp380m[-54:-42]) + ((0.3085113*r14crp380m[-55]))
        f15irm = np.sum(hist15[0:53]) / np.sum(r15crp380m[-54:-42]) + ((0.3085113*r15crp380m[-55]))
        f16irm = np.sum(hist16[0:53]) / np.sum(r16crp380m[-54:-42]) + ((0.3085113*r16crp380m[-55]))
        f17irm = np.sum(hist17[0:53]) / np.sum(r17crp380m[-54:-42]) + ((0.3085113*r17crp380m[-55]))

        ra14lrp380m = r14lrp380 * f14irm
        ra14qrp380m = r14qrp380 * f14irm
        ra14crp380m = r14crp380 * f14irm
        ra15lrp380m = r15lrp380 * f15irm
        ra15qrp380m = r15qrp380 * f15irm
        ra15crp380m = r15crp380 * f15irm
        ra16lrp380m = r16lrp380 * f16irm
        ra16qrp380m = r16qrp380 * f16irm
        ra16crp380m = r16crp380 * f16irm
        ra17lrp380m = r17lrp380 * f17irm
        ra17qrp380m = r17qrp380 * f17irm
        ra17crp380m = r17crp380 * f17irm

        ralrp380m = ra14lrp380m + ra15lrp380m + ra16lrp380m + ra17lrp380m
        raqrp380m = ra14qrp380m + ra15qrp380m + ra16qrp380m + ra17qrp380m

        #Plot Figure 14

        f1, a1 = plt.subplots(nrows=4,ncols=2,figsize=(15,15))

        a1[0,0].bar(bin_centers+offset,hist14,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 14 Melts')
        a1[0,0].bar(bin_centers2-offset,sa14lrp380m[:-1],yerr=np.sqrt(sa14lrp380m[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[0,0].bar(bin_centers2-offset,sa14qrp380m[:-1],yerr=np.sqrt(sa14qrp380m[:-1]),color='#af4d43',edgecolor='k',bottom=sa14lrp380m[:-1],width=0.07183406/3,label='sub-basin melt')
        a1[0,1].bar(bin_centers+offset,hist14,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 14 Melts')
        a1[0,1].bar(bin_centers2-offset,ra14lrp380m[:-1],yerr=np.sqrt(ra14lrp380m[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[0,1].bar(bin_centers2-offset,sa14qrp380m[:-1],yerr=np.sqrt(sa14qrp380m[:-1]),color='#af4d43',edgecolor='k',bottom=ra14lrp380m[:-1],width=0.07183406/3,label='sub-basin melt')

        a1[1,0].bar(bin_centers+offset,hist15,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 15 Melts')
        a1[1,0].bar(bin_centers2-offset,sa15lrp380m[:-1],yerr=np.sqrt(sa15lrp380m[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[1,0].bar(bin_centers2-offset,sa15qrp380m[:-1],yerr=np.sqrt(sa15qrp380m[:-1]),color='#af4d43',edgecolor='k',bottom=sa15lrp380m[:-1],width=0.07183406/3,label='sub-basin melt')
        a1[1,1].bar(bin_centers+offset,hist15,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 15 Melts')
        a1[1,1].bar(bin_centers2-offset,ra15lrp380m[:-1],yerr=np.sqrt(ra15lrp380m[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[1,1].bar(bin_centers2-offset,sa15qrp380m[:-1],yerr=np.sqrt(sa15qrp380m[:-1]),color='#af4d43',edgecolor='k',bottom=ra15lrp380m[:-1],width=0.07183406/3,label='sub-basin melt')

        a1[2,0].bar(bin_centers+offset,hist16,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 16 Melts')
        a1[2,0].bar(bin_centers2-offset,sa16lrp380m[:-1],yerr=np.sqrt(sa16lrp380m[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[2,0].bar(bin_centers2-offset,sa16qrp380m[:-1],yerr=np.sqrt(sa16qrp380m[:-1]),color='#af4d43',edgecolor='k',bottom=sa16lrp380m[:-1],width=0.07183406/3,label='sub-basin melt')
        a1[2,1].bar(bin_centers+offset,hist16,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 16 Melts')
        a1[2,1].bar(bin_centers2-offset,ra16lrp380m[:-1],yerr=np.sqrt(ra16lrp380m[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[2,1].bar(bin_centers2-offset,sa16qrp380m[:-1],yerr=np.sqrt(sa16qrp380m[:-1]),color='#af4d43',edgecolor='k',bottom=ra16lrp380m[:-1],width=0.07183406/3,label='sub-basin melt')

        a1[3,0].bar(bin_centers+offset,hist17,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 17 Melts')
        a1[3,0].bar(bin_centers2-offset,sa17lrp380m[:-1],yerr=np.sqrt(sa17lrp380m[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[3,0].bar(bin_centers2-offset,sa17qrp380m[:-1],yerr=np.sqrt(sa17qrp380m[:-1]),color='#af4d43',edgecolor='k',bottom=sa17lrp380m[:-1],width=0.07183406/3,label='sub-basin melt')
        a1[3,1].bar(bin_centers+offset,hist17,width=(0.07183406/3),color='goldenrod',edgecolor='k',label='Observed Apollo 17 Melts')
        a1[3,1].bar(bin_centers2-offset,ra17lrp380m[:-1],yerr=np.sqrt(ra17lrp380m[:-1]),color='b',edgecolor='k',bottom=0,width=0.07183406/3,label='Local melt')
        a1[3,1].bar(bin_centers2-offset,sa17qrp380m[:-1],yerr=np.sqrt(sa17qrp380m[:-1]),color='#af4d43',edgecolor='k',bottom=ra17lrp380m[:-1],width=0.07183406/3,label='sub-basin melt')


        a1[0,0].set_ylim(0,8)
        a1[0,1].set_ylim(0,8)
        a1[1,0].set_ylim(0,4)
        a1[1,1].set_ylim(0,4)
        a1[2,0].set_ylim(0,16)
        a1[2,1].set_ylim(0,16)
        a1[3,0].set_ylim(0,5)
        a1[3,1].set_ylim(0,5)

        a1[0,0].set_title('NPF')
        a1[0,1].set_title('RPF')

        a1[3,0].set_xlabel('Age (Ga)')
        a1[3,1].set_xlabel('Age (Ga)')

        for i in range(4):
            for j in range(2):
                a1[i,j].set_xlim(3.87,3.0)
                #a1[i,j].set_xlabel('Age (Ga)')
                a1[i,j].set_ylabel('Number of samples')
                a1[i,j].legend()
                
        # a1[0,0].text(3.8,7.25,'(a)',fontsize=16)
        # a1[0,1].text(3.8,7.25,'(b)',fontsize=16)
        # a1[1,0].text(3.8,2.75,'(c)',fontsize=16)
        # a1[1,1].text(3.8,2.75,'(d)',fontsize=16)
        # a1[2,0].text(3.8,15.25,'(e)',fontsize=16)
        # a1[2,1].text(3.8,15.25,'(f)',fontsize=16)
        # a1[3,0].text(3.8,2.75,'(g)',fontsize=16)
        # a1[3,1].text(3.8,2.75,'(h)',fontsize=16)
                
        matplotlib.rcParams.update({'font.size': 16})
        plt.tight_layout()
        plt.savefig('figure_14.pdf')

        return


if __name__ == '__main__':
    path = os.path.join(os.getcwd(),'results_data')

    a14la = pd.read_csv(os.path.join(path,'a14localages.csv'))
    a14qa = pd.read_csv(os.path.join(path,'a14qmcages.csv'))
    a14norms = pd.read_csv(os.path.join(path,'a14norms.csv'))
    a14melts = pd.read_csv(os.path.join(path,'a14melts.csv'))
    a14totmelts = pd.read_csv(os.path.join(path,'a14totmelts.csv'))
    a15la = pd.read_csv(os.path.join(path,'a15localages.csv'))
    a15qa = pd.read_csv(os.path.join(path,'a15qmcages.csv'))
    a15norms = pd.read_csv(os.path.join(path,'a15norms.csv'))
    a15melts = pd.read_csv(os.path.join(path,'a15melts.csv'))
    a15totmelts = pd.read_csv(os.path.join(path,'a15totmelts.csv'))
    a16la = pd.read_csv(os.path.join(path,'a16localages.csv'))
    a16qa = pd.read_csv(os.path.join(path,'a16qmcages.csv'))
    a16norms = pd.read_csv(os.path.join(path,'a16norms.csv'))
    a16melts = pd.read_csv(os.path.join(path,'a16melts.csv'))
    a16totmelts = pd.read_csv(os.path.join(path,'a16totmelts.csv'))
    a17la = pd.read_csv(os.path.join(path,'a17localages.csv'))
    a17qa = pd.read_csv(os.path.join(path,'a17qmcages.csv'))
    a17norms = pd.read_csv(os.path.join(path,'a17norms.csv'))
    a17melts = pd.read_csv(os.path.join(path,'a17melts.csv'))
    a17totmelts = pd.read_csv(os.path.join(path,'a17totmelts.csv'))
    combined_local = pd.read_csv(os.path.join(path,'combined_local.csv'))
    combined_qmc = pd.read_csv(os.path.join(path,'combined_qmc.csv'))

    #Michael et al. (2018) melt samples

    a14 = pd.read_csv(os.path.join(path,'a14meltages.csv'),header=None)
    a15 = pd.read_csv(os.path.join(path,'a15meltages.csv'),header=None)
    a16 = pd.read_csv(os.path.join(path,'a16meltages.csv'),header=None)
    a17 = pd.read_csv(os.path.join(path,'a17meltages.csv'),header=None)

    times = np.array([0.07183406, 0.14366813, 0.21550219, 0.28733626,
       0.35917032, 0.43100439, 0.50283845, 0.57467251, 0.64650658,
       0.71834064, 0.79017471, 0.86200877, 0.93384283, 1.0056769 ,
       1.07751096, 1.14934503, 1.22117909, 1.29301316, 1.36484722,
       1.43668128, 1.50851535, 1.58034941, 1.65218348, 1.72401754,
       1.7958516 , 1.86768567, 1.93951973, 2.0113538 , 2.08318786,
       2.15502193, 2.22685599, 2.29869005, 2.37052412, 2.44235818,
       2.51419225, 2.58602631, 2.65786037, 2.72969444, 2.8015285 ,
       2.87336257, 2.94519663, 3.0170307 , 3.08886476, 3.16069882,
       3.23253289, 3.30436695, 3.37620102, 3.44803508, 3.51986915,
       3.59170321, 3.66353727, 3.73537134, 3.8072054 , 3.87903947,
       3.95087353, 4.02270759, 4.09454166, 4.16637572, 4.23820979,
       4.31004385])

    newtimes = np.flip(times)

    width = (times[1] - times[0]) / 3
    offset = width / 2
    bin_centers = (times[:-1] + times[1:]) / 2
    bin_centers2 = (newtimes[:-1] + newtimes[1:]) / 2

    a14_0 = np.array([a14[0]])
    a15_0 = np.array([a15[0]])
    a16_0 = np.array([a16[0]])
    a17_0 = np.array([a17[0]])

    hist14, _ = np.histogram(a14_0,bins=times)
    hist15, _ = np.histogram(a15_0,bins=times)
    hist16, _ = np.histogram(a16_0,bins=times)
    hist17, _ = np.histogram(a17_0,bins=times)

    a14c = np.histogram(a14[0], bins=times)#, density=True)
    a15c = np.histogram(a15[0], bins=times)#, density=True)
    a16c = np.histogram(a16[0], bins=times)#, density=True)
    a17c = np.histogram(a17[0], bins=times)#, density=True)

    nage = 60
    interval = 628.0
    pix = 8.8e3
    intervalGa = ctem.craterproduction.T_from_scale(interval,'NPF_Moon')
    binsize = intervalGa / nage
    trueages = []
    xax = np.arange(nage)
    for a in xax:
        trueage = a * binsize
        trueages.append(trueage[0])
    order = trueages[::-1]

    a14crp, a14lrp, a14qrp = calculate_combined_relative_probabilities(a14la.drop('Unnamed: 0',axis=1),a14qa.drop('Unnamed: 0',axis=1))
    a15crp, a15lrp, a15qrp = calculate_combined_relative_probabilities(a15la.drop('Unnamed: 0',axis=1),a15qa.drop('Unnamed: 0',axis=1))
    a16crp, a16lrp, a16qrp = calculate_combined_relative_probabilities(a16la.drop('Unnamed: 0',axis=1),a16qa.drop('Unnamed: 0',axis=1))
    a17crp, a17lrp, a17qrp = calculate_combined_relative_probabilities(a17la.drop('Unnamed: 0',axis=1),a17qa.drop('Unnamed: 0',axis=1))

    
    generate_figure_1()
    generate_figure_2()
    generate_figure_7()
    generate_figure_8()
    generate_figure_9()
    generate_figure_10()
    generate_figure_11()
    generate_scaled_plots(outliers=False)
    generate_scaled_plots(outliers=True)

    ###Un-comment these lines to give verbose NPF-to-RPF and RPF-to-NPF bin boundary times. It will show "[3.9]" for ages outside the bounds of the RPF (>~4.02 Ga)

    # for time in times:
    #     print("RPF age", time, "corresponds to NPF age", convert(time))
    #     print("NPF age", time, "corresponds to RPF age", test(time))

    ###

    ###Un-comment these lines for statistical calculation of required simulation number

    # _, a14lrs = calculate_required_simulations(a14la.drop('Unnamed: 0',axis=1),confidence_level=0.95,margin_of_error=0.05)
    # _, a14qrs = calculate_required_simulations(a14qa.drop('Unnamed: 0',axis=1),confidence_level=0.95,margin_of_error=0.05)
    # _, a15lrs = calculate_required_simulations(a15la.drop('Unnamed: 0',axis=1),confidence_level=0.95,margin_of_error=0.05)
    # _, a15qrs = calculate_required_simulations(a15qa.drop('Unnamed: 0',axis=1),confidence_level=0.95,margin_of_error=0.05)
    # _, a16lrs = calculate_required_simulations(a16la.drop('Unnamed: 0',axis=1),confidence_level=0.95,margin_of_error=0.05)
    # _, a16qrs = calculate_required_simulations(a16qa.drop('Unnamed: 0',axis=1),confidence_level=0.95,margin_of_error=0.05)
    # _, a17lrs = calculate_required_simulations(a17la.drop('Unnamed: 0',axis=1),confidence_level=0.95,margin_of_error=0.05)
    # _, a17qrs = calculate_required_simulations(a17qa.drop('Unnamed: 0',axis=1),confidence_level=0.95,margin_of_error=0.05)

    # print('A14 Local: ', a14lrs)
    # print('A14 QMC: ', a14qrs)
    # print('A15 Local: ', a15lrs)
    # print('A15 QMC: ', a15qrs) 
    # print('A16 Local: ', a16lrs)
    # print('A16 QMC: ', a16qrs) 
    # print('A17 Local: ', a17lrs)
    # print('A17 QMC: ', a17qrs)  

    ###   
