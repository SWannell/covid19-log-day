# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 01:10:00 2020

@author: setat
"""

import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import math
import numpy as np

fp = 'https://github.com/CSSEGISandData/COVID-19/raw/master/\
csse_covid_19_data/csse_covid_19_time_series/\
time_series_covid19_{}_global.csv'

cases_over = {'deaths': 100, 'confirmed': 5000}
day_start_cases = 100
# metric, cases_over = 'confirmed', 5000
# metric, cases_over = 'deaths', 100


def munge_csv(fp, metric, case_floor):
    df = pd.read_csv(fp.format(metric))
    df = df.groupby('Country/Region').sum()
    df.drop(['Lat', 'Long'], axis=1, inplace=True)
    df = df[df.iloc[:, -1] > case_floor]
    df = df.T
    df.index = pd.to_datetime(df.index)
    df.to_csv('AmendedData\\{}ByCountry.csv'.format(metric.title()))
    return df


def count_from_day_x(data, list_of_countries):
    from_day_x = {}
    for country in list_of_countries:
        _ = data[country][data[country] > day_start_cases]
        _.reset_index(drop=True, inplace=True)
        from_day_x[country] = _
    return from_day_x


def counts_to_df(count_dict, country_list, out_fp):
    longest = max([len(count_dict[k]) for k in count_dict.keys()])
    from_case_start = pd.DataFrame(columns=country_list, index=range(longest))
    for country in country_list:
        from_case_start[country] = count_dict[country]
    from_case_start.to_csv('AmendedData\\{}.csv'.format(out_fp))
    return from_case_start


def small_mult_size(el_count):
    # Get the x and y dims for the small mults
    y = math.floor(el_count ** 0.5)
    while el_count // y != el_count / y:
        y -= 1
    x = el_count // y
    if y == 1:
        y = math.floor(el_count ** 0.5)
        x = (el_count // y) + 1
    return (x, y)


def plot_mults(country_list, data, el_count, x, y, metric):
    color = plt.cm.Dark2(np.linspace(0, 1, el_count))
    # rainbow colormap was prettier but less clear
    figs, axs = plt.subplots(y, x, sharey=True, figsize=(x*4, y*4))
    for (i, ax), c in zip(enumerate(axs.flatten()), color):
        country = country_list[i]
        _ = data.drop(country, axis=1)
        ax.plot(_, color='grey', alpha=0.3)
        ax.plot(data[country], color=c, linewidth=3)
        ax.set_title(country)
        ax.set_yscale('log')
    plt.suptitle('{} for top {} countries'.format(metric.title(), el_count),
                 fontsize=30)
    plt.xlabel('Days since 100 mark passed')
    plt.savefig('Outputs\\{}.png'.format(out_fp))


# Get country list from deaths, plot deaths
metric = 'deaths'
out_fp = '{}ByCountryFrom{}thCaseDay'.format(metric.title(),
                                             day_start_cases)
df = munge_csv(fp, metric, cases_over[metric])
country_list = df.columns
country_count = len(country_list)
dct = count_from_day_x(df, country_list)
from_case_start = counts_to_df(dct, country_list, out_fp)
(x, y) = small_mult_size(country_count)
plot_mults(country_list, from_case_start, country_count, x, y, metric)

# Use previous country list, plot confirmed
metric = 'confirmed'
out_fp = '{}ByCountryFrom{}thCaseDay'.format(metric.title(),
                                             day_start_cases)
df = munge_csv(fp, metric, cases_over[metric])
dct = count_from_day_x(df, country_list)
from_case_start = counts_to_df(dct, country_list, out_fp)
(x, y) = small_mult_size(country_count)
plot_mults(country_list, from_case_start, country_count, x, y, metric)


base = np.linspace(0, len(from_case_start))
double_in = lambda k: [100 * (2 ** (i/k)) for i in base]
plt.plot(base, double_in(3), label="Three days")
plt.plot(base, double_in(7), label="Week")
plt.plot(base, double_in(14), label="Two weeks")
plt.yscale("log")
plt.legend()

# Should get whole df, then filter on deaths, then get country list. Then
# store country list, and use it for both filters and plots