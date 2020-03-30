# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 01:10:00 2020
@author: setat
"""

import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import matplotlib.ticker as mtick
import math
import numpy as np

fp = 'https://github.com/CSSEGISandData/COVID-19/raw/master/\
csse_covid_19_data/csse_covid_19_time_series/\
time_series_covid19_{}_global.csv'

cases_over = {'deaths': 100, 'confirmed': 5000}
day_start_cases = 100
# metric, cases_over = 'confirmed', 5000
# metric, cases_over = 'deaths', 100


def munge_csv(fp, metric, case_floor, countries_specified=None):
    df = pd.read_csv(fp.format(metric))
    df = df.groupby('Country/Region').sum()
    df.drop(['Lat', 'Long'], axis=1, inplace=True)
    if countries_specified is not None:
        df = df.filter(country_list, axis=0)
    else:
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
    empty_count = x*y - el_count
    if empty_count > 0:
        for ax in axs.flatten()[-empty_count:]:
            ax.set_visible(False)
    plt.suptitle('{} for top {} countries'.format(metric.title(), el_count),
                 fontsize=30)
    plt.xlabel('Days since {} mark passed'.format(day_start_cases))
    plt.savefig('Outputs\\{}.png'.format(out_fp))
    plt.show()


def plot_annotated(data, el_count, metric):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    data.plot(ax=ax, colormap='Dark2')
    labels = {}
    for country in data.columns:
        _ = data[country].dropna()
        valid_days_from_zero = len(_)-1
        labels[country] = (valid_days_from_zero, _.loc[valid_days_from_zero])
    top_counts = from_case_start.sum().sort_values(ascending=False)
    uk_position = list(top_counts.index).index('United Kingdom')
    only_label = top_counts.index[:uk_position+1]
    for country in only_label:
        ax.annotate(country, labels[country])
    plt.title('{} for top {} countries'.format(metric.title(), el_count),
              fontsize=30)
    plt.xlabel('Days since {} mark passed'.format(day_start_cases))
    plt.savefig('Outputs\\{}_annotated.png'.format(out_fp))
    plt.show()


# Get country list from deaths, plot deaths
metric = 'deaths'
out_fp = '{}ByCountryFrom{}thCaseDay'.format(metric.title(),
                                             day_start_cases)
df = munge_csv(fp, metric, cases_over[metric])
country_list = df.columns
country_count = len(country_list)
color = plt.cm.Dark2(np.linspace(0, 1, country_count))
dct = count_from_day_x(df, country_list)
from_case_start = counts_to_df(dct, country_list, out_fp)
(x, y) = small_mult_size(country_count)
plot_mults(country_list, from_case_start, country_count, x, y, metric)
plot_annotated(from_case_start, country_count, metric)

# Plot just UK
country = "United Kingdom"
just_uk = [i for i in from_case_start[country] if i > 0]
base = np.linspace(0, len(just_uk))
double_in = lambda k: [just_uk[0] * (2 ** (i/k)) for i in base]
fig, ax = plt.subplots(1, 1)
ax.plot(base, double_in(3), '--', label="2x every three days", c='0.1')
ax.plot(base, double_in(7), '--', label="2x every week", c='0.4')
ax.plot(base, double_in(14), '--', label="2x every fortnight", c='0.7')
ax.plot(just_uk, linewidth=3, label=country, c='#ee2a24')
ax.set_title('{} in {}'.format(metric.title(), country))
plt.xlabel('Days since {} {} confirmed'.format(day_start_cases, metric))
plt.legend()
plt.savefig('Outputs\\{}In{}From{}thCaseDay.png'.format(metric.title(),
            country.replace(' ', ''), day_start_cases))

# Plot diff-in-diff
figs, axs = plt.subplots(3, 1, figsize=(8,12))
_ = df['United Kingdom'][df['United Kingdom'] > 100]
ttls = ['{} per day', 'New {} per day', '{} diff-in-new per day']
for ax, ttl in zip(axs.flatten(), ttls):
    _.plot(ax=ax)
    ax.set_title(ttl.format(metric))
    _ = _.diff()
plt.suptitle('UK daily changes', fontsize=30, y=0.95)
plt.savefig('Outputs\\UKDailyChanges.png')

# Use previous country list, plot confirmed
metric = 'confirmed'
out_fp = '{}ByCountryFrom{}thCaseDay'.format(metric.title(),
                                             day_start_cases)
df = munge_csv(fp, metric, cases_over[metric],
               countries_specified=country_list)
dct = count_from_day_x(df, country_list)
from_case_start = counts_to_df(dct, country_list, out_fp)
(x, y) = small_mult_size(country_count)
plot_mults(country_list, from_case_start, country_count, x, y, metric)
plot_annotated(from_case_start, country_count, metric)

# UK rate of change per day
uk_rate_change = df['United Kingdom'] / df['United Kingdom'].shift(1)
uk_rate_change = uk_rate_change - 1
ax_ukrate = uk_rate_change.plot()
ax_ukrate.set_title('UK {} rate change by day'.format(metric))
ax_ukrate.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.show()

# UK new cases by day
uk_new_confirmed = df['United Kingdom'].diff()
uk_new_confirmed.plot(style='.-', figsize=(10, 6))
plt.title('UK new {} cases by day'.format(metric))
plt.show()