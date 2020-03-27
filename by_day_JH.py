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

#metric, cases_over = 'confirmed', 5000
metric, cases_over = 'deaths', 100

day_start_cases = 100

df = pd.read_csv(fp.format(metric))
df = df.groupby('Country/Region').sum()
df.drop(['Lat', 'Long'], axis=1, inplace=True)
df = df[df.iloc[:, -1] > cases_over]
df = df.T
df.index = pd.to_datetime(df.index)
df.to_csv('AmendedData\\{}ByCountry.csv'.format(metric.title()))
#df.plot()

out_fp = '{}ByCountryFrom{}thCaseDay'.format(metric.title(),
                                             day_start_cases)

from_day_x = {}
#country_list = ['United Kingdom', 'China']
country_list = df.columns
for country in country_list:
    _ = df[country][df[country] > day_start_cases]
    _.reset_index(drop=True, inplace=True)
    from_day_x[country] = _

longest = max([len(from_day_x[k]) for k in from_day_x.keys()])

from_case_start = pd.DataFrame(columns=country_list, index=range(longest))
for country in country_list:
    from_case_start[country] = from_day_x[country]
from_case_start.to_csv('AmendedData\\{}.csv'.format(out_fp))

# Get the x and y dims for the small mults
country_count = len(country_list)
y = math.floor(country_count ** 0.5)
while country_count // y != country_count / y:
    y -= 1
x = country_count // y
if y == 1:
    y = math.floor(country_count ** 0.5)
    x = (country_count // y) + 1
print(x, y)


color = plt.cm.Dark2(np.linspace(0,1,country_count))
# rainbow colormap was prettier but less clear
figs, axs = plt.subplots(y, x, sharey=True, figsize=(x*4, y*4))
for (i, ax), c in zip(enumerate(axs.flatten()), color):
    country = country_list[i]
    _ = from_case_start.drop(country, axis=1)
    ax.plot(_, color='grey', alpha=0.3)
    ax.plot(from_case_start[country], color=c, linewidth=3)
    ax.set_title(country)
    ax.set_yscale('log')
plt.suptitle('{} for top {} countries'.format(metric.title(), country_count),
             fontsize=30)
plt.xlabel('Days since 100 mark passed')
plt.savefig('Outputs\\{}.png'.format(out_fp))