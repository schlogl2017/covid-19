#!/usr/bin/env python
# coding: utf-8


import sys
from datetime import datetime, timedelta
from difflib import get_close_matches
from collections import Counter
import numpy as np 
import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dateutil.parser import parse
import plotly.express as px
import plotly.graph_objs as go
from utils_covid import fix_country_names, get_names


# some colors and dates to use in graphics
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801'
today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)
ontem = yesterday.strftime('%d/%m/%y')



# #### Dwonloading the data
# The world Covid-19 data is downloaded from **CSSEGISandData** repository from Jhon Hopkins University.
# The source is located at: https://github.com/CSSEGISandData
url1 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url2 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url3 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

# global data
df_confirmed = pd.read_csv(url1)
df_fatalities = pd.read_csv(url2)
df_recoveries = pd.read_csv(url3)


print(df_confirmed.info())
print(df_confirmed.shape, '\n')


print(df_fatalities.info())
print(df_fatalities.shape, '\n')


print(df_recoveries.info())
print(df_recoveries.shape, '\n')


# cheking  if all columns are have the same name
dfs = [df_confirmed, df_fatalities, df_recoveries]
print('Checking if the columns are equal')
same = all([len(dfs[0].columns.intersection(df.columns)) == dfs[0].shape[1] for df in dfs])
print(same, '\n')


# tiding and munging
def get_large_to_long_format_JHU_data(df, col_values='Confirmed'):
    dates = df.columns[4:]
    df = df.melt(id_vars=['Province/State',
                          'Country/Region', 
                          'Lat', 
                          'Long'],
                 value_vars=dates, 
                 var_name='Date', 
                 value_name=col_values)
    return df


confirmed = get_large_to_long_format_JHU_data(df_confirmed, col_values='Confirmed')
deaths = get_large_to_long_format_JHU_data(df_fatalities, col_values='Deaths')
recovered = get_large_to_long_format_JHU_data(df_recoveries, col_values='Recovered')


print(confirmed.info())
print(confirmed.shape, '\n')


print(deaths.info())
print(deaths.shape, '\n')


print(recovered.info())
print(recovered.shape, '\n'):
# dropping canada recovered rows
recovered = recovered[recovered['Country/Region']!='Canada']

print(confirmed.shape, deaths.shape, recovered.shape)


def merge_JHU_data_frames(df1, df2, df3):
    full_df = pd.merge(left=df1, right=df2, how='left',
                      on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])
    full_df = pd.merge(left=full_df, right=df3, how='left',
                      on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])
    return full_df


# Mergin the 3 data frames
world_covid = merge_JHU_data_frames(confirmed, deaths, recovered)
print(world_covid.info())
# checking null values
world_covid.isnull().sum()


# # fill na with 0
# full_table['Recovered'] = full_table['Recovered'].fillna(0).astype('int')
# full_table.isnull().sum()
def clean_data(df):
    # Convert to proper date format
    df['Date'] = pd.to_datetime(df['Date'])
    # fiil the na data
    df['Recovered'] = df['Recovered'].fillna(0).astype('int')
    
    if 'Demised' in df.columns:
        df.rename(columns={'Demised':'Deaths'}, inplace=True)

    if 'Country/Region' in df.columns:
        df.rename(columns={'Country/Region':'country'}, inplace=True)
    
    if 'Province/State' in df.columns:
        df.rename(columns={'Province/State':'province'}, inplace=True)
        
    if 'Last Update' in df.columns:
        df.rename(columns={'Last Update':'datetime'}, inplace=True)
        
    if 'Suspected' in df.columns:
        df = df.drop(columns='Suspected', axis=1, inplace=True)
    
    # all column names as lower case
    df.columns = map(str.lower, df.columns)

    return df


clean_data(world_covid)


# In[30]:


world_covid.isnull().sum()


# In[31]:


def clean_data(df):
    # Convert to proper date format
    df['Date'] = pd.to_datetime(df['Date'])
    # fiil the na data
    df['Recovered'] = df['Recovered'].fillna(0).astype('int')
    if 'Demised' in df.columns:
        df.rename(columns={'Demised':'Deaths'}, inplace=True)
    if 'Country/Region' in df.columns:
        df.rename(columns={'Country/Region':'country'}, inplace=True)
    if 'Province/State' in df.columns:
        df.rename(columns={'Province/State':'province'}, inplace=True)
    if 'Last Update' in df.columns:
        df.rename(columns={'Last Update':'datetime'}, inplace=True)
    if 'Suspected' in df.columns:
        df = df.drop(columns='Suspected', axis=1, inplace=True)
    # all column names as lower case
    df.columns = map(str.lower, df.columns)
    return df


# In[32]:


get_names(world_covid, 'country')


# In[34]:


# removing canada's recovered values
world_covid = world_covid[world_covid['province'].str.contains('recovered') != True]

# removing county wise data to avoid double counting
world_covid = world_covid[world_covid['province'].str.contains(',') != True]


# In[35]:


# Active Case = confirmed - deaths - recovered
world_covid['active'] = world_covid['confirmed'] - world_covid['deaths'] - world_covid['recovered']


# In[36]:


world_covid.info()


# In[37]:


world_covid.isnull().sum()


# In[38]:


world_covid.head()


# In[39]:


# checking where the lat/long values are null
world_covid[world_covid['lat'] == world_covid['lat'].isnull()]


# In[40]:


# filling missing values 
# fill missing province/state value with ''
world_covid[['province']] = world_covid[['province']].fillna('')

# random rows
world_covid.sample(6)


# In[41]:


world_covid.info()


# In[42]:


world_covid.loc[world_covid['province'] == 'Hubei']


# In[43]:


# looks like there are some confirmed data in Hubei province with bada data
world_covid[(world_covid['date'] == '2/12/20') & (world_covid['province'] == 'Hubei')]


# In[44]:


def change_val(df, day, ref_col, col_date, val_col, data_entry):
    for key, val in data_entry.items():
        df.loc[(world_covid[col_date] == day) & (df[ref_col] == key), val_col] = val


# In[45]:


# new values
feb_12_conf = {'Hubei' : 34874}


# In[46]:


# changing values
change_val(world_covid, '2/12/20', 'province', 'date', 'confirmed', feb_12_conf)


# In[47]:


world_covid[(world_covid['date']=='2/12/20') & (world_covid['province']=='Hubei')]


# In[49]:


# checking for the data containing ships with COVID-19 reported cases
world_covid['province'].str.contains('Grand Princess').sum()


# In[50]:


# ship rows containing ships with COVID-19 reported cases
ship_rows = world_covid['province'].str.contains('Grand Princess') |             world_covid['province'].str.contains('Diamond Princess') |             world_covid['province'].str.contains('Diamond Princess') |             world_covid['province'].str.contains('MS Zaandam')

to_drop = world_covid[ship_rows].index

# dropping this lines
world_covid.drop(to_drop, inplace=True)


# In[51]:


# checking the results of dropping
world_covid['province'].str.contains('Grand Princess').sum()


# In[52]:


# Grouped by day, country
world_covid_grouped = world_covid.groupby(['date', 
                                           'country'])[['confirmed',
                                                        'deaths', 
                                                        'recovered', 
                                                        'active']].sum().reset_index()


# In[53]:


world_covid_grouped


# In[55]:


tmp = world_covid_grouped.groupby('date')[['recovered', 'deaths', 'active']].sum().reset_index()
tmp = tmp.melt(id_vars="date", value_vars=['recovered', 'deaths', 'active'],
                 var_name='case', value_name='count')
tmp.head()


# In[61]:


def plot_area(df, cl_x, col_y, col_color, title, date, path, is_save=False):
    fig = go.Figure()
    fig = px.area(tmp, 
                  x=cl_x, 
                  y=col_y, 
                  color=col_color, 
                  height=600, 
                  width=700,
                  title=f'<b>{title} ({date})<b>', 
                  color_discrete_sequence = [rec, dth, act])
    if is_save:
        fig.write_html(f'{path}/{title}_{date}.html', auto_open=False)
    fig.show()


# In[63]:


plot_area(tmp, 'date', 'count', 'case', 'World covid-19 cases', date=yesterday, path="",is_save=False)


# In[62]:


plot_area(tmp, 'date', 'count', 'case', 'World covid-19 cases', date=yesterday, path="Graficos",is_save=True)


# In[64]:


# new cases 
temp = world_covid_grouped.groupby(['country', 'date', ])[['confirmed', 'deaths', 'recovered']]
temp = temp.sum().diff().reset_index()


# In[65]:


temp


# In[66]:


# mask
f = temp['country'] != temp['country'].shift(1)


# In[67]:


temp.loc[f, 'confirmed'] = np.nan
temp.loc[f, 'deaths'] = np.nan
temp.loc[f, 'recovered'] = np.nan


# In[68]:


temp.head()


# In[69]:


temp.info()


# In[70]:


temp.isnull().sum()


# In[71]:


# renaming columns
temp.columns = ['country', 'date', 'new_cases', 'new_deaths', 'new_recovered']


# In[72]:


temp.head()


# In[73]:


# merging new values
world_covid_grouped = pd.merge(world_covid_grouped, temp, on=['country', 'date'])


# In[74]:


world_covid_grouped.head()


# In[75]:


world_covid_grouped.info()


# In[76]:


# filling na with 0
world_covid_grouped = world_covid_grouped.fillna(0)


# In[77]:


world_covid_grouped.info()


# In[78]:


world_covid_grouped.isnull().sum()


# In[79]:


# fixing data types
cols = ['new_cases', 'new_deaths', 'new_recovered']
world_covid_grouped[cols] = world_covid_grouped[cols].astype('int')


# In[80]:


world_covid_grouped.tail()


# In[82]:


world_covid_grouped.info()


# In[83]:


world_covid_grouped.isnull().sum()


# In[148]:


def world_choropleth(df, 
                     location_col, 
                     color_col, 
                     hover_col, 
                     anim_col, 
                     date, 
                     name,
                     path, 
                     color_scale='matter',
                     locationmode='country names', 
                     is_save=False):
    fig = px.choropleth(df, 
                    locations=location_col, 
                    color=np.log(world_covid_grouped[color_col]),
                    locationmode='country names', 
                    hover_name=hover_col,
                    animation_frame=world_covid_grouped[anim_col].dt.strftime('%Y-%m-%d'),
                    title=name,
                    color_continuous_scale='matter')
    fig.update(layout_coloraxis_showscale=True)
    if is_save:
        fig.write_html(f'{path}/{name}_{date}.html', auto_open=False)
    fig.show()


# In[150]:


world_choropleth(world_covid_grouped, 
                 'country', 
                 'confirmed', 
                 'country',
                 'date',
                 yesterday,
                 f'<b>Cases over time<b> (log scale)',
                 'Graficos')


# In[151]:


# fig = px.choropleth(world_covid_grouped, 
#                     locations="country", 
#                     color=np.log(world_covid_grouped["confirmed"]),
#                     locationmode='country names', 
#                     hover_name="country",
#                     animation_frame=world_covid_grouped["date"].dt.strftime('%Y-%m-%d'),
#                     title='<b>Cases over time<b> (log scale)',
#                     color_continuous_scale='matter')
# fig.update(layout_coloraxis_showscale=True)
# fig.show()


# In[156]:


def plot_map(df, col_value,col_locations, date, pallet):
    df = df[df[col_value] > 0]
    fig = px.choropleth(df, 
                        locations=col_locations, 
                        locationmode='country names',
                        color=col_value, 
                        hover_name=col_locations, 
                        title=f'<b>Covid country wise {col_value} cases ( {date })<b>', 
                        hover_data=[col_value], 
                        color_continuous_scale=pallet,
                        fitbounds='geojson',
                        projection='natural earth',
                       )
    fig.show()


# In[158]:


pal = px.colors.sequential.Viridis
plot_map(country_wise, 'confirmed', 'country', yesterday, pal)


# In[84]:


# day wise table
day_wise = world_covid_grouped.groupby('date')[['confirmed',
                                                'deaths', 
                                                'recovered', 
                                                'active', 
                                                'new_cases', 
                                                'new_deaths', 
                                                'new_recovered']].sum().reset_index()


# In[85]:


day_wise.head()


# In[86]:


day_wise.info()


# In[98]:


day_wise.isnull().sum()


# In[88]:


# number cases per 100 cases
day_wise['deaths_100_cases'] = round((day_wise['deaths']/day_wise['confirmed']) * 100, 2)
day_wise['recovered_100_cases'] = round((day_wise['recovered']/day_wise['confirmed']) * 100, 2)
day_wise['deaths_100_recovered'] = round((day_wise['deaths']/day_wise['recovered']) * 100, 2)


# In[89]:


day_wise.head()


# In[90]:


day_wise.info()


# In[91]:


day_wise.isnull().sum()


# In[92]:


#  get the last day of the data with the actual infection data
day_wise[['date', 'confirmed', 'deaths', 'recovered', 'active']].tail(1)


# In[93]:


tmp = world_covid_grouped.groupby('date')[['confirmed',
                                                'deaths', 
                                                'recovered', 
                                                'active', 
                                                'new_cases', 
                                                'new_deaths', 
                                                'new_recovered']].sum().reset_index()


# In[95]:


day_wise['new_cases'].min()


# In[96]:


day_wise['new_deaths'].min()


# In[97]:


day_wise['new_recovered'].min()


# In[99]:


func = lambda x: 0 if x < 0 else x


# In[100]:


day_wise['new_cases'] = day_wise['new_cases'].apply(func)
day_wise['new_deaths'] = day_wise['new_deaths'].apply(func)
day_wise['new_recovered'] = day_wise['new_recovered'].apply(func)
day_wise['deaths_100_cases'] = day_wise['deaths_100_cases'].apply(func)
day_wise['recovered_100_cases'] = day_wise['recovered_100_cases'].apply(func)
day_wise['deaths_100_recovered'] = day_wise['deaths_100_recovered'].apply(func)


# In[113]:


def plot_daywise(df, xcol, ycol, title, title_x, title_y, color, date, path, is_save=False):
    fig = px.bar(df, x=xcol, y=ycol, width=700, color_discrete_sequence=[color])
    fig.update_layout(title=f'<b>{title} ({date})<b>', xaxis_title=title_x, yaxis_title=title_y)
    if is_save:
        fig.write_html(f'{path}/{title}_{date}.html', auto_open=False)
    fig.show()


# In[117]:


plot_daywise(day_wise,
             'date',
             'active', 
             'World active Covid-19 cases', 
             'Date', 
             'Active cases',
             'red',
             yesterday, 
             'Graficos', 
             True)


# In[115]:


def plot_daywise_line(df, xcol, ycol, title, title_x, title_y, color, date, path, is_save=False):
    fig = px.line(df, x=xcol, y=ycol, width=700, color_discrete_sequence=[color])
    fig.update_layout(title=f'<b>{title} ({date})<b>', xaxis_title=title_x, yaxis_title=title_y)
    if is_save:
        fig.write_html(f'{path}/{title}_{date}.html', auto_open=False)
    fig.show()


# In[118]:


plot_daywise_line(day_wise,
             'date',
             'confirmed', 
             'World confirmed Covid cases', 
             'Date', 
             'Active cases',
             'red',
             yesterday, 
             'Graficos', 
             True)


# In[123]:


plot_daywise_line(day_wise,
             'date',
             'new_recovered', 
             'World new recovered Covid cases', 
             'Date', 
             'Active cases',
             'green',
             yesterday, 
             'Graficos', 
             )


# In[122]:


plot_daywise_line(day_wise,
             'date',
             'new_cases', 
             'World new Covid cases', 
             'Date', 
             'Active cases',
             'red',
             yesterday, 
             'Graficos', 
             )


# In[137]:


plot_daywise_line(day_wise,
             'date',
             'new_deaths', 
             'World new deaths by Covid', 
             'Date', 
             'Deaths',
             'black',
             yesterday, 
             'Graficos', 
             )


# In[139]:


plot_daywise_line(day_wise,
             'date',
             'deaths_100_cases', 
             'World deaths by Covid', 
             'Date', 
             'Deaths (by 100 cases)',
             'black',
             yesterday, 
             'Graficos', 
             )


# In[125]:


temp1 = day_wise[['date','confirmed', 'deaths', 'recovered', 'active']].tail(1)
temp1 = temp1.melt(id_vars="date", value_vars=['confirmed','active', 'deaths', 'recovered'])
temp1


# In[143]:


def bar_plot(df, x_col, y_col, title, date, path, is_save=False):
    fig = px.bar(temp1.sort_values(by='value', ascending=False), 
             x=x_col, 
             y=y_col,
             hover_data=[x_col], 
             color=x_col, 
             text=y_col, 
             height=400)
    fig.update_layout(title_text=f'<b>{title} ( {date} )<b>')
    if is_save:
        fig.write_html(f'{path}/{title}_{data}.html', auto_open=False)
    fig.show()


# In[144]:


bar_plot(temp1, 'variable', 'value', 'World Covid-19 cases', yesterday, 'Graficos', is_save=False)


# In[ ]:


# save as .csv file
# day_wise.to_csv('day_wise.csv', index=False)


# In[159]:


# lastest cases country wise
# getting latest values
country_wise = world_covid_grouped[world_covid_grouped['date'] == max(world_covid_grouped['date'])].reset_index(drop=True)


# In[160]:


country_wise.head()


# In[161]:


country_wise.info()


# In[162]:


country_wise.isnull().sum()


# In[163]:


country_wise.shape


# In[164]:


# group by country
country_wise = country_wise.groupby('country')[['confirmed', 
                                               'deaths', 
                                               'recovered', 
                                               'active',
                                               'new_cases', 
                                               'new_deaths', 
                                               'new_recovered']].sum().reset_index()
country_wise.shape


# In[165]:


country_wise.head()


# In[166]:


country_wise.info()


# In[167]:


country_wise.isnull().sum()


# In[171]:


most_active_case = country_wise.groupby('country')[['active']].sum().reset_index()


# In[172]:


idx = most_active_case['active'].argmax()
most_active_case.iloc[idx]


# In[173]:


most_active_case.loc[most_active_case[['active']].idxmax()]


# In[174]:


# per 100 cases
country_wise['deaths_100_cases'] = round((country_wise['deaths']/country_wise['confirmed']) * 100, 2)
country_wise['recovered_100_cases'] = round((country_wise['recovered']/country_wise['confirmed']) * 100, 2)
country_wise['deaths_100_recovered'] = round((country_wise['deaths']/country_wise['recovered']) * 100, 2)

cols = ['deaths_100_cases', 'recovered_100_cases', 'deaths_100_recovered']
country_wise[cols] = country_wise[cols].fillna(0)


# In[175]:


country_wise


# In[176]:


country_wise.info()


# In[177]:


country_wise.isnull().sum()


# In[178]:


# 1 week increase and % change
confirmed_today = world_covid_grouped[world_covid_grouped['date'] == max(world_covid_grouped['date'])].reset_index(drop=True)[['country',
                                                                                                                               'confirmed']]


# In[179]:


confirmed_today


# In[180]:


# data from today - the last week data
last_week = world_covid_grouped[world_covid_grouped['date'] == max(world_covid_grouped['date'])-timedelta(days=7)].reset_index(drop=True)[['country',
                                                                                                                                           'confirmed']]


# In[181]:


last_week


# In[182]:


temp2 = pd.merge(confirmed_today, last_week, on='country', suffixes=('_today', '_last_week'))
temp2['1_week_change'] = temp2['confirmed_today'] - temp2['confirmed_last_week']
temp2 = temp2[['country', 'confirmed_last_week', '1_week_change']]


# In[183]:


temp2


# In[184]:


data = temp2.sort_values('confirmed_last_week', ascending=False).head(10)
data


# In[185]:


x = data["country"]
y1 = data["confirmed_last_week"]
y2 = data["1_week_change"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',
                    name='confirmed_last_week'))
fig.add_trace(go.Scatter(x=x, y=y2,
                    mode='lines+markers',
                    name='1_week_change'))
fig.update_layout(title_text=f'<b>Diferença semanal de casos confirmados de Covid-19 a nivel mundial: {ontem}<b>')

fig.show()


# In[186]:


country_wise = pd.merge(country_wise, temp2, on='country')
country_wise['1_week_%_increase'] = round(country_wise['1_week_change']/country_wise['confirmed_last_week'] * 100, 2)
country_wise.head()


# In[187]:


country_wise.shape


# In[188]:


country_wise.info()


# In[189]:


country_wise.isnull().sum()


# In[ ]:


# save as .csv file
# country_wise.to_csv('country_wise_latest.csv', index=False)


# In[190]:


china_province_wise = world_covid_grouped[world_covid_grouped['country'] == 'China']


# In[191]:


china_province_wise


# In[192]:


brasil_province_wise = world_covid_grouped[world_covid_grouped['country'] == 'Brazil']


# In[193]:


brasil_province_wise


# In[194]:


def plot_hbar(df, xcol, ycol, n, date, figtitle, xtitle, ytitle, hover_data=[], is_save=False):
    fig = px.bar(df.sort_values(xcol).tail(n),
                 x=xcol, 
                 y=ycol, 
                 color=ycol,  
                 text=xcol, 
                 orientation='h', 
                 width=700, 
                 hover_data=hover_data,
                 color_discrete_sequence = px.colors.qualitative.Dark24)
    fig.update_layout(title=f'{figtitle} ( {date} )', 
                      xaxis_title=xtitle, 
                      yaxis_title=ytitle, 
                      yaxis_categoryorder = 'total ascending',
                      uniformtext_minsize=8, 
                      uniformtext_mode='hide')
    if is_save:
        fig.write_html(f'{figtitle}_{date}.html', auto_open=False)    
    fig.show()


# In[195]:


plot_hbar(country_wise, 
          'confirmed', 
          'country', 
          15, 
          yesterday, 
          '<b>Top 15 countries with most Covid-19 cases<b>',  
          'Cases', 
          'Country')


# In[198]:


plot_hbar(country_wise, 
          'active', 
          'country', 
          15, 
          yesterday, 
          '<b>Top 15 countries with most active Covid-19 cases<b>',  
          'Cases', 
          'Country',
          hover_data=['active']
         )


# In[199]:


plot_hbar(country_wise, 
          'deaths', 
          'country', 
          15, 
          yesterday, 
          '<b>Top 15 countries with most number of fatalities by Covid-19 cases<b>',  
          'Cases', 
          'Country',
          hover_data=['deaths']
         )


# In[200]:


plot_hbar(country_wise, 
          'new_deaths', 
          'country', 
          15, 
          yesterday, 
          '<b>Top 15 countries with most new deaths by Covid-19 cases<b>',  
          'Cases', 
          'Country',
          hover_data=['new_deaths']
         )


# In[201]:


plot_hbar(country_wise, 
          'new_cases', 
          'country', 
          15, 
          yesterday, 
          '<b>Top 15 countries with most new_cases of Covid-19 cases<b>',  
          'Cases', 
          'Country',
          hover_data=['new_cases']
         )


# In[202]:


def plot_stacked(df, xcol, ycol, color_col, title, pallet):
    fig = px.bar(df, 
                 x=xcol, 
                 y=ycol, 
                 color=color_col, 
                 height=600, 
                 title=title, 
                 color_discrete_sequence = pallet)
    fig.update_layout(showlegend=True)
    fig.show()


# In[203]:


pal = px.colors.cyclical.mygbm
plot_stacked(world_covid_grouped, 'date', 'confirmed', 'country', 'Confirmed', pal)


# In[204]:


def plot_line(col):
    fig = px.line(df, 
                 x=xcol, 
                 y=ycol, 
                 color=color_col, 
                 height=600, 
                 title=title, 
                 color_discrete_sequence = pallet)
    fig.update_layout(showlegend=True)
    fig.show()


# In[206]:


pal = px.colors.cyclical.Edge
plot_stacked(world_covid_grouped, 'date', 'confirmed', 'country', 'Confirmed', pal)


# In[207]:


tp = pd.merge(world_covid_grouped[['date', 'country', 'confirmed', 'deaths']], 
                day_wise[['date', 'confirmed', 'deaths']], on='date')
tp['%confirmed'] = round(tp['confirmed_x']/tp['confirmed_y'], 3)*100
tp['%deaths'] = round(tp['deaths_x']/tp['deaths_y'], 3)*100
tp.head()


# In[208]:


fig = px.bar(tp, x='date', y='%confirmed', color='country', 
             range_y=(0, 100), title='% of confirmed cases from each country', 
             color_discrete_sequence=px.colors.qualitative.Prism)
fig.show()


# In[209]:


fig = px.bar(tp, x='date', y='%deaths', color='country', 
             range_y=(0, 100), title='% of death cases from each country', 
             color_discrete_sequence=px.colors.qualitative.Prism)
fig.show()


# In[210]:


def gt_n(n):
    countries = world_covid_grouped[world_covid_grouped['confirmed']>n]['country'].unique()
    temp = world_covid_grouped[world_covid_grouped['country'].isin(countries)]
    temp = temp.groupby(['country', 'date'])['confirmed'].sum().reset_index()
    temp = temp[temp['confirmed']>n]
    # print(temp.head())

    min_date = temp.groupby('country')['date'].min().reset_index()
    min_date.columns = ['country', 'min_date']
    # print(min_date.head())

    from_nth_case = pd.merge(temp, min_date, on='country')
    from_nth_case['date'] = pd.to_datetime(from_nth_case['date'])
    from_nth_case['min_date'] = pd.to_datetime(from_nth_case['min_date'])
    from_nth_case['num_days'] = (from_nth_case['date'] - from_nth_case['min_date']).dt.days
    # print(from_nth_case.head())

    fig = px.line(from_nth_case, x='num_days', y='confirmed', color='country', 
                  title='N days from '+str(n)+' case', height=600)
    fig.show()


# In[211]:


gt_n(100000)


# In[213]:


def plot_treemap(col):
    fig = px.treemap(country_wise, path=["country"], values=col, height=700,
                 title=col, color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.data[0].textinfo = 'label+text+value'
    fig.show()


# In[214]:


plot_treemap('confirmed')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# skipping rows with ships info
# full_table = full_table[~(ship_rows)]


#     for a, b, c in zip(l[::3], l[1::3], l[2::3]):
#         print(f'{a:<30}{b:<30}{c:<}')
# 
#     Afghanistan                   Andorra                       Belarus
#     Benin                         Belgium                       Afghanistan
#     Andorra                       Belarus                       Benin

# In[218]:


def get_moving_average(tmp, col):
    df = tmp.copy()
    return df[col].rolling(window=2).mean()

def get_exp_moving_average(tmp, col):
    df = tmp.copy()
    return df[col].ewm(span=2, adjust=True).mean()


# In[ ]:


get_moving_average(data_brazil, 'casosAcumulado')


# In[ ]:


get_exp_moving_average(data_brazil, 'casosAcumulado')


# In[ ]:


s = pd.Series(range(5))
s.rolling(window=2).sum()


# In[ ]:





# In[ ]:


def get_new_cases(tmp, col):
    diff_list = []
    tmp_df_list = []
    df = tmp.copy()

    for i, day in enumerate(df.sort_values('file_date').file_date.unique()):
        tmp_df = df[df.file_date == day]
        tmp_df_list.append(tmp_df[col].sum())

        if i == 0:
            diff_list.append(tmp_df[col].sum())
        else:
            diff_list.append(tmp_df[col].sum() - tmp_df_list[i-1])

    return diff_list


# In[ ]:


def get_day_counts(d, country):
    '''
    For each country, get the days of the spread since 500
    cases
    '''
    data = d.copy()
    result_df = pd.DataFrame([])
    result_df = data.groupby(['file_date']).agg({'confirmed': 'sum',
                                                'recovered': 'sum',
                                                'deaths': 'sum'})
    result_df['date'] = data['file_date'].unique()
    result_df['country'] = country
        
    result_df = result_df[result_df.confirmed >= 500]
    result_df.insert(loc=0, column='day', value=np.arange(len(result_df)))
    return result_df


# In[ ]:


def plot_forecast(tmp_df, train, index_forecast, forecast, confint):
    '''
    Plot the values of train and test, the predictions from ARIMA and the shadowing
    for the confidence interval.
    
    '''

    # For shadowing
    lower_series = pd.Series(confint[:, 0], index=index_forecast)
    upper_series = pd.Series(confint[:, 1], index=index_forecast)
    
    print('... saving graph')
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plt.title('ARIMA - Prediction for cumalitive case counts {} days in the future'.format(days_in_future))    
    plt.plot(tmp_df.cumulative_cases, label='Train',marker='o')
    plt.plot(tmp_df.pred, label='Forecast', marker='o')
    tmp_df.groupby('date')[['']].sum().plot(ax=ax)
    plt.fill_between(index_forecast, 
                     upper_series, 
                     lower_series, 
                     color='k', alpha=.1)
    plt.ylabel('Infections')
    plt.xlabel('Date')
    fig.legend().set_visible(True)
    fig = ax.get_figure()
    fig.savefig(os.path.join(image_dir, 'cumulative_forecasts.png'))


# In[ ]:


def forecast(tmp_df, train, index_forecast, days_in_future):
    
    # Fit model with training data
    model = auto_arima(train, trace=False, error_action='ignore', suppress_warnings=True)
    model_fit = model.fit(train)
        
    forecast, confint = model_fit.predict(n_periods=len(index_forecast), return_conf_int=True)

    forecast_df = pd.concat([tmp_df, pd.DataFrame(forecast, index = index_forecast, columns=['pred'])], axis=1, sort=False)
    date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(train_start, forecast_end)]
    forecast_df['date'] = pd.Series(date_range).astype(str)
    forecast_df[''] = None # Dates get messed up, so need to use pandas plotting
        
    # Save Model and file
    print('... saving file:', forecast_file)
    forecast_df.to_csv(os.path.join(data_dir, forecast_file))
        
    plot_forecast(forecast_df, train, index_forecast, forecast, confint)


# In[ ]:


gpd_df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# In[ ]:


covid_conf.shape


# In[ ]:


tmp_df_conf = covid_conf.copy()
tmp_df_death = covid_death.copy()
tmp_df_rec = covid_rec.copy()


# In[ ]:


clean_df_conf = clean_data(tmp_df_conf)


# In[ ]:


clean_df_conf.head(3)


# In[ ]:


clean_df_conf.info()


# In[ ]:


clean_df_conf.shape


# In[ ]:


clean_df_death = clean_data(tmp_df_death)


# In[ ]:


clean_df_death.info()


# In[ ]:


clean_df_death.shape


# In[ ]:


clean_df_death.head(3)


# In[ ]:


clean_df_rec = clean_data(tmp_df_rec)


# In[ ]:


clean_df_rec.info()


# In[ ]:


clean_df_rec.shape


# In[ ]:


clean_df_rec.head(3)


# In[ ]:


conf = fix_country_names(clean_df_conf)


# In[ ]:


deaths = fix_country_names(clean_df_conf)


# In[ ]:


recovred = fix_country_names(clean_df_conf)


# In[ ]:


country_list = list(map(lambda x:x.lower().strip(), set(conf.country.values)))


# In[ ]:


check_specified_country(conf, 'Brazil')


# In[ ]:


iso_3 = pd.read_csv('https://gist.githubusercontent.com/tadast/8827699/raw/f5cac3d42d16b78348610fc4ec301e9234f82821/countries_codes_and_coordinates.csv')


# In[ ]:


iso_3.head()


# In[ ]:


iso_3.shape


# In[ ]:


iso_3.drop(['Numeric code','Latitude (average)','Longitude (average)'], axis=1, inplace=True)


# In[ ]:


iso_3.rename(columns={'Country':'country','Alpha-2 code': 'a2code','Alpha-3 code': 'a3code'}, inplace=True)


# In[ ]:


iso_3


# In[ ]:


conf.merge(iso_3, on='country')


# In[ ]:


np.where((covid_conf['Country/Region']  == 'United States of America'))


# In[ ]:


# covid_conf['Country/Region'] = covid_conf['Country/Region'].replace('US', 'United States of America')
# covid_death['Country/Region'] = covid_death['Country/Region'].replace('US', 'United States of America')
# covid_rec['Country/Region'] = covid_rec['Country/Region'].replace('US', 'United States of America')


# In[ ]:


covid_conf['Country/Region']


# In[ ]:





# In[ ]:





# In[ ]:


path = 'Data/esus-vepi.LeitoOcupacao.csv'


# In[ ]:


hosp_ocu = pd.read_csv(path, low_memory=False)


# In[ ]:


hosp_ocu.drop(['_id', 
               '_p_usuario', 
               'origem', 
               'excluido',
               'estadoNotificacao', 
               'municipioNotificacao',
               'validado',
               '_created_at', 
               '_updated_at'], 
              axis=1, 
             inplace=True)


# In[ ]:


hosp_ocu.head()


# In[ ]:


hosp_ocu.tail()


# In[ ]:


data_brazil = pd.read_csv('Data/HIST_PAINEL_COVIDBR_07abr2021.csv', low_memory=False)


# In[ ]:


data_brazil.head(3)


# In[ ]:


data_brazil = data_brazil.dropna(subset=['estado'])


# In[ ]:


data_brazil


# In[ ]:


data_brazil.reset_index(drop=True, inplace=True)


# In[ ]:


data_brazil.columns


# In[ ]:


data_brazil.drop(['coduf', 
                  'codmun', 
                  'codRegiaoSaude',
                  'nomeRegiaoSaude',
                  'populacaoTCU2019',
                  'Recuperadosnovos', 
                  'emAcompanhamentoNovos', 
                  'interior/metropolitana',
                  'Unnamed: 17'],
                axis=1, inplace=True)


# In[ ]:


data_brazil.head()


# In[ ]:


Counter(data_brazil['estado'])


# In[ ]:


states = pd.read_csv('https://raw.githubusercontent.com/magnobiet/states-cities-brazil/master/CSV/states.csv')[['name', 'abbr']]


# In[ ]:


states


# In[ ]:


covid_conf.head()


# In[ ]:


def country_linear(df, country_col, country_list, title, data, path_to_save, fig_save=False):
    date_cols = df.columns[[col[-2:]=='20' or col[-2:]=='21' for col in df]]
    cases = [df[df[country_col] == country][date_cols] for country in country_list]
    cases = [list(i.T[i.T.columns[0]]) for i in cases]
    fig = go.Figure()
    for i in cases:
        fig.add_trace(go.Scatter(x=date_cols, y=i, name=f'<b>{country_list[cases.index(i)]}<b>'))
        fig.update_layout(title=f'<b>{title} ({data})')
        if fig_save:
            fig.write_html(f'{path_to_save}/Countries_scatter_{data}.html')
    fig.show()


# In[ ]:


country_linear(covid_conf, 
               'Country/Region', 
               ['Brazil', 'United States of America', 'Argentina'], 
               'Country confirmed cases', 
               '8/04/20', 
               path_to_save=None, 
               fig_save=False)


# In[ ]:


country_linear(covid_death, 
               'Country/Region', 
               ['Brazil', 'United States of America', 'Argentina'], 
               'Country confirmed deaths', '8/04/20', 
               path_to_save=None)


# In[ ]:


country_linear(covid_rec, 
               'Country/Region', 
               ['Brazil', 'United States of America', 'Argentina'], 
               'Country recovered cases', '8/04/20', 
               path_to_save=None)


# np.setdiff1d:
#     Find the set difference of two arrays.Return the unique values in ar1 that are not in ar2.

# In[ ]:


# s, s1 = '2020-03-20', '2021-04-17'
# s[2:4], s1[2:4]


# In[ ]:


# data_brazil_st.data[[col[2:4]=='20' or col[2:4]=='21' for col in data_brazil_st.data]]


# In[ ]:


def world_map(df, col, date=None, title=None, path_to_save=None, if_save=False):
    # col = column of interest. Ex confirmed
    df['Country/Region'] = df['Country/Region'].replace('US', 'United States of America')
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.index = world['name']
    world = world.reindex(df['Country/Region'])
    world['iso_a3'] = world['iso_a3'].fillna('NaN')
    df['iso_a3'] = world['iso_a3'].reset_index(drop=True)
    date_cols = list(df.columns[[col[-2:]=='20' or col[-2:]=='21' for col in df]])
    countries = np.array([[i]*len(date_cols) for i in df['iso_a3']]).flatten()
    dates = date_cols*len(df['Country/Region'])
    values = np.array([list(i) for i in df[date_cols].iloc]).flatten()
    data = pd.DataFrame({'country':countries, 'date':dates, col:values})
    fig = px.choropleth(data, locations='country', color=col, animation_frame='date')
    fig.update_layout(title=f'<b>{title} ({date}<b>)')
    if if_save:
            fig.write_html(f'{path_to_save}/Countries_animated_time_series_{date}.html')
    fig.show()


# In[ ]:


world_map(covid_conf, 'confirmed', date='2021-04-07', title='Covid-19 Confirmed Cases')


# In[ ]:


world_map(covid_death, 'confirmed', date='2021-04-07', title='Covid-19 Confirmed Death Cases')


# In[ ]:


world_map(covid_rec, 'confirmed', date='2021-04-07', title='Covid-19 Confirmed Recovered Cases')


# In[ ]:


def get_new_data_frame(estado, df):
    filtro = (df['estado'] == estado)
    data = df.loc[filtro]
    data = data.copy()
    return data.reset_index(drop=True)


# In[ ]:


data_brazil.municipio.str.contains('Itajaí').sum()


# In[ ]:


sc_covid = get_new_data_frame('SC', data_brazil)


# In[ ]:


sc_covid.head()


# In[ ]:


sc_covid.tail()


# In[ ]:


data = sc_covid['data'].max()


# In[ ]:


def state_bar(df, col_name1, col_name2, title, path=None, data=None, if_save=False):
    fig = px.bar(df, 
             x=col_name1, 
             y=col_name2, 
             text=col_name1,
             color=col_name1,
             hover_data=[col_name1, col_name2])
    fig.update_traces(texttemplate='%{text:}', textposition='outside')
    fig.update_layout(template="simple_white", 
                  title_text=f'<b>{title} - {data}<b>', showlegend=False)
    fig.show()
    if if_save:
        fig.write_html(f'{path}/{title}{data}.html')


# In[ ]:


state_bar(df, col_name1, col_name2, title, path=None, data=None, if_save=False)


# In[ ]:


state_bar(sc_covid, 'data', 'casosAcumulado', 'Casos Novos', data=data)


# In[ ]:


dates = pd.Series(data_brazil.data[[col[2:4]=='20' or col[2:4]=='21' for col in data_brazil.data]])


# In[ ]:


dates


# In[ ]:


# name=np.unique(data_brazil['estado'])


# In[ ]:


def state_linear(df, col_name1, col_name2, title, path=None, data=None, if_save=False):
    date_cols = df.columns[[col[2:4]=='20' or col[2:4]=='21' for col in df]]
    fig = go.Figure()
    for i in state_cases:
        fig.add_trace(go.Scatter(x=col_name1, 
                                 y=[sum(i[j]) for j in date_cols], name=np.unique(df['Province_State'])[num]))
        num += 1
    fig.show()


# In[ ]:


def state_linear(suffix):
    path = '../input/novel-corona-virus-2019-dataset/time_series_covid_19_'+suffix+'_US.csv'
    df = pd.read_csv(path)

    cases = []
    date_cols = df.columns[[col[-2:]=='20' or col[-2:]=='21' for col in df]]
    state_cases=[df[df['Province_State'] == i][date_cols] for i in np.unique(df['Province_State'])]
    num = 0

    fig = go.Figure()
    for i in state_cases:
        fig.add_trace(go.Scatter(x=date_cols, y=[sum(i[j]) for j in date_cols], name=np.unique(df['Province_State'])[num]))
        num += 1
    fig.show()

def region_linear(case, country_name):
    df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
    df = df.fillna('NaN')
    countries = [i[1].reset_index(drop=True) for i in df.groupby('Country/Region')]
    cases = [i.iloc[list(i.index)[-1]][case] for i in countries]
    country = countries[[i[0] for i in df.groupby('Country/Region')].index(country_name)]
    cases = [list(i[1][case]) for i in country.groupby('Province/State')]

    date_cols = np.unique(country['ObservationDate'])
    fig = go.Figure()
    for region in cases:
        fig.add_trace(go.Scatter(x=date_cols, y=region, name=np.unique(country['Province/State'])[cases.index(region)]))
    fig.update_layout(title=country_name)
    fig.show()



# In[ ]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world[['name', 'iso_a3']]


# In[ ]:


world[['name']].name.values


# In[ ]:


world[['iso_a3']].values


# In[ ]:


world.index = world['name']
world = world.reindex(covid_conf['Country/Region'])


# In[ ]:


world.head()


# In[ ]:


world['iso_a3'] = world['iso_a3'].fillna('NaN')
covid_conf['iso_a3'] = world['iso_a3'].reset_index(drop=True)
covid_conf[['iso_a3']]


# In[ ]:


covid_conf[covid_conf['iso_a3'] == 'BRA']


# In[ ]:


covid_conf.columns[[col[-2:]=='20' for col in covid_conf]]


# In[ ]:


covid_conf.columns[[col[-2:]=='20' or col[-2:]=='21' for col in covid_conf]]


# In[ ]:


x = np.array([[5, 6, 7, 8, 9]])
x[-2:]


# In[ ]:


date_cols = list(covid_conf.columns[[col[-2:]=='20' or col[-2:]=='21' for col in covid_conf]])
countries = np.array([[i]*len(date_cols) for i in covid_conf['iso_a3']]).flatten()
pd.Series(date_cols)


# In[ ]:


pd.Series(countries)


# In[ ]:


dates = date_cols*len(covid_conf['Country/Region'])

values = np.array([list(i) for i in covid_conf[date_cols].iloc]).flatten()
data = pd.DataFrame({'country':countries, 'date':dates, 'confirmed':values})
data


# In[ ]:


fig = px.choropleth(data, locations='country', color='confirmed', animation_frame='date')
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




