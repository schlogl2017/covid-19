#!/usr/bin/env python
# coding: utf-8


import numpy as np
import geopandas as gpd
import pandas as pd
import plotly_express as px
import plotly.io as pio


gdf = gpd.read_file('sc_municipios/42MUE250GC_SIR.shp')


print(gdf.sample(3))


gdf.isnull().any()


gdf.info()


gdf = gdf.rename(columns={'NM_MUNICIP': 'municipios', 'CD_GEOCMU': 'ibge'})


gdf.sample(3)


gdf.isnull().any()


df = pd.read_csv('data2/covid19-15_05_20.csv')


df.info()


df.isnull().any()


fl_sc = (df['state'] == 'SC')
df_sc = df.loc[fl_sc]


df_sc.sample(3)


cit = (df['place_type'] == 'city')
sc_cities = df_sc.copy()
sc_cities = df_sc.loc[cit]


sc_cities.sample(3)


sc_cities.isnull().any()


na = sc_cities['city_ibge_code'].isnull()
drops = sc_cities.loc[na].index


sc_cities = sc_cities.copy()
sc_cities.drop(drops, axis=0, inplace=True)
sc_cities.sample(3)


sc_cities.isnull().any()


sc_cities.columns


sc_cities.drop(['state',
             'place_type',
             'is_last',
             'estimated_population_2019',
             'confirmed_per_100k_inhabitants'],
            axis=1,
            inplace=True)


sc_cities['city'] = sc_cities['city'].str.upper()


sc_cities.sample(3)


sc_cities.rename(columns={'date': 'data',
                          'city': 'municipios',
                          'confirmed': 'confirmado',
                          'deaths': 'obitos',
                          'city_ibge_code': 'ibge',
                          'death_rate': 'mortalidade'},
                 inplace=True)


sc_cities = sc_cities.sort_values('data')


df2 = pd.merge(gdf, sc_cities, on=['municipios'],  how='left')


na = df2['data'].isnull()
idx_nan = df2.loc[na].index
idx_nan


df2.drop(idx_nan, inplace=True)


gdf_points = df2.copy()
gdf_points['geometry'] = gdf_points['geometry'].centroid
gdf_points_4674 = gdf_points.to_crs("EPSG:4674")


gdf_points_4674 = gdf_points_4674.sort_values('data')


gdf_points_4674.head()


gdf_points_4674.size


fig_time= px.scatter_mapbox(gdf_points_4674,
                            lat=gdf_points_4674.geometry.y,
                            lon=gdf_points_4674.geometry.x, 
                            hover_name= 'municipios', 
                            hover_data=['municipios', 'confirmado'], 
                            size = gdf_points_4674['confirmado'].values,
                            animation_frame='data', 
                            mapbox_style='open-street-map', 
                            template='plotly_dark', 
                            zoom=5.5,
                            size_max=50)
#fig_time.show()


pio.write_html(fig_time, file='graficos/sc_COVID_realtime.html', auto_open=True)



