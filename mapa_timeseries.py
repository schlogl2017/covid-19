#!usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import geopandas as gpd
import pandas as pd
import plotly_express as px
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser(description='Criando bubble maps interativos com\
 dados Covid-19 em Santa Catarina')
parser.add_argument('--input1',
                    '-i',
                    type=str,
                    required=True,
                    dest='filename1',
                    help='Nome do arquivo .shp para leitura com Geopandas.')
parser.add_argument('--input2',
                    '-ii',
                    type=str,
                    required=True,
                    dest='filename2',
                    help='Nome do arquivo .csv para leitura com pandas.')
parser.add_argument('--output',
                    '-o',
                    type=str,
                    required=True,
                    dest='outputfile',
                    help='Nome do arquivo .csv para leitura com pandas.')

args = parser.parse_args()


gdf = gpd.read_file(args.filename1)


print(gdf.sample(3), '\n')


print(gdf.isnull().any(), '\n')


print(gdf.info(), '\n')


gdf = gdf.rename(columns={'NM_MUNICIP': 'municipios', 'CD_GEOCMU': 'ibge'})


print(gdf.sample(3), '\n')


print(gdf.isnull().any(), '\n')


df = pd.read_csv(args.filename2)


print(df.info(), '\n')


print(df.isnull().any(), '\n')


fl_sc = (df['state'] == 'SC') & (df['city'] != 'Importados/Indefinidos')
df_sc = df.loc[fl_sc]


print(df_sc.sample(3), '\n')


cit = (df['place_type'] == 'city') & (df['city'] != 'Importados/Indefinidos')
sc_cities = df_sc.copy()
sc_cities = df_sc.loc[cit]


print(sc_cities.sample(3), '\n')


print(sc_cities.isnull().any(), '\n')


na = sc_cities['city_ibge_code'].isnull()
drops = sc_cities.loc[na].index


sc_cities = sc_cities.copy()
sc_cities.drop(drops, axis=0, inplace=True)
print(sc_cities.sample(3), '\n')


print(sc_cities.isnull().any(), '\n')


sc_cities.drop(['epidemiological_week',
                'order_for_place',
                'state',
                'place_type',
                'last_available_confirmed_per_100k_inhabitants',
                'new_confirmed',
                'new_deaths',
                'last_available_death_rate',
                'estimated_population_2019',
                'is_last',
                'is_repeated'],
               axis=1,
               inplace=True)


sc_cities['city'] = sc_cities['city'].str.upper()


print(sc_cities.sample(3), '\n')


sc_cities.rename(columns={'date': 'data',
                          'city': 'municipios',
                          'city_ibge_code': 'ibge',
                          'last_available_confirmed': 'confirmados',
                          'last_available_deaths': 'obitos'
                         },
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


print(gdf_points_4674.head(), '\n')


fig_time= px.scatter_mapbox(gdf_points_4674,
                            lat=gdf_points_4674.geometry.y,
                            lon=gdf_points_4674.geometry.x,
                            hover_name= 'municipios',
                            hover_data=['municipios', 'confirmados'],
                            size = gdf_points_4674['confirmados'].values,
                            animation_frame='data',
                            mapbox_style='open-street-map',
                            template='plotly_dark',
                            zoom=5.5,
                            size_max=50)
#fig_time.show()



fig_time.write_html(args.outputfile + '.html')


print('Script finalizado com sucesso!')
print('Cheque o mapa no diretório.')
