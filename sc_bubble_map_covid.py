#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from datetime import date
import geopandas as gpd
import pandas as pd
import plotly.express as px


# This script assumes that the csv is comma delimited.
# The source of the csv files are: https://brasil.io/dataset/covid19/caso/
# csv = city
# data do dia atual
# else need to be modified


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


# lendo as coordenadas do estado com o geopandas
sc = gpd.read_file(args.filename1)
print('Conferindo o GeoDataFrame.')
print(sc.sample(3), '\n')


# checando as informações do data frame
print(sc.info(), '\n')
print('Checando se há dados nulos')
print(sc.isnull().sum(), '\n')


# renomeando as colunas do dataframe
cols = sc.columns
print(f'Nome original das colunas: {cols}', '\n')
sc = sc.rename(columns={'NM_MUNICIP': 'municipios', 'CD_GEOCMU': 'ibge'})
n_cols = sc.columns
print(f'Renomeando as colunas: {n_cols}')
print(sc.sample(3), '\n')


# lendo os dados do csv com informações do covid em SC
df = pd.read_csv(args.filename2)
df['city'] = df['city'].str.upper()
print('Checando os dados do dataframe com os dados para mapear.')
print(df.sample(3), '\n')


# Alterando a data para datetime obj
df['date'] = pd.to_datetime(df['date'])
print(df['date'].dtype, '\n')


# Filtando os dados de Santa Catarina
fil = (df['place_type'] == 'city') & (df['state'] == 'SC') & (df['city'] != 'Importados/Indefinidos')
sc_cid = df[fil]
print('Selecionando os dados de Santa Catarina')
print(sc_cid.sample(3), '\n')


idx = sc_cid.loc[pd.isna(sc_cid['city_ibge_code']), :].index.values


# deletando as colunas com NaN dos codigos dos municipios do ibge
sc_cid = sc_cid.drop(idx)
NA = (sc_cid['city_ibge_code'].isna())
print(sc_cid[NA], '\n')


# deixando de fora do df colunas que não serão utilizadas
sc_cid.drop(['epidemiological_week',
             'order_for_place',
             'state',
             'place_type',
             'last_available_confirmed_per_100k_inhabitants',
             'last_available_death_rate',
             'estimated_population_2019',
             'is_last',
             'is_repeated'], axis=1, inplace=True)


# checando se há dados nulos no data frame
print(sc_cid.isnull().any(), '\n')


# Renomeando colunas
sc_cid.rename(columns= {'date': 'data',
                        'city': 'municipios',
                        'city_ibge_code': 'ibge',
                        'last_available_confirmed': 'confirmados',
                        'new_confirmed': 'novos_casos',
                        'last_available_deaths': 'obitos',
                        'new_deaths': 'novos_obitos'
                       }, inplace=True)

print(sc_cid.sample(3), '\n')


# fazendo o merge do dois dataframes
# renomenado as colunas do novo GeoDataFrame
df2 = pd.merge(sc, sc_cid, on=['municipios'],  how='left')
df2 = df2.rename(columns={'ibge_x': 'ibge'}).drop(['ibge_y', ], axis=1)
# colocando as datas nos dados NaN
df2['data'] = date.today().strftime('%Y-%m-%d')
# preenchendo os outros dados vazios
df2['confirmados'].fillna(0, inplace=True)
df2['obitos'].fillna(0, inplace=True)

# Gerando o grafico
# tranformando geometry em points
gdf_points = df2.copy()
gdf_points['geometry'] = gdf_points['geometry'].centroid
gdf_points_4674 = gdf_points.to_crs("EPSG:4674")

px.set_mapbox_access_token(
    "pk.eyJ1Ijoic2hha2Fzb20iLCJhIjoiY2plMWg1NGFpMXZ5NjJxbjhlM2ttN3AwbiJ9.RtGYHmreKiyBfHuElgYq_w")
fig = px.scatter_mapbox(gdf_points_4674,
                        lat=gdf_points_4674.geometry.y,
                        lon=gdf_points_4674.geometry.x,
                        size="confirmados",
                        color="confirmados",
                        hover_name="confirmados",
                        color_continuous_scale=px.colors.sequential.Bluered,
                        size_max=15,
                        zoom=6)


# Salvando o grafico
fig.write_html(args.outputfile + '.html')


print('Script finalizado com sucesso!')
print('Cheque o mapa no diretório.')
