#!usr/bin/env python
# -*- coding: utf-8 -*-
# This script assumes that the csv is comma delimited.
# The source of the csv files are: https://brasil.io/dataset/covid19/caso/
# csv = city
# data do dia atual
# else need to be modified


import argparse
import pandas as pd
from datetime import date, timedelta
import plotly_express as px


parser = argparse.ArgumentParser(description='Criando grafico interativos com\
 dados confirmados de Covid-19 nos municipios de Santa Catarina')
parser.add_argument('--input',
                    '-i',
                    type=str,
                    required=True,
                    dest='filename1',
                    help='Nome do arquivo .shp para leitura com Geopandas.')
parser.add_argument('--output',
                    '-o',
                    type=str,
                    required=True,
                    dest='outputfile',
                    help='Nome do arquivo .csv para leitura com pandas.')
args = parser.parse_args()


# lendo os dados do csv com informações do covid em SC
df = pd.read_csv(args.filename1)
print(df.sample(3), '\n')


# selecionando os dados de Santa Catarina
sc = (df['state'] == 'SC') & (df['place_type'] == 'city') & (df['city'] != 'Importados/Indefinidos')
sc_cid = df[sc].copy()
print(sc_cid.sample(3), '\n')


# dispensando as colunas com dados não utilizados
# tirando do data frame as colunas que não entram na análise
sc_cid.drop(['epidemiological_week',
             'order_for_place',
             'state',
             'city_ibge_code',
             'place_type',
             'last_available_confirmed_per_100k_inhabitants',
             'last_available_death_rate',
             'estimated_population_2019',
             'is_last',
             'is_repeated'], axis=1, inplace=True)
print(sc_cid.sample(3), '\n')


# renomeando as colunas
sc_cid.rename(columns= {'date': 'data',
                        'city': 'cidades',
                        'last_available_confirmed': 'confirmados',
                        'new_confirmed': 'novos_casos',
                        'last_available_deaths': 'obitos',
                        'new_deaths': 'novos_obitos'
                       }, inplace=True)
print(sc_cid.sample(3), '\n')


# Os dados do site sempre estão com um dia de atraso
# então fazendo a seleção do dia atual do dado
today = date.today().isoformat()
yesterday = date.today() - timedelta(days=1)
atual = yesterday.isoformat()
print('Data atual dos dados no arquivo:')
print(atual, '\n')


# fazendo a seleção dos dados atuais
hj = (sc_cid['data'] == atual) & (sc_cid['confirmados'] >= 10)
hoje = sc_cid[hj].copy()
hoje.sort_values('confirmados', inplace=True, ascending=False)
print(hoje.sample(3), '\n')


# criando o grafico
fig = px.bar(hoje, y='confirmados', x='cidades')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
# Salvando o grafico
fig.write_html(args.outputfile + '.html')


print('Script finalizado com sucesso!')
print('Cheque o mapa no diretório.')
