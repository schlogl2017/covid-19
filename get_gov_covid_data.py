#!usr/bin/env python


import json
import requests
import pandas as pd
from datetime import datetime


df = pd.DataFrame(None, columns=['date', 'hour', 'state',
                                             'suspects', 'refuses', 'cases'])


url = "http://plataforma.saude.gov.br/novocoronavirus/resources/scripts/database.js"
request = requests.get(url)


content = request.content.decode('utf8').replace('var database=', '')
data = json.loads(content)


suspects = value.get('suspects', 0)
refuses = value.get('refuses', 0)
cases = value.get('cases', 0)
deaths = value.get('deaths', 0)


for record in data['brazil']:
    for value in record:
        state = STATES[int(value['uid'])]
        suspects = value.get('suspects', 0)
        refuses = value.get('refuses', 0)
        cases = value.get('cases', 0)
        deaths = value.get('deaths', 0)
        
        df = df.append(dict(zip(df.columns, 
                                [record['date'], record['time'], state,
                                 	suspects, refuses, cases])),
                       ignore_index=True)

df.to_csv('Data/brazil_covid19.csv', index=False)
