#!usr/bin/env python
import pandas as pd

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


def replace_spaces(name):
    """It substitutes the spaces with a underline.
    Input:
    name - string
    Output:
    name - undescored name
    """
    return name.replace(' ', '_')

def replace_undereline(name):
    """It substitutes the undescore with a space.
    Input:
    name - string
    Output:
    name - spaced name
    """    
    return name.replace('_', ' ')


def get_top_N_countries(df, col_country, col_value, n=10):
    top = df.groupby(col_country).agg({col_value: 
                                       'sum'}).sort_values(col_value,
                                                           ascending=False).head(n)
    return top   


def get_names(df, col_name):
    """Returns the names/values from a pandas data frame
    column.
    Input:
    df - data frame
    col_name - column name to get the values
    Output - returns the values of the coulmn from the datframe.    
    """
    countries = sorted(list(set(df[col_name].values)))
    for a,b,c in zip(countries[::3],countries[1::3],countries[2::3]):
        # align right with 30 spaces
        print('{:<30}{:<30}{:<}'.format(a,b,c))
        
    print('\n\033[1;31mNUMBER OF COUNTRIES/AREAS INFECTED:\033[0;0m', len(countries))

# for a, b, c in zip(l[::3], l[1::3], l[2::3]):
#    print(f'{a:<30}{b:<30}{c:<}')

# Afghanistan                   Andorra                       Belarus
# Benin                         Belgium                       Afghanistan
# Andorra                       Belarus                       Benin  
   
def get_similar_countries(c, country_list):
    pos_countries = get_close_matches(c, country_list)
    
    if len(pos_countries) > 0:
        print(c, 'was not listed. did you mean', pos_countries[0].capitalize() + '?')
        sys.exit()
    else:
        print(c, 'was not listed.')
        sys.exit()


def check_specified_country(df, country):
    if country:
        print('Country specified')
        if country.lower() == 'china':
            print(country, 'was not listed. did you mean Mainland China?')
            
        elif country.lower() not in country_list:
            get_similar_countries(country, country_list)
            
        else:
            print('... filtering data for', country)
            if len(country) == 2:
                df = df[df.country == country.upper()]
            else:
                df = df[df.country == country.capitalize()]
            return df
    else:
        print('No specific country specified')
        return df


def get_top_countries(df, col, n):
    # Get top N infected countries
    df = df[df[col] == df[col].max()]
    return df.groupby(['country']).agg({'confirmed': 'sum'}).sort_values('confirmed',ascending=False).head(n).index 


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


def drop_duplicates(df_raw):
    '''
    Take the max date value for each province for a given date
    '''
    days_list = []
    
    for datetime in df_raw.date.unique():
        tmp_df = df_raw[df_raw.date == datetime]
        tmp_df = tmp_df[df_raw.file_date != datetime].sort_values(['file_date']).drop_duplicates('Province/State', keep='last')
        days_list.append(tmp_df)

    return days_list


def fix_country_names(df):
    '''
    Cleaning up after data management Cisjordania West Bank and Gaza
    '''
    # Asian Countries
    df['country'] = np.where((df['country']  == 'Korea, South'),'South Korea', df['country'])
    df['country'] = np.where((df['country']  == 'Taiwan*'),'Taiwan', df['country'])
    df['country'] = np.where((df['country']  == 'West Bank and Gaza'),'Cisjordania', df['country'])
    #European Countries
    df['country'] = np.where((df['country']  == 'Bosnia and Herzegovina'),'Bosnia', df['country'])
    # others
    df['country'] = np.where((df['country']  =='Australian Capital Territory'), 'Australia', df['country'])
    #African Countries
    df['country'] = np.where((df['country']  == 'Congo (Brazzaville)'),'Congo', df['country'])
    df['country'] = np.where((df['country']  == 'Congo (Kinshasa)'),'Congo', df['country'])
    return df


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


def get_new_data_frame(estado, df):
    filtro = (df['estado'] == estado)
    data = df.loc[filtro]
    data = data.copy()
    return data.reset_index(drop=True)


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


def merge_JHU_data_frames(df1, df2, df3):
    full_df = pd.merge(left=df1, right=df2, how='left',
                      on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])
    full_df = pd.merge(left=full_df, right=df3, how='left',
                      on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])
    return full_df



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





























