def melt(df, var_name):
    return df.melt(
        id_vars=df.columns[0], 
        value_vars=df.columns[1:], 
        var_name='date', 
        value_name=var_name
    ).dropna()
    
 def update_pandas_settings():
    pd.options.mode.chained_assignment = None
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.options.display.float_format = '{:.1f}'.format


def str_to_date(date_str, fmt='%Y-%m-%d'):
    """Convert string date to datetime object."""
    return datetime.datetime.strptime(date_str, fmt).date()
 
 def us_to_iso_date(datestr):
    return datetime.strptime(datestr, '%m/%d/%y').strftime('%Y-%m-%d')
 
 
 def transform_and_standardize(df, var_name):
    df = df.drop(columns=['Lat', 
                          'Long'])
    df = df.merge(df_regions,
                  how='left',
                  on=['Country/Region', 
                      'Province/State']).rename(columns={'OWID Country Name': 'location'})
    if df['location'].isnull().any():
        print("missing mappings in %s for:" % var_name)
        print(df[df['location'].isnull()][['Country/Region', 
                                           'Province/State', 
                                           'location']])
        assert False
    df = df.groupby('location').sum().reset_index()
    df = melt_csv(df, var_name)
    df['date'] = df['date'].map(us_to_iso_date)
    return df.sort_values(by=['location', 'date'])
 
 
 def get_grouped_by_date_states_data(estado, df):
    """Returns a new data frame with the time series from the states selected
    grouped by date, confirmed cases and deaths cases"""
    filtro = (df['state'] == estado)
    data = df.loc[filtro]
    na = data['city'].isnull()
    drop = list(data.loc[na].index)
    data = data.copy()
    data.drop(drop, inplace=True)
    return data.groupby(['date'])[['last_available_confirmed', 
                                                  'last_available_deaths']].agg('sum').reset_index()
 
 
 def states_data(df, state, place_type):
    """Returns a new data frame with data from the brazilian states selected"""
    temp = df.copy()
    idx_to_drop = temp.index[temp['city'] == 'Importados/Indefinidos'].to_list()
    temp.drop(idx_to_drop, inplace=True)
    f = (df_IO['state'] == state) & (df_IO['place_type'] == place_type)
    temp = temp.loc[f].reset_index(drop=True)
    func = lambda x: 0 if x < 0 else x
    temp['new_confirmed'] = temp['new_confirmed'].apply(func)
    return temp
 
 
 def world_map(suffix):
    # downloading the world data from https://github.com/CSSEGISandData
    path = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{suffix}_global.csv'
    df = pd.read_csv(path)
    df['Country/Region'] = df['Country/Region'].replace('US', 'United States of America')
    
    # getting the iso symbols for the countries
    iso = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    iso.index = iso['name']
    # indexing the iso df with the country column of the covid data
    iso = iso.reindex(df['Country/Region'])
    iso['iso_a3'] = iso['iso_a3'].fillna('NaN')
    # creating the iso column iith the iso a3 symbols in the covid data frame
    df['iso_a3'] = iso['iso_a3'].reset_index(drop=True)
    
    date_cols = list(df.columns[4:-1])
    countries = np.array([[i] * len(date_cols) for i in df['iso_a3']]).flatten()
    dates = date_cols * len(df['Country/Region'])
    values = np.array([list(i) for i in df[date_cols].iloc]).flatten()
    data = pd.DataFrame({'country':countries, 'date':dates, suffix:values})

    fig = px.choropleth(data, locations='country', color=suffix, animation_frame='date')
    fig.show()
 
 
func = lambda x: 0 if x < 0 else x

 
 def world_data(case_type='confirmed'):
    case = pd.read_csv(f'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_{case_type}_global.csv&filename=time_series_covid19_{case_type}_global.csv')
    case = case[case['Province/State'].isna() == True]
    case.index = case['Country/Region'].to_list()
    case = case.drop(['Province/State', 'Lat', 'Long', 'Country/Region'], axis = 1)
    return case
 
 
 def world_data_raw(case_type='confirmed'):
    case = pd.read_csv(f'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_{case_type}_global.csv&filename=time_series_covid19_{case_type}_global.csv')
    case = case[case['Province/State'].isna() == True]
#     case.index = case['Country/Region'].to_list()
#     case = case.drop(['Province/State', 'Lat', 'Long', 'Country/Region'], axis = 1)
    return case
 
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
 
def data_by_country(df_raw, case_type, country_name):
    """Returns a new data  frame from a more raw data frame"""
    # Calculates the difference of a Dataframe element compared with another element
    # in the Dataframe (default is element in previous row).
    # gets the daily cases
    df_country = pd.DataFrame({'data': df_raw.loc[country_name].diff().index, 
                                 case_type: list(df_raw.loc[country_name].diff())})
    # selecting data with values greater or equal to one
    df_country = df_country[df_country[case_type] >= 1]
    return df_country.reset_index(drop=True)    
 
def make_bar_plot(df, xdata, ydata, title, xaxis_title, yaxis_title, date):
    """Data frames 2 cols: data, casos"""
    fig = px.bar(df, 
             x=xdata, 
             y=ydata)
    
    fig.update_layout(title=f'<b>{title}: {date}<b>',
                   xaxis_title=xaxis_title,
                   yaxis_title=f'{yaxis_title} - ( {date} )')
    fig.show() 
 
 def get_case_fatality_rate(df_deaths, df_confirmed):
    df_crf = (df_deaths) / (df_confirmed)
    for country in df_crf.index:
        df_crf.loc[country].replace(np.nan, 0.0, inplace=True)
    return df_crf                          
 
 def change_country_color(country_list, color, color2, country):
    """ returns a list of color to destate the country of interest"""
    colors = [color] * len(country_list)
    for i, c in enumerate(country_list):
        if c == country:
            colors[i] = color2
    return colors
 
 
 def get_horizontal_bar_plot_from_series(data, 
                                        title,
                                        country_list,
                                        color1, 
                                        color2,
                                        date, 
                                        country,
                                        xaxis_title,
                                        yaxis_title='country'):
    # country_list = cfr_hoje_sa.index.to_list()
    # country_list -> to come ordered
    
    colors = change_country_color(country_list, color1, color2, country)
    fig = go.Figure(go.Bar(x=np.round(data.values, 3),
                           y=data.index,
                           text=data.values,
                           marker_color=colors, 
                           orientation='h'))

    fig.update_traces(marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6,
                      texttemplate='<b>%{x}<b>', textposition='outside')
    
    fig.update_layout(title_text=f'<b>{title}: {date}<b>',
                  xaxis_tickfont_size=14,
                  yaxis=dict(title=f'{yaxis_title}',
                             titlefont_size=16,
                             tickfont_size=14,))
    
    fig.update_xaxes(title_text=xaxis_title)

    fig.show()
 
 
 def get_interest_country_data(df, 
                              country_list, 
                              pop_dict, 
                              title, 
                              fig_name,
                              ylabel,
                              date,
                              n,
                              suffix='png',
                              is_save=False, 
                              is_log=False):
    """ JHU large csv file data country.index(rows) dates(cols)
    """
    # 10000
    n = n
    plt.style.use('ggplot')
    plt.figure(figsize=(20, 15))
    plt.rcParams["font.weight"] = "bold"
    for country in country_list:
        if is_log:
            df.loc[country].apply(lambda x: np.log10((x / pop_dict[country]) * n + 1)).plot(style='.-',
                                                                                            fontsize = 15)
        else:
            df.loc[country].plot(style='.-', fontsize = 15)
    plt.xlabel('Date', fontsize = 15)
    plt.ylabel(f'{ylabel}', fontsize = 15)
    plt.title(f'{title} ({date})', fontsize = 20)
    plt.legend(country_list, loc='upper left', fontsize = 15)
    if is_save:
        plt.savefig(f'{fig_name}.{suffix}')
    plt.show()
 
 def get_name_for_legend(country_list):
    """Returns a dictionary to be used if needed to change the legend
    in plotly plots"""
    plotly_names = [f'wide_variable_{i}' for i in range(len(country_list))]
    dict_legends = dict(zip(plotly_names, country_list))
    return dict_legends
 
 def customLegend(fig, nameSwap):
    """This not change the name in hover"""
    for i, dat in enumerate(fig.data):
        for elem in dat:
            if elem == 'name':
                fig.data[i].name = nameSwap[fig.data[i].name]
    return(fig)
 
 def customLegendPlotly(fig, nameSwap): 
        'This change the hover name'
        for i, dat in enumerate(fig.data): 
            for elem in dat: 
                if elem == 'hovertemplate':
                    fig.data[i].hovertemplate = fig.data[i].hovertemplate.replace(fig.data[i].name, nameSwap[fig.data[i].name]) 
                    for elem in dat: 
                        if elem == 'name':
                            fig.data[i].name = nameSwap[fig.data[i].name] 
        return fig 
 
 
 def data_frame_log10_case_million_covid_countries(df_type_case, dict_pop, country_list):
    # make a list of pandas series log10
    data = [df_type_case.loc[country].apply(lambda x: np.log10((x / dict_pop[country]) * 1000000 + 1)) for 
        country in country_list]
    # concatenate all series
    cases_10_million = pd.concat(data, axis=1, keys=[n.name for n in data]).T
    return cases_10_million
 
 
 def line_plots_countries(df, country_list, title, x_axis_name, y_axis_name, leg_title, date):
    """For large df data county.index/date.values"""
    fig = go.Figure()
    
    for country in country_list:
        fig.add_trace(go.Scatter(x=df.columns.values,
                                 y=df.loc[country],
                             mode='lines+markers',
                             hovertemplate=None,
                             name=country))
        
        fig.update_layout(title=f'{title} ( {dia} )',
                          xaxis_title=x_axis_name,
                          yaxis_title=y_axis_name,)
        
        fig.update_layout(legend_title_text=f'{leg_title}')
        
        fig.update_layout(hovermode="x unified")
    fig.show() 
 
 
 def get_interest_country_data_per_day(df,
                                      country_list, 
                                      pop_dict, 
                                      title, 
                                      fig_name,
                                      day_initial,
                                      ylabel,
                                      suffix='png', 
                                      is_save=False, 
                                      is_log=False):
    """ JHU large csv file data 
    country.index(rows) dates(cols)
    """
    day_0 = day_initial
    n = 1000000
    plt.style.use('ggplot')
    plt.figure(figsize=(20, 15))
    plt.rcParams["font.weight"] = "bold"
    for country in country_list:
        if is_log:
            df.loc[country][df.loc[i] > day_0].apply(lambda x: np.log10((x / pop_dict[country]) * n + 1)).plot(style='.-',
                                                                                            fontsize = 15)
        else:
            df.loc[country][df.loc[i]>day_0].plot(style='.-', fontsize = 15)
    plt.xlabel('Date', fontsize = 15)
    plt.ylabel(f'{ylabel}', fontsize = 15)
    plt.title(f'{title}', fontsize = 20)
    plt.legend(country_list, loc='upper left', fontsize = 15)
    if is_save:
        plt.savefig(f'{fig_name}.{suffix}')
    plt.show()
 
 
 def line_plots_countries_cases_start(df, 
                                   country_list, 
                                   pop_dict, 
                                   title, 
                                   day_init, 
                                   x_axis_name, 
                                   y_axis_name, 
                                   leg_title, date):
    """For large df data county.index/date.values"""
    fig = go.Figure()
    
    for country in country_list:
        fig.add_trace(go.Scatter(x=df.columns.values,
                                 y=df.loc[country][df.loc[country] > day_init].apply(lambda x:
                                         np.log10((x/pop_dict[i])*1e+6+1)),
                                 mode='lines+markers',
                                 hovertemplate=None,
                                 name=country))
        
        fig.update_layout(title=f'{title} {day_init} ( {dia} )',
                          xaxis_title=x_axis_name,
                          yaxis_title=y_axis_name,)
        
        fig.update_layout(legend_title_text=f'{leg_title}')
        
        fig.update_layout(hovermode="x unified")
    fig.show()  
 
 
 def ode_model(z, t, beta, sigma, gamma, mu):
    S, E, I, R, D = z
    N = S + E + I + R + D
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I - mu*I
    dRdt = gamma*I
    dDdt = mu*I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def ode_solver(t, initial_conditions, params):
    initE, initI, initR, initN, initD = initial_conditions
    beta, sigma, gamma, mu = params
    initS = initN - (initE + initI + initR + initD)
    res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma, mu))
    return res
 
 
 def my_predict(parameters):
    initE = parameters[0]
    initI = parameters[1]
    initR = parameters[2]
    initN = parameters[3]
    initD = parameters[4]
    beta = parameters[5]
    sigma = parameters[6]
    gamma = parameters[7]
    mu = parameters[8]
    
    initial_conditions = [initE, initI, initR, initN, initD]
    params = [beta, sigma, gamma, mu]
    tspan = np.arange(0, days, 1)
    sol = ode_solver(tspan, initial_conditions, params)
    S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
    return np.sqrt(np.sum((np.log10(D+1)-np.log10(np.array(list(deaths.loc['Brazil']))+1))**2)) / len(days_data)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
