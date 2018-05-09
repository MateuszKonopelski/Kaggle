### Further Analysis: date_time



train_set.date_time = pd.to_datetime(train_set.date_time)       # convert to datetime

def date_time_summary(df):
    id_date = df.date_time.dt.date
    dates = df.groupby([id_date])['date_time'].count()
    mean_datetime = dates.mean()
    std_datetime = dates.std()
    outstanding = (dates > mean_datetime + std_datetime).sum() + (dates < mean_datetime - std_datetime).sum() 

    print(
    'Summary():\nmin date: {0}\nmax date:{1}\nunique dates: {2}\nThere are {3} day with more data than mean +/-2std'.
    format(np.min(df.date_time), 
           np.max(df.date_time),
           len(df.date_time.dt.date.unique()),
           outstanding))

date_time_summary(train_set)


def top_clusters_per_month(df):
    datetime_data = df.loc[:, ['date_time', 'hotel_cluster']]
    datetime_data['year'] = df.date_time.apply(lambda x: x.year)
    datetime_data['month'] = df.date_time.apply(lambda x: x.month)
    datetime_data['days'] = df.date_time.apply(lambda x: x.day)
    datetime_data = datetime_data.loc[:, ['year', 'days', 'month', 'date_time', 'hotel_cluster']]

    datetime_data_tc = datetime_data.groupby(['year', 'month', 'hotel_cluster'])['date_time'].count()
    datetime_data_tc = datetime_data_tc.groupby(level=['year','month']).nlargest(5).reset_index(level=[0,1], drop=True)
    return datetime_data_tc.unstack().replace(np.nan, value='') 


top_clusters_per_month(train_set)


def count_per_month_graph():
    # Fisrt graph (bar): total observatiions per montha nd year
    datetime_data.groupby(['year', 'month'])['hotel_cluster'].count().unstack(level=0).plot(kind='bar')
    plt.title('# of observed customers', fontdict={'fontsize':16})
    
    # Second graph (line): total observations per day (rolling 31 days average)
    fig, ax = plt.subplots(figsize=(10, 5))
    # separate line for each top cluster
    datetime_data_t5c = datetime_data.query('hotel_cluster == 91 or hotel_cluster == 41 or hotel_cluster == 64 or hotel_cluster == 48 or hotel_cluster == 5').loc[:, ['hotel_cluster','date_time']]
    datetime_data_t5c = datetime_data_t5c.groupby(['hotel_cluster', datetime_data_t5c.date_time.dt.date]).count()
    for cluster in datetime_data_t5c.index.get_level_values(0).unique():
        series = datetime_data_t5c.query('hotel_cluster == @cluster').rolling(31).mean()
        data = series.as_matrix()
        dates = series.index.get_level_values(1)
        label = 'hotel cluster {0}'.format(cluster)
        ax.plot(dates, data,label=label)
    ax.legend()
    # all clusters line
    series = train_set.groupby(train_set.date_time.dt.date)['date_time'].count().rolling(31).mean()
    data = series.as_matrix()
    dates = series.index.get_level_values(0)
    label = 'All hotels'
    ax2 = ax.twinx()
    ax2.plot(dates, data,label=label, linestyle='-', linewidth=2, color='black')
    ax2.legend(loc='lower right')
    ax2.set_title('# of observed customers', fontdict={'fontsize':20})
    fig.tight_layout()
 
 
 
