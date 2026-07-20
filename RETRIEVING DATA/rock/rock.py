import pandas as pd
import sqlite3 as sq 
import pandas.io.sql as pds

path = "classic_rock.db"

con = sq.connect(path)

query = '''select * from rock_songs'''

allrocksongs_observations = pd.read_sql(query, con)

print(allrocksongs_observations.head())

query1='''
SELECT Artist, Release_Year, COUNT(*) AS num_songs, AVG(PlayCount) AS avg_plays  
    FROM rock_songs
    GROUP BY Artist, Release_Year
    ORDER BY num_songs desc;
'''

# Execute the query
observations_generator = pds.read_sql(query1,
                            con,
                            coerce_float=True, # Doesn't efefct this dataset, because floats were correctly parsed
                            parse_dates=['Release_Year'], # Parse `Release_Year` as a date
                            chunksize=5 # Allows for streaming results as a series of shorter tables
                           )

for index, observations in enumerate(observations_generator):
    if index < 5:
        print(f'Observations index: {index}'.format(index))
        print(observations)