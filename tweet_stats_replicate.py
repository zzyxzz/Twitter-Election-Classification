import pandas as pd
tweets_stats = {'Election': ['Venezuela', 'Philippines', 'Ghana'],
                'Language': ['Spanish', 'English', 'English'],
                'Violence': [294, 193, 188],
                'None': [1890, 1562, 1075],
                'Non-Election': ['3474 (60%)', '2408 (58%)', '1999 (61%)'],
                'Total': [5747, 4163, 3253]}

df = pd.DataFrame.from_dict(tweets_stats)

print df[['Election', 'Language', 'Violence', 'None', 'Non-Election', 'Total']]
