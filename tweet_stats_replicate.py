import pandas as pd
import csv

fp_gh = 'downloads/data/ghana-tweet.csv'
fp_ph = 'downloads/data/philippines-tweet.csv'
fp_vz = 'downloads/data/venezuela-tweet.csv'

file_paths = [fp_vz, fp_ph, fp_gh]
election = ['Venezuela', 'Philippines', 'Ghana']
language = ['Spanish', 'English', 'English']

violence = []
_none = []
_none_elect = []
_none_elect_percent = []
total = []

for i, fp in enumerate(file_paths):
    with open(fp, 'r') as f:
        reader = csv.DictReader(f)
        vio_count = 0
        _none_count = 0
        _none_elect_count = 0
        total_count = 0

        for line in reader:
            total_count += 1
            if line['election'] == 'yes':
                if line['violence'] == 'violence':
                    vio_count += 1
                else:
                    _none_count += 1
            else:
                _none_elect_count += 1
        violence.append(vio_count)
        _none.append(_none_count)
        _none_elect.append(_none_elect_count)
        _none_elect_percent.append(round(float(_none_elect_count)*100/total_count, ndigits=1))
        total.append(total_count)

tweets_stats = {'Election': election,
                'Language': language,
                'Violence': violence,
                'None': _none,
                'Non-Election': ['{} ({}%)'.format(_none_elect[i], _none_elect_percent[i]) for i in xrange(len(_none_elect))],
                'Total': total}

df = pd.DataFrame.from_dict(tweets_stats)
print df[['Election', 'Language', 'Violence', 'None', 'Non-Election', 'Total']]