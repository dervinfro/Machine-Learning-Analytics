import pandas as pd

pd.options.display.max_columns=None
pd.options.display.max_rows=None

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/Wrightsample.csv')


north = df[df['TREATB']=='North']
sorth = df[df['TREATB']=='South']
outer = df[df['TREATB']=='Outside']

north_pre = north[north['PT']=='Pre-barrier']
north_post = north[north['PT']=='Post-barrier']

diff_means = north_post['carthefts_pc'].mean() - north_pre['carthefts_pc'].mean()
perc_reduction = diff_means/north_pre['carthefts_pc'].mean() * 100
print('There is a reduction of: {} and a percent change of: {} '.format(round(diff_means,3), round(perc_reduction,3)))
