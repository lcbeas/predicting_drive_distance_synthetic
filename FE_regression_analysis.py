import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from fixed_effects_regression import pDF,hDF, maindf
from math import sqrt

df = pd.read_csv('FE_regression.csv') # load dataframe from R

plyrfctr = []
holefctr = []
players = []
holes = []
players.append('Player 0') # need to add this because dummy variables do not list Player 0 or Hole 0 because values are 0
plyrfctr.append(0)
holes.append('Hole 0')
holefctr.append(0)


for row, factor in enumerate(df['Factor']): # all of this is just to clean up data frame
	if factor.startswith('data$Player'):
		factor = factor[11:]
		players.append(factor)
		plyrfctr.append(df.iloc[row].loc['data$Distance'])

	if factor.startswith('data$Hole'):
		factor = factor[9:]
		holes.append(factor)
		holefctr.append(df.iloc[row].loc['data$Distance'])



plydf = pd.DataFrame({'Player': players, 'Factor': plyrfctr}) # unordered but correctly matched
hldf = pd.DataFrame({'Hole':holes, 'Factor': holefctr})

#print(plydf)
#print(hldf)

pindex = plydf['Player'].str.strip('Player ') # three blocks below just to rescale by weighted mean and correctly order the data frames
hindex = hldf['Hole'].str.strip('Hole ')
plydf['newpIdx'] = pindex.astype(int)		# creating indexes to reorder this data frame
hldf['newhIdx'] = hindex.astype(int)


tlist1 = maindf['Player'].value_counts(sort= False).to_dict()						# counts the frequency of each player value in synthetic data
tlist3 = maindf['Player'].value_counts(normalize = True, sort= False).to_dict()		# counts the relative frequency of each player value
tlist2 = maindf['Hole'].value_counts(sort=False).to_dict()							# counts the frequency of each hole value
tlist4 = maindf['Hole'].value_counts(normalize = True,sort= False).to_dict()		# counts the relative frequency of each hole value

plydf['Freq'] = plydf['Player'].map(tlist1)							# maps above frequencies to data frame - will use for weighted avg.
plydf['relfreq'] = plydf['Player'].map(tlist3)
hldf['Freq'] = hldf['Hole'].map(tlist2)
hldf['relfreq'] = hldf['Hole'].map(tlist4)


plyr_rescale = np.array(plydf['Factor'])
#plyr_mean = np.mean(plydf['Factor'])								# un-weighted mean
plyr_mean = np.average(plydf['Factor'], weights=plydf['Freq'])		# weighted mean
plyr_rescale = plyr_rescale - plyr_mean
plydf['Factor'] = plyr_rescale										# re-scaled to center the weighted mean at 0
plydf = plydf.sort_values(by='newpIdx')								# sorted by index will get df in same order as the "true value" dataframe


hole_rescale = np.array(hldf['Factor'])
#hole_mean = np.mean(hldf['Factor'])								# un-weighted mean
hole_mean = np.average(hldf['Factor'], weights=hldf['Freq'])		# weighted mean
hole_rescale = hole_rescale - hole_mean	
hldf['Factor'] = hole_rescale										# re-scaled to center the weighted mean at 0
hldf = hldf.sort_values(by='newhIdx')								# sorted by index will get df in same order as the "true value" dataframe


pRMSE = sqrt(mean_squared_error(pDF['p_avg'], plydf['Factor']))			# taking RMSE from predicted values compared to our "true values" set out in first file
print(f"Root mean squared error for the player average is {pRMSE}")

hRMSE = sqrt(mean_squared_error(hDF['h_avg'], hldf['Factor']))			# taking RMSE from predicted values compared to our "true values" set out in first file
print(f"Root mean squared error for the hole average is {hRMSE}")