import pandas as pd 
import numpy as np 

TOTAL_SHOTS = 10000
driver_distance = 20
avg = 295 - driver_distance	# tour average drive
players = 50
holes = 200
prcnt_topplyrs_hardhole = 0.8 # for a hard hole (i.e. top 33% of holes) what percent of top players (top 50%) saturate the data 
prcnt_topplyrs_medhole = 0.4 # same as above for medium
prcnt_topplyrs_easyhole = 0.3 # same as above for easy
prcnt_hard_holes = .3 # what percent of the total shots are at hard holes
prcnt_med_holes = .45 # same as above for medium holes
prcnt_easy_holes = .25 # same as above for easy holes
driver_top_hard = 0.6
driver_bot_hard = 0.4
driver_top_med = 0.7
driver_bot_med = 0.6
driver_top_easy = 0.9
driver_bot_easy = 0.8
sd_error = 10	# set sd for error
np.random.seed(888)

if (prcnt_easy_holes+prcnt_med_holes+prcnt_hard_holes!=1): exit('Percentage of holes must add up to 1')


u = np.random.normal(0, 6, size = players) # players "true value" avg. standardized to tour average drive
u = np.sort(u)[::-1]						# sort the players avg. vector from smallest to largest
h = np.random.normal(0,10, size = holes) 	# hole "true value" avg. standardized to tour average drive
h = np.sort(h)[::-1]						# sort the holes avg. vector from smallest to largest

plist = np.array([f'Player {i}' for i in range(players)]).tolist()
pld = {'Player': plist, 'p_avg': u}
pDF = pd.DataFrame(pld, columns=['Player','p_avg'])		# player DF includes player name and true value to refer back to later
pDF.set_index('Player', inplace = True)	

hlist = np.array([f'Hole {i}' for i in range(holes)]).tolist()
hld = {'Hole': hlist, 'h_avg': h}
hDF = pd.DataFrame(hld, columns=['Hole','h_avg'])		# hole DF includes hole name and true value to refer back to later
hDF.set_index('Hole', inplace = True)



def convert_to_avg(players, holes):	# takes player and hole name vectors as input and returns matching true value vectors
	pscore = []
	hscore = []
	for player in players:
		pscore.append(pDF.loc[player, 'p_avg'])
	for hole in holes:
		hscore.append(hDF.loc[hole, 'h_avg'])
	pvect = np.asarray(pscore)
	hvect = np.asarray(hscore)
	return pvect, hvect


def calc_holes(difficulty, nholes): # used this to fix problem of uniform distribution across holes and players
	
	if difficulty == 'hard':		# given adjustable parameters at top, calculates a semi-realistic scenario for hole and player distributions
		mix = prcnt_topplyrs_hardhole
		d_t_mix = driver_top_hard
		d_b_mix = driver_bot_hard
		holebydif = hld['Hole'][(round(holes*0/3)):(round(holes*1/3))]  
	elif difficulty == 'medium':
		mix = prcnt_topplyrs_medhole
		d_t_mix = driver_top_med
		d_b_mix = driver_bot_med
		holebydif = hld['Hole'][(round(holes*1/3)):(round(holes*2/3))]
	elif difficulty == 'easy':
		mix = prcnt_topplyrs_easyhole
		d_t_mix = driver_top_easy
		d_b_mix = driver_bot_easy
		holebydif = hld['Hole'][(round(holes*2/3)):(round(holes*3/3))]
	else: return '\nerror\n'

	topplyrs = np.random.choice(pld['Player'][-(round(players/2)):], round(mix*nholes)) # takes random player from top half for the percent top players of each hole
	d_top = np.random.choice([0, 1], size= round(mix*nholes), p=[1-d_t_mix, d_t_mix])
	botplyrs = np.random.choice(pld['Player'][:round(players/2)], round(nholes-(mix*nholes))) # takes random player from bottom half for the rest of the holes played
	d_bot = np.random.choice([0, 1], size= round(nholes - (mix*nholes)), p=[1-d_b_mix, d_b_mix])
	plyrs = np.append(topplyrs, botplyrs) # each player added is a shot
	drvr = np.append(d_top, d_bot)
	hls = np.random.choice(holebydif, nholes) # random selection of holes in certain category. each selection is a shot

	return plyrs, hls, drvr 	# plyrs and hls are same length. Length is equivalent to the amount of shots we want to create


def mainFun(totalshots, hardholes, medholes, easyholes): # choose how many total shots to create data frame (i.e. rows) and % of each type of hole
	
	hard = round(totalshots*hardholes) # number of hard holes
	med = round(totalshots*medholes)	# number of med holes
	easy = round(totalshots*easyholes)	# number of easy holes
	
	p1, h1, d1 = calc_holes('hard', hard) # returns hard many shots on hard holes
	p2, h2, d2 = calc_holes('medium', med)
	p3, h3, d3 = calc_holes('easy', easy)
	p4 = np.append(p1,p2)
	p = np.append(p4,p3) # length of p will be how many shots the data frame is = totalshots
	
	h4 = np.append(h1,h2)
	h = np.append(h4,h3) # length of h will be how many shots the data frame is = totalshots

	d4 = np.append(d1,d2)
	d = np.append(d4,d3) # length of h will be how many shots the data frame is = totalshots
	
	return p, h, d


plyrvector, holevector, drvrvector = mainFun(TOTAL_SHOTS, prcnt_hard_holes, prcnt_med_holes, prcnt_easy_holes) # putting into a dataframe 
errvector = np.random.normal(0, sd_error, size = TOTAL_SHOTS)
pavgvector, havgvector = convert_to_avg(plyrvector, holevector) # takes player name and hole name to return pavg, havg in order to calculate distance (hidden layers)
distvector = avg + pavgvector + havgvector + errvector + driver_distance*drvrvector  # calculates "actual distance" for synthetic data

maindf = pd.DataFrame({'Distance':distvector, 'Player': plyrvector, 'Hole': holevector, 'P_avg': pavgvector, 'H_avg': havgvector, 'Driver': drvrvector, 'Error': errvector})

maindf.to_csv('synthetic_data.csv') # reading to file to run lfe in R- function not available/as intuitive in python

print(maindf.head())