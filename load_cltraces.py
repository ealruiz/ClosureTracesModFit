import sys, os, shutil
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import emcee
import pickle as pk
mypath = os.path.dirname(__file__)
sys.path.append(mypath)
from closureTraces_method import closureTraces

## LATEX FONTS:
#if True:
#  plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#  params = {'text.usetex' : True,
#          'font.size' : 18,
#          'font.family' : 'lmodern',
#          }
#  plt.rcParams.update(params)

############################
####### CONFIGURATION ######
### Number of CPUs for the MCMC
NCPUs = 8
### Number of walkers (for each dimension, min. 3)
NTWD   = 6
### Number of "training" iterations (for each dimension) of the MCMC
NTRAIN = 151
### Number of iterations (for each dimension) of the MCMC
NITER  = 3663
### Fraction of the number of iterations to "burn" in the histogram
BURN_FRAC = 3
### Font size and Number of bins in the histogram
FONT_SIZE = 15
Nbins = 22

### Set cell size of the images
CELL_SIZE = 'muas'
SIZES = {'arcsec': np.pi/180./3600.,
			'as': np.pi/180./3600.,   
			'mas': np.pi/180./3600./1000.,
			'muas': np.pi/180./3600./1.e6,
			'deg': np.pi/180.,
			'rad': 1.0}
FourierFact = 2.*np.pi * SIZES[CELL_SIZE] * 1.j

### Testing script: set source properties and parameters manually.
### Final script: TODO: load Stokes I image. Not know results. Use the Stokes I image to define the parameter space for the MCMC.
case_id = 4

CASES = {
			4:{'case':'Polarized double source: pol + weak pol, no RM',
			'VISname':'Sim4_4spw_2source_pol_weakpol_noRM',
			'I':[1.0, 0.25],
			'Qfrac':[0.3, 0.06], 'Ufrac':[0.15, -0.02],
			'Vfrac':[0.02, -0.001],
			'RM':[0.0, 0.0],
			'spec_index':[0.0, 0.0],
			'RA_offset':[0, 50],'Dec_offset':[0, 20],
         },
			
			5:{'case':'Polarized double source: pol + weak pol, RM',
			'VISname':'Sim5_4spw_2source_pol_weakpol_RM',
			'I':[1.0, 0.25],
			'Qfrac':[0.3, 0.06], 'Ufrac':[0.15, -0.02],
			'Vfrac':[0.02, -0.001],
			'RM':[2.e6, 5.e5],
			'spec_index':[0.0, 0.0],
			'RA_offset':[0, 50],'Dec_offset':[0, 20],
         },
			
			7:{'case':'Polarized double source: pol + weak pol, RM, Dterms',
			'VISname':'Sim7_4spw_2source_pol_weakpol_RM_Dterms',
			'I':[1.0, 0.25],
			'Qfrac':[0.3, 0.03], 'Ufrac':[0.15, 0.04],
			'Vfrac':[0.02, -0.001],
			'RM':[2.e6, 5.e5],
			'spec_index':[0.0, 0.0],
			'RA_offset':[0, 50],'Dec_offset':[0, 20],
         },
			}

CASE = CASES[case_id]

I1 = CASE['I'][0]
I2 = CASE['I'][1]

Q1 = CASE['Qfrac'][0] * I1
Q2 = CASE['Qfrac'][1] * I2
U1 = CASE['Ufrac'][0] * I1
U2 = CASE['Ufrac'][1] * I2
polI1_true = np.sqrt(Q1*Q1 + U1*U1)
polI2_true = np.sqrt(Q2*Q2 + U2*U2)
phi1_true  = 0.5*np.arctan2(U1,Q1)*180./np.pi
EVPA1_true = phi1_true if phi1_true > 0. else phi1_true + 180.
phi2_true  = 0.5*np.arctan2(U2,Q2)*180./np.pi
EVPA2_true = phi2_true if phi2_true > 0. else phi2_true + 180.

V1 = CASE['Vfrac'][0] * I1
V2 = CASE['Vfrac'][1] * I2

RM1_true = CASE['RM'][0]/1.e6
RM2_true = CASE['RM'][1]/1.e5

RM1_true = 0.0001 # To trigger more parameters, let's see if we can make case_id 4 and 6 better...

RAoffset1  = CASE['RA_offset'][0]
Decoffset1 = CASE['Dec_offset'][0]
RAoffset2  = CASE['RA_offset'][1]
Decoffset2 = CASE['Dec_offset'][1]

# Number of dimensions
if RM1_true == 0 and RM2_true == 0:
	NPARAMS = 2
	NCOMPONENTS = len(CASE['I'])
	NDIM = NPARAMS*NCOMPONENTS
else:
	NPARAMS = 3
	NCOMPONENTS = len(CASE['I'])
	NDIM = NPARAMS*NCOMPONENTS
NWALKER = NTWD*NDIM

burnout = NWALKER*NTRAIN + NWALKER*int(NITER/BURN_FRAC)
#########################
### END OF CONFIGURATION
#########################

### Load all data and compute the chi2 temperature sigma
visname = CASE['VISname']

visfile = os.path.join(os.getcwd(),'Data','%s.ms'%visname)
CL_TRACES = closureTraces(visfile,NCPU=1,saveChiSq=True)

### Set reference wavelenght
lambdas = 2.99792458e8/CL_TRACES.FREQUENCY
ref_lambda2 = lambdas[0]*lambdas[0]

### Compute sigma value for the chi2 temperature
CL_TRACES.I[127-RAoffset1,127+Decoffset1] = I1
CL_TRACES.I[127-RAoffset2,127+Decoffset2] = I2
CL_TRACES.loadModel()
myChi2 = CL_TRACES.getChi2()

fname  = "chi2_distribution.dat"
cpname = "%s_chi2_distribution.dat"%visname
shutil.copyfile(fname, cpname)
data = np.loadtxt(cpname)
ChiSqRe = np.array([chi2 for chi2 in data[:,0]])
ChiSqIm = np.array([chi2 for chi2 in data[:,1]])
sigmareal = np.std(ChiSqRe)
sigmaimag = np.std(ChiSqIm)
sigma = (sigmareal+sigmaimag)/2

# Plot chi2 distribution
fig = pl.figure(figsize=(7,6))
fig.subplots_adjust(wspace=0.05,hspace=0.05,right=0.97,left=0.025,top=0.95)

sub1 = fig.add_subplot(121)
sub1.yaxis.tick_right()
histRe = sub1.hist(ChiSqRe,bins=Nbins)
sub1.set_xlabel('Chi2 (Real) distribution',fontsize=FONT_SIZE)
sub1.text(0.75,0.8,'std = %.5f '%sigmareal,fontsize=FONT_SIZE,horizontalalignment='center',transform=sub1.transAxes)
sub1.set_yticks([])

sub2 = fig.add_subplot(122)
sub2.yaxis.tick_right()
histIm = sub2.hist(ChiSqIm,bins=Nbins)
sub2.set_xlabel('Chi2 (Imag.) distribution',fontsize=FONT_SIZE)
sub2.text(0.75,0.8,'std = %.5f '%sigmaimag,fontsize=FONT_SIZE,horizontalalignment='center',transform=sub2.transAxes)
sub2.set_yticks([])

pl.title("Chi2 scaling factor: %.7f"%sigma, fontsize=FONT_SIZE)
pl.savefig("%s_chi2_hist.png"%visname)
pl.close()

del CL_TRACES


### Load all data again, now with more CPUs and without saving the chi2 distribution
visname = CASE['VISname']
CL_TRACES = closureTraces(visfile,NCPU=NCPUs)


### Get closure traces of the data, as a 1D array
DATA_CLTRACES      = []
DATA_CLTRACES_WGT  = []
for qindx,quad in enumerate(CL_TRACES.quadAntennas):
	print("Computing DATA closure traces of quadruplet: %s"%quad)
	CL_TRACES.getDataTraces(qindx)
	# Flatten the 2D arrays to 1D and extend the lists
	DATA_CLTRACES.extend(CL_TRACES.output.flatten())
	DATA_CLTRACES_WGT.extend(CL_TRACES.output_WEIGHT.flatten())

DATA_CLTRACES      = np.array(DATA_CLTRACES)
DATA_CLTRACES_WGT  = np.array(DATA_CLTRACES_WGT)

### Set reference wavelenght
lambdas = 2.99792458e8/CL_TRACES.FREQUENCY
ref_lambda2 = lambdas[0]*lambdas[0]



### Update the model image
CL_TRACES.I[127-RAoffset1,127+Decoffset1] = I1
CL_TRACES.I[127-RAoffset2,127+Decoffset2] = I2
CL_TRACES.loadModel()


### Get closure traces of the model, as a 1D array
MODEL_CLTRACES     = []
MODEL_CLTRACES_WGT = []
for qindx,quad in enumerate(CL_TRACES.quadAntennas):
	print("Computing MODEL closure traces of quadruplet: %s"%quad)
	CL_TRACES.getModelTraces(qindx)
	MODEL_CLTRACES.extend(CL_TRACES.output.flatten())
	MODEL_CLTRACES_WGT.extend(CL_TRACES.output_WEIGHT.flatten())

MODEL_CLTRACES     = np.array(MODEL_CLTRACES)
MODEL_CLTRACES_WGT = np.array(MODEL_CLTRACES_WGT)


'''
##### Function for the MCMC: compute the chi2 of the closure traces, given a set of parameters p

def ResidualsChi2noRM(p, I1,I2, RAoffset1,Decoffset1, RAoffset2,Decoffset2, loadFreqChan):
	""" Returns the residuals of a model, given a list of fitting parameters p for a source at coordinates (RAoffset,Decoffset):
	p0,p2: fractional polarization of components 1,2 at the lowest freq.
	p1,p3: EVPA of components 1,2 at the lowest freq.
	"""
	p0,p1, p2,p3 = p
	print('Current parameter values: \n polI1: %s | EVPA1: %s \n polI2: %s| EVPA2: %s \n'%(p0,p1,p2,p3))
	
	if p0 < 0. or p0 > I1 or p2 < 0. or p2 > I2:
		return -1.e20
	if p1 < 0. or p1 > np.pi or p3 < 0. or p3 > np.pi:
		return -1.e20
	
	# Assing info to the model image pixels (with cell='1.0muas')
	# Q = I * m * cos(2*EVPA) # EVPA = EVPA(nu0) + RM*(lambda^2 - lambda0^2)
	## p1 = EVPA(nu0); p0 = m*I;
	## Q = p0 * cos(2*p1)
	# U = I * m * sin(2*EVPA) # EVPA = EVPA(nu0) + RM*(lambda^2 - lambda0^2)
	## U = p0 * sin(2*p1)
	CL_TRACES.Q[127-RAoffset1,127+Decoffset1] = p0 * np.cos(2.*p1)
	CL_TRACES.U[127-RAoffset1,127+Decoffset1] = p0 * np.sin(2.*p1)
		
	CL_TRACES.Q[127-RAoffset2,127+Decoffset2] = p3 * np.cos(2.*p3)
	CL_TRACES.U[127-RAoffset2,127+Decoffset2] = p3 * np.sin(2.*p3)
	
	## Compute the model visibilities:
	CL_TRACES.loadModel()
	
	# Computing chi2
	myChi2 = CL_TRACES.getChi2()
	return -myChi2/sigma/sigma

def ResidualsChi2(p, I1,I2, RAoffset1,Decoffset1, RAoffset2,Decoffset2, loadFreqChan):
	""" Returns the residuals of a model, given a list of fitting parameters p for a source at coordinates (RAoffset,Decoffset):
	p0,p3: fractional polarization of components 1,2 at the lowest freq.
	p1,p4: EVPA of components 1,2 at the lowest freq.
	p2: RM of component 1 (in units of 1e6 rad/m^2)
	p5: RM of component 2 (in units of 1e5 rad/m^2)
	"""
	p0,p1,p2, p3,p4,p5 = p
	print('Current parameter values: \n polI1: %s | EVPA1: %s | RM1: %s \n polI2: %s| EVPA2: %s | RM2: %s \n'%(p0,p1,p2,p3,p4,p5))
	
	if p0 < 0. or p0 > I1 or p3 < 0. or p3 > I2:
		return -1.e20
	if p1 < 0. or p1 > np.pi or p4 < 0. or p4 > np.pi:
		return -1.e20
	if p2 < -15. or p2 > 15. or p5 < -10. or p5 > 10.:
		return -1.e20
	
	for ch in range(loadFreqChan):
		lambda2 = lambdas[ch]*lambdas[ch]
		# Assing info to the model image pixels (with cell='1.0muas')
		# Q = I * m * cos(2*EVPA) # EVPA = EVPA(nu0) + RM*(lambda^2 - lambda0^2)
		## p0 = EVPA(nu0); p1 = m*I; p2 = RM;
		## q0 = p1*cos(2*p0); q1 = p1*sin(2*p0)
		## Q = q0 * cos(2*p2*(lambda^2 - lambda0^2)) - 
		##	    q1 * sin(2*p2*(lambda^2 - lambda0^2))
		# U = I * m * sin(2*EVPA) # EVPA = EVPA(nu0) + RM*(lambda^2 - lambda0^2)
		## U = q1 * cos(2*p2*(lambda^2 - lambda0^2)) + 
		##	    q0 * sin(2*p2*(lambda^2 - lambda0^2))
		CL_TRACES.Q[127-RAoffset1,127+Decoffset1] = p0 * np.cos(2.*(p1+p2*1.e6*(lambda2-ref_lambda2)))
		CL_TRACES.U[127-RAoffset1,127+Decoffset1] = p0 * np.sin(2.*(p1+p2*1.e6*(lambda2-ref_lambda2)))
		
		CL_TRACES.Q[127-RAoffset2,127+Decoffset2] = p3 * np.cos(2.*(p4+p5*1.e5*(lambda2-ref_lambda2)))
		CL_TRACES.U[127-RAoffset2,127+Decoffset2] = p3 * np.sin(2.*(p4+p5*1.e5*(lambda2-ref_lambda2)))
		
		## Compute the model visibilities:
		CL_TRACES.loadModel(chanlist=[ch])
	
	# Computing chi2
	myChi2 = CL_TRACES.getChi2()
	return -myChi2/sigma/sigma

	
print("\nFirst step:\nLoading constant model. Will be improved with a MCMC search of 3 parameters: EVPA, pol. Intensity and RM of the 2nd source.\n")
for ch in range(CL_TRACES.nChan):
	lambda2 = lambdas[ch]*lambdas[ch]
	# Assign info to the model image pixels (with cell='1.0muas')
	CL_TRACES.I[127-RAoffset1,127+Decoffset1] = I1
	CL_TRACES.I[127-RAoffset2,127+Decoffset2] = I2
	
	## Compute the model visibilities:
	CL_TRACES.loadModel(chanlist=[ch])
	
### MCMC
print("\nBeginning MCMC search:")
if RM1_true == 0 and RM2_true == 0:
	sampler = emcee.EnsembleSampler(NWALKER,NDIM,ResidualsChi2noRM, args = (I1,I2,RAoffset1,Decoffset1,RAoffset2,Decoffset2,CL_TRACES.nChan) )
else:
	sampler = emcee.EnsembleSampler(NWALKER,NDIM,ResidualsChi2, args = (I1,I2,RAoffset1,Decoffset1,RAoffset2,Decoffset2,CL_TRACES.nChan) )

if os.path.exists('%s_MARKOV_RMfit_last_chain.dat'%visname):
	infile = open('%s_MARKOV_RMfit_last_chain.dat'%visname,'rb')
	curr_chain = pk.load(infile)
	infile.close()
else:
	pini = np.zeros((NWALKER,NDIM))
	if RM1_true == 0 and RM2_true == 0:
		pini[:,0] = np.random.random(NWALKER)*I1
		pini[:,1] = np.random.random(NWALKER)*np.pi
		pini[:,2] = np.random.random(NWALKER)*I2
		pini[:,3] = np.random.random(NWALKER)*np.pi
	else:
		pini[:,0] = np.random.random(NWALKER)*I1
		pini[:,1] = np.random.random(NWALKER)*np.pi
		pini[:,2] = np.random.random(NWALKER)*20.-10.
		pini[:,3] = np.random.random(NWALKER)*I2
		pini[:,4] = np.random.random(NWALKER)*np.pi
		pini[:,5] = np.random.random(NWALKER)*20.-10.
	state = sampler.run_mcmc(pini,NTRAIN)
	curr_chain = state.coords

#tic = time.time()

currit = 0
for sample in sampler.sample(curr_chain,iterations=NITER):
	 currit += 1
	 print('ITERATION %i'%currit)
	 MARKOV = np.copy(sampler.flatchain[:,:])
	 output_file = open('%s_MARKOV_aux_RMfit.dat'%visname,'wb')
	 pk.dump(MARKOV,output_file)
	 output_file.close()
	 try:
	   tau = sampler.get_autocorr_time(c=1)
	   autocorr.append([index,np.mean(tau),np.shape(sampler.flatchain)[0]])
	   index += 1
	   converged = np.all(tau*100 < currit)
	   converged &= np.all(np.abs(old_tau-tau)/tau < 0.01)
	   if converged:
	     break
	   old_tau = tau
	 except:
	   pass
#tac = time.time()
#print('\n',tac-tic,'seconds')

DATNAM = '%s_MARKOV_RMfit'%visname
if os.path.exists('%s.dat'%DATNAM):
	infile = open('%s.dat'%DATNAM,'rb')
	MARKOV1 = pk.load(infile)
	infile.close()

	MARKOV2 = np.concatenate([MARKOV1,MARKOV])
	output_file = open('%s.dat'%DATNAM,'wb')
	pk.dump(MARKOV2,output_file)
	output_file.close()
else:
	MARKOV = np.copy(sampler.flatchain[:,:])
	output_file = open('%s.dat'%DATNAM,'wb')
	pk.dump(MARKOV,output_file)
	output_file.close()

MARKOV_BACKUP = np.copy(sample.coords)
output_file = open('%s_MARKOV_RMfit_last_chain.dat'%visname,'wb')
pk.dump(MARKOV_BACKUP,output_file)
output_file.close()


file_data = open('%s.dat'%DATNAM,'rb')
MARKOV = pk.load(file_data)
file_data.close()
with open("%s.txt"%DATNAM, "w") as file:
	if RM1_true == 0 and RM2_true == 0:
		file.write("polI1\tEVPA1\tpolI2\tEVPA2\n")
		for indx,p0 in enumerate(MARKOV[:,0]):
			p1 = MARKOV[:,1][indx]
			p2 = MARKOV[:,2][indx]
			p3 = MARKOV[:,3][indx]
			file.write("%.7f\t%.7f\t%.7f\t%.7f\n"%(p0,p1,p2,p3))
	else:
		file.write("polI1\tEVPA1\tRM1\tpolI2\tEVPA2\tRM2\n")
		for indx,p0 in enumerate(MARKOV[:,0]):
			p1 = MARKOV[:,1][indx]
			p2 = MARKOV[:,2][indx]
			p3 = MARKOV[:,3][indx]
			p4 = MARKOV[:,4][indx]
			p5 = MARKOV[:,5][indx]
			file.write("%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\n"%(p0,p1,p2,p3,p4,p5))

if RM1_true == 0 and RM2_true == 0:
	PI1, phi1 = [], []
	PI2, phi2 = [], []
	with open("%s.txt"%DATNAM, "r") as file:
		lines = file.readlines()
		for line in lines[1:]:
			columns = line.strip().split('\t')
			PI1.append(np.float(columns[0]))
			phi1.append(np.float(columns[1]))
			PI2.append(np.float(columns[2]))
			phi2.append(np.float(columns[3]))
	
	polI1 = np.array(PI1)
	EVPA1 = np.array(phi1)*180./np.pi
	polI2 = np.array(PI2)
	EVPA2 = np.array(phi2)*180./np.pi
	
	#mask
	mask = np.where(np.logical_and(polI2>0.,np.logical_and(polI2<I2,np.logical_and(polI1>0.,polI1<I1,),),))[0]
	mask = mask[burnout:]
	
	polI1_mean  = np.median(polI1[mask])
	polI1_std   = np.std(polI1[mask])
	EVPA1_mean  = np.median(EVPA1[mask])
	EVPA1_std   = np.std(EVPA1[mask])
	polI2_mean  = np.median(polI2[mask])
	polI2_std   = np.std(polI2[mask])
	EVPA2_mean  = np.median(EVPA2[mask])
	EVPA2_std   = np.std(EVPA2[mask])
	
	polI1 = polI1[mask]
	EVPA1 = EVPA1[mask]
	polI2 = polI2[mask]
	EVPA2 = EVPA2[mask]
	
	# Figs
	### Histogram source 1
	fig = pl.figure(figsize=(12,5))
	fig.subplots_adjust(wspace=0.05,hspace=0.25,right=0.97,left=0.025,top=0.97,bottom=0.15)
	
	sub1 = fig.add_subplot(121)
	sub1.yaxis.tick_right()
	histoQ = sub1.hist(polI1,bins=Nbins)
	sub1.plot(np.array([polI1_true,polI1_true]),np.array([0.,np.max(histoQ[0])]),':k')
	sub1.set_xlabel('polI 1 (Jy)',fontsize=FONT_SIZE)
	pl.setp(sub1.get_xticklabels(),'fontsize',FONT_SIZE)
	sub1.text(0.75,0.8,'median = %.5f \n std = %.5f '%(polI1_mean,polI1_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub1.transAxes)
	sub1.set_yticks([])
	sub1.set_xlim(0.,I1)
	
	sub2 = fig.add_subplot(122)
	sub2.yaxis.tick_right()
	histoU = sub2.hist(EVPA1,bins=Nbins)
	sub2.plot(np.array([EVPA1_true,EVPA1_true]),np.array([0.,np.max(histoU[0])]),':k')
	sub2.set_xlabel('EVPA 1 (deg.)',fontsize=FONT_SIZE)
	pl.setp(sub2.get_xticklabels(),'fontsize',FONT_SIZE)
	sub2.text(0.4,0.8,'median = %.5f \n std = %.5f'%(EVPA1_mean,EVPA1_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub2.transAxes)
	sub2.set_yticks([])
	
	pl.savefig(DATNAM+"_histograms_sou1.png")
	pl.close()
	
	### Histograms
	fig = pl.figure(figsize=(12,5))
	fig.subplots_adjust(wspace=0.05,hspace=0.25,right=0.97,left=0.025,top=0.97,bottom=0.15)
	
	sub1 = fig.add_subplot(121)
	sub1.yaxis.tick_right()
	histoQ = sub1.hist(polI2,bins=Nbins)
	sub1.plot(np.array([polI2_true,polI2_true]),np.array([0.,max(histoQ[0])]),':k')
	sub1.set_xlabel('polI 2 (Jy)',fontsize=FONT_SIZE)
	pl.setp(sub1.get_xticklabels(),'fontsize',FONT_SIZE)
	sub1.text(0.75,0.8,'median = %.5f \n std = %.5f '%(polI2_mean,polI2_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub1.transAxes)
	sub1.set_yticks([])
	sub1.set_xlim(0.,I2)
	
	sub2 = fig.add_subplot(122)
	sub2.yaxis.tick_right()
	histoU = sub2.hist(EVPA2,bins=Nbins)
	sub2.plot(np.array([EVPA2_true,EVPA2_true]),np.array([0.,max(histoU[0])]),':k')
	sub2.set_xlabel('EVPA 2 (deg.)',fontsize=FONT_SIZE)
	pl.setp(sub2.get_xticklabels(),'fontsize',FONT_SIZE)
	sub2.text(0.4,0.8,'median = %.5f \n std = %.5f'%(EVPA2_mean,EVPA2_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub2.transAxes)
	sub2.set_yticks([])
	
	pl.savefig(DATNAM+"_histograms_sou2.png")
	pl.close()
	
	### Histograms
	EVPA_diff_true = phi2_true - phi1_true
	EVPA_diff = EVPA2 - EVPA1
	EVPA_diff_mean = np.median(EVPA_diff)
	EVPA_diff_std  = np.std(EVPA_diff)
	fig = pl.figure(figsize=(6,6))
	fig.subplots_adjust(wspace=0.05,hspace=0.25,right=0.97,left=0.025,top=0.97,bottom=0.15)
	
	sub1 = fig.add_subplot(111)
	sub1.yaxis.tick_right()
	histo = sub1.hist(EVPA_diff,bins=Nbins)
	sub1.plot(np.array([EVPA_diff_true,EVPA_diff_true]),np.array([0.,max(histo[0])]),':k')
	sub1.set_xlabel('EVPA 2 - EVPA 1 (deg.)',fontsize=FONT_SIZE)
	pl.setp(sub1.get_xticklabels(),'fontsize',FONT_SIZE)
	sub1.text(0.75,0.8,'median = %.5f \n std = %.5f '%(EVPA_diff_mean,EVPA_diff_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub1.transAxes)
	sub1.set_yticks([])
	
	pl.savefig(DATNAM+"_histograms_EVPAdiff.png")
	pl.close()
else:
	PI1, phi1, RM1t = [], [], []
	PI2, phi2, RM2t = [], [], []
	with open("%s.txt"%DATNAM, "r") as file:
		lines = file.readlines()
		for line in lines[1:]:
			columns = line.strip().split('\t')
			PI1.append(np.float(columns[0]))
			phi1.append(np.float(columns[1]))
			RM1t.append(np.float(columns[2]))
			PI2.append(np.float(columns[3]))
			phi2.append(np.float(columns[4]))
			RM2t.append(np.float(columns[5]))
	
	polI1 = np.array(PI1)
	EVPA1 = np.array(phi1)*180./np.pi
	RM1   = np.array(RM1t)
	polI2 = np.array(PI2)
	EVPA2 = np.array(phi2)*180./np.pi
	RM2   = np.array(RM2t)
	
	#mask
	mask = np.where(np.logical_and(polI2>0.,np.logical_and(polI2<I2,np.logical_and(polI1>0.,polI1<I1,),),))[0]
	mask = mask[burnout:]
	
	polI1_mean  = np.median(polI1[mask])
	polI1_std   = np.std(polI1[mask])
	EVPA1_mean  = np.median(EVPA1[mask])
	EVPA1_std   = np.std(EVPA1[mask])
	RM1_mean = np.median(RM1[mask])
	RM1_std  = np.std(RM1[mask])
	polI2_mean  = np.median(polI2[mask])
	polI2_std   = np.std(polI2[mask])
	EVPA2_mean  = np.median(EVPA2[mask])
	EVPA2_std   = np.std(EVPA2[mask])
	RM2_mean = np.median(RM2[mask])
	RM2_std  = np.std(RM2[mask])
	
	# Figs
	### Histogram source 1
	fig = pl.figure(figsize=(12,5))
	fig.subplots_adjust(wspace=0.05,hspace=0.25,right=0.97,left=0.025,top=0.97,bottom=0.15)
	
	sub1 = fig.add_subplot(131)
	sub1.yaxis.tick_right()
	histoQ = sub1.hist(polI1[mask],bins=Nbins)
	sub1.plot(np.array([polI1_true,polI1_true]),np.array([0.,np.max(histoQ[0])]),':k')
	sub1.set_xlabel('polI 1 (Jy)',fontsize=FONT_SIZE)
	pl.setp(sub1.get_xticklabels(),'fontsize',FONT_SIZE)
	sub1.text(0.75,0.8,'median = %.5f \n std = %.5f '%(polI1_mean,polI1_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub1.transAxes)
	sub1.set_yticks([])
	sub1.set_xlim(0.,I1)
	
	sub2 = fig.add_subplot(132)
	sub2.yaxis.tick_right()
	histoU = sub2.hist(EVPA1[mask],bins=Nbins)
	sub2.plot(np.array([EVPA1_true,EVPA1_true]),np.array([0.,np.max(histoU[0])]),':k')
	sub2.set_xlabel('EVPA 1 (deg.)',fontsize=FONT_SIZE)
	pl.setp(sub2.get_xticklabels(),'fontsize',FONT_SIZE)
	sub2.text(0.4,0.8,'median = %.5f \n std = %.5f'%(EVPA1_mean,EVPA1_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub2.transAxes)
	sub2.set_yticks([])
	
	sub3 = fig.add_subplot(133)
	sub3.yaxis.tick_right()
	histoRM = sub3.hist(RM1[mask],bins=Nbins)
	sub3.plot(np.array([RM1_true,RM1_true]),np.array([0.,np.max(histoRM[0])]),':k')
	sub3.set_xlabel('RM 1 ($10^6 \ rad / m^2$)',fontsize=FONT_SIZE)
	pl.setp(sub3.get_xticklabels(),'fontsize',FONT_SIZE)
	sub3.text(0.75,0.8,'median = %.5f \n std = %.5f'%(RM1_mean,RM1_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub3.transAxes)
	sub3.set_yticks([])
	
	pl.savefig(DATNAM+"_histograms_sou1.png")
	pl.close()
	
	### Histograms
	fig = pl.figure(figsize=(12,5))
	fig.subplots_adjust(wspace=0.05,hspace=0.25,right=0.97,left=0.025,top=0.97,bottom=0.15)
	
	sub1 = fig.add_subplot(131)
	sub1.yaxis.tick_right()
	histoQ = sub1.hist(polI2[mask],bins=Nbins)
	sub1.plot(np.array([polI2_true,polI2_true]),np.array([0.,max(histoQ[0])]),':k')
	sub1.set_xlabel('polI 2 (Jy)',fontsize=FONT_SIZE)
	pl.setp(sub1.get_xticklabels(),'fontsize',FONT_SIZE)
	sub1.text(0.75,0.8,'median = %.5f \n std = %.5f '%(polI2_mean,polI2_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub1.transAxes)
	sub1.set_yticks([])
	sub1.set_xlim(0.,I2)
	
	sub2 = fig.add_subplot(132)
	sub2.yaxis.tick_right()
	histoU = sub2.hist(EVPA2[mask],bins=Nbins)
	sub2.plot(np.array([EVPA2_true,EVPA2_true]),np.array([0.,max(histoU[0])]),':k')
	sub2.set_xlabel('EVPA 2 (deg.)',fontsize=FONT_SIZE)
	pl.setp(sub2.get_xticklabels(),'fontsize',FONT_SIZE)
	sub2.text(0.4,0.8,'median = %.5f \n std = %.5f'%(EVPA2_mean,EVPA2_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub2.transAxes)
	sub2.set_yticks([])
	
	sub3 = fig.add_subplot(133)
	sub3.yaxis.tick_right()
	histoRM = sub3.hist(RM2[mask],bins=Nbins)
	sub3.plot(np.array([RM2_true,RM2_true]),np.array([0.,max(histoRM[0])]),':k')
	sub3.set_xlabel('RM 2 ($10^5 \ rad / m^2$)',fontsize=FONT_SIZE)
	pl.setp(sub3.get_xticklabels(),'fontsize',FONT_SIZE)
	sub3.text(0.75,0.8,'median = %.5f \n std = %.5f'%(RM2_mean,RM2_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub3.transAxes)
	sub3.set_yticks([])
	
	pl.savefig(DATNAM+"_histograms_sou2.png")
	pl.close()
	
	### Histograms
	EVPA_diff_true = phi2_true - phi1_true
	EVPA_diff = EVPA2[mask] - EVPA1[mask]
	EVPA_diff_mean = np.median(EVPA_diff)
	EVPA_diff_std  = np.std(EVPA_diff)
	fig = pl.figure(figsize=(6,6))
	fig.subplots_adjust(wspace=0.05,hspace=0.25,right=0.97,left=0.025,top=0.97,bottom=0.15)
	
	sub1 = fig.add_subplot(111)
	sub1.yaxis.tick_right()
	histo = sub1.hist(EVPA_diff,bins=Nbins)
	sub1.plot(np.array([EVPA_diff_true,EVPA_diff_true]),np.array([0.,max(histo[0])]),':k')
	sub1.set_xlabel('EVPA 2 - EVPA 1 (deg.)',fontsize=FONT_SIZE)
	pl.setp(sub1.get_xticklabels(),'fontsize',FONT_SIZE)
	sub1.text(0.75,0.8,'median = %.5f \n std = %.5f '%(EVPA_diff_mean,EVPA_diff_std),fontsize=FONT_SIZE,horizontalalignment='center',transform=sub1.transAxes)
	sub1.set_yticks([])
	
	pl.savefig(DATNAM+"_histograms_EVPAdiff.png")
	pl.close()
'''
