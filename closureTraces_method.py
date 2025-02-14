from __future__ import absolute_import
import numpy as np
import os, sys
import itertools
import gc

mypath = os.path.dirname(__file__)
sys.path.append(mypath)

import _closureTraces as CppTraces # import C++ script to compute cl traces

USE_32BIT_VARIABLES = True ## Set to False for 64-bit types
if USE_32BIT_VARIABLES:
	dTypeVis = np.complex64 # type for the visibilities (complex numbers)
	dType = np.float32 # type for other arrays
else:
	print(
        "WARNING: Using 64-bit types (complex128 and float64).\n"
        "Ensure the C++ script is updated to use consistent types (setting #define USE_32BIT_VARIABLES 0).\n"
        "To switch to 32-bit types, set USE_32BIT_VARIABLES = True."
    )
	dTypeVis = np.complex128
	dType = np.float64

class closureTraces(object):
	""" 
	Reads a fits file (or measurement set) and initialized the C++ engine to compute closure traces. 
	
	Initialization arguments:
		-- visname :: Name of the fits file or the measurement set.
		-- antennas :: List of codes of the antennas to be used.
							If empty, use all the antennas.
	
	Methods:
		-- getDataTraces(quad=0)
		-- getModelTraces(quad=0)
		-- getChi2()
	
	Variables:
		-- TIME :: MJD in seconds.
		-- UVW :: UVW coordinates (in meters).
		-- nAnt/nBas :: Number of antennas/baselines.
		-- DATA/MODEL :: Data/Model visibilities (arranged as baseline-time-frequency-stokes).
		-- OBSERVED :: boolean array that tells whether the baseline was observing at that time.
		-- output :: array to store the (either data or model) closure traces.
	"""
	
	def __init__(self, visname,antennas=[], cell='1.0muas',npix=256, NCPU=4, saveChiSq=False):
		
		self.NCPU = NCPU
		self.saveChiSq = 1 if saveChiSq else 0
		
		if os.path.isfile(visname):
			from astropy.io import fits as pf
			## Open UVFITS:  
			try:
				temp = pf.open(visname)  # fits object
			except:
				raise Exception("ERROR: Problem with UVFITS file: %s"%os.path.basename(uvfits))
			
			############ Set Correlation names and indices ############
			STOKES_IDI = {1:'I',2:'Q',3:'U',4:'V',-1:'RR',-2:'LL',-3:'RL',-4:'LR',-5:'XX',-6:'YY',-7:'XY',-8:'YX'}
			
			# find stokes parameters
			for key in temp['PRIMARY'].header.keys():
				if temp['PRIMARY'].header[key]=='STOKES':
					Stk0 = int(temp['PRIMARY'].header['CRVAL'+key[-1]])
					DelStk = int(temp['PRIMARY'].header['CDELT'+key[-1]])
					RefStk = int(temp['PRIMARY'].header['CRPIX'+key[-1]])
					NStk = int(temp['PRIMARY'].header['NAXIS'+key[-1]])
					break
			if NStk!=4:
				raise Exception('Not a full polarization dataset!')
			
			StkVector = [Stk0+DelStk*(i-RefStk) for i in range(1,NStk+1)]
			self.corrNames = [STOKES_IDI[i] for i in StkVector]
			
			## Assign unique indices to the pol. products:
			if 'XX' in self.corrNames: # linear correlators
				iRR = self.corrNames.index('XX')
				iRL = self.corrNames.index('XY')
				iLR = self.corrNames.index('YX')
				iLL = self.corrNames.index('YY')
				self.isRL = 0
			else:								# circular correlators
				iRR = self.corrNames.index('RR')
				iRL = self.corrNames.index('RL')
				iLR = self.corrNames.index('LR')
				iLL = self.corrNames.index('LL')
				self.isRL = 1
			
			stkOrder = [iRR,iLL,iRL,iLR] ## ORDER IS: RR, LL, RL, LR (or XX, YY, XY, YX)
			
			################## Antennas ##################
			# find antenna names
			Ain = -1
			for i in range(len(temp)):
				if temp[i].name=='AIPS AN':
					Ain = i
					break
			if Ain<0:
				raise Exception('There is no antenna table')
			
			self.antNames = [str(i) for i in temp[Ain].data['ANNAME']] # get antenna names
			## Figure out list of usable antennas:
			self.usedAntennas = []
			if len(antennas)==0: # if empty, use all antennas
				self.usedAntennas = [str(ai) for ai in self.antNames]
			else:
				for ai in antennas:
					if ai not in self.antNames:
						raise Exception("Antenna %s not in measurement set!"%ai)
					self.usedAntennas.append(ai)
			
			self.nAnt = len(self.usedAntennas)			# number of used antennas
			self.nBas = self.nAnt*(self.nAnt-1)//2		# number of baselines
			
			## Ordered list of baselines:
			BASID = []
			for i in range(self.nAnt-1):			# from i = ant0 to i = ant(N-1)
				for j in range(i+1,self.nAnt):		# from ant_i to antN
					BASID.append([i,j,i*256+j])			# all baselines
			self.BASID = np.array(BASID,dtype=np.int32)
			## BASID is a 1-D list organized by antenna and baseline, i.e.:
			## [ant0_0Bas0,...,ant0_Last0Bas, ant1_1Bas0,...,ant1_Last1Bas, ..., ..., ant(N-1)_Nbas]
			
			############ Frequencies and channels ############
			# get reference frequency
			# for all the keys of the header of the PRIMARY data table (metadata to the data) in the fits file
			for key in temp['PRIMARY'].header.keys():
				if temp['PRIMARY'].header[key]=='FREQ': # get the reference frequency for each IF
					RefFrec = float(temp['PRIMARY'].header['CRVAL'+key[-1]])
					break
			
			# find freq table
			Fin = -1
			for i in range(len(temp)):
				if temp[i].name=='AIPS FQ':
					Fin = i
					break
			if Fin<0:
				raise Exception('There is no frequency table')
				# 'IF FREQ': offset with respect the ref freq of each IF
			if type(temp[Fin].data['CH WIDTH'][0]) is np.float32:
				Width_chan = temp[Fin].data['CH WIDTH'][0]	# width of the freq channels
				Width_total = temp[Fin].data['TOTAL BANDWIDTH'][0]	# total bandwidth
				FREQS = RefFrec + temp[Fin].data['IF FREQ']	# frequency of the first channel of each IF
			else:
				Width_chan = temp[Fin].data['CH WIDTH'][0][0]
				Width_total = temp[Fin].data['TOTAL BANDWIDTH'][0][0]
				FREQS = RefFrec + temp[Fin].data['IF FREQ'][0]
			NCHAN = int(Width_total/Width_chan)	# number of channels
			NIF = len(FREQS)	# number of IFs
			
			## All spws are joined into one array
			self.nSpw  = NIF
			self.nChan = NIF*NCHAN			# total number of channels
			self.FREQUENCY = np.zeros(self.nChan,dtype=dType)
			self.TWOPIFREQ = np.zeros(self.nChan,dtype=dType)
			for i in range(NIF):
				self.FREQUENCY[i*NCHAN:(i+1)*NCHAN] = FREQS[i] + np.linspace(0,Width_total,NCHAN)
			self.TWOPIFREQ[:] = 2*np.pi*self.FREQUENCY[:]
			## FREQUENCY is an 1-D array organized by spw and chan, i.e.:
			## [spw0_ch0,...,spw0_chn, spw1_ch0,...,spw1_chn, ..., ..., spwN_ch0,...,spwN_chn]
			
			##################     DATA     ##################
			TimeId = []
			EntryNames = [k for k in filter(lambda x: 'PTYPE' in x, temp['PRIMARY'].header.keys())] # filter header keys
			for key in EntryNames:
				if temp['PRIMARY'].header[key].startswith('UU'):
					UId = int(key[-1])-1
				if temp['PRIMARY'].header[key].startswith('VV'):
					VId = int(key[-1])-1
				if temp['PRIMARY'].header[key].startswith('BASELINE'):
					BasId = int(key[-1])-1
				if temp['PRIMARY'].header[key].startswith('DATE'):
					TimeId.append(int(key[-1])-1)
			
			true_BASID = []
			for i in range(len(self.usedAntennas)-1):
				for j in range(i+1,len(self.usedAntennas)):
					I=self.antNames.index(self.usedAntennas[i])+1
					J=self.antNames.index(self.usedAntennas[j])+1
					true_BASID.append(256*I+J)
			
			### number of used visibilities
			nvis = len(temp['PRIMARY'].data)
			Usedvis = []
			for n in range(nvis):
				if temp['PRIMARY'].data[n][BasId] in true_BASID:
					Usedvis.append(n)
			nUsedvis = len(Usedvis)
			print('Will load %i visibilities'%nUsedvis)
			uvw = np.zeros((nUsedvis,3),dtype=dType)
			a1 = np.zeros(nUsedvis,dtype=np.int32)
			a2 = np.zeros(nUsedvis,dtype=np.int32)
			time = np.zeros(nUsedvis,dtype=np.float64)
			visib = np.zeros((nUsedvis,self.nChan,4),dtype=dTypeVis)
			weights = np.zeros((nUsedvis,self.nChan,4),dtype=dType)
			
			time0 = temp['PRIMARY'].data[0][TimeId[0]]
			self.usedviscopy = np.copy(Usedvis)
			for vi,vis in enumerate(Usedvis):
				datum = temp['PRIMARY'].data[vis]
				uvw[vi,0] = datum[UId]
				uvw[vi,1] = datum[VId]
				CombBas = int(datum[BasId])
				a1[vi] = int(CombBas//256)-1 
				a2[vi] = int(CombBas%256)-1
				time[vi] = (datum[TimeId[0]]-time0) + datum[TimeId[1]]
				#print(a1[vi],a2[vi],time[vi],datum[-1][0,0,0,:,:,0])
				for spi in range(NIF): 		#[0,0,IF,CHAN,corr,RE|IM|WEIGHT]
					temp_vis = datum[-1][0,0,spi,:,:,0]+datum[-1][0,0,spi,:,:,1]*1.j
					## ORDER IS: RR, LL, RL, LR (or XX, YY, XY, YX)
					visib[vi,spi*NCHAN:(spi+1)*NCHAN,0] = temp_vis[:,iRR]
					visib[vi,spi*NCHAN:(spi+1)*NCHAN,1] = temp_vis[:,iLL]
					visib[vi,spi*NCHAN:(spi+1)*NCHAN,2] = temp_vis[:,iRL]
					visib[vi,spi*NCHAN:(spi+1)*NCHAN,3] = temp_vis[:,iLR]
					
					temp_weights = datum[-1][0,0,spi,:,:,2]
					weights[vi,spi*NCHAN:(spi+1)*NCHAN,0] = temp_weights[:,iRR]
					weights[vi,spi*NCHAN:(spi+1)*NCHAN,1] = temp_weights[:,iLL]
					weights[vi,spi*NCHAN:(spi+1)*NCHAN,2] = temp_weights[:,iRL]
					weights[vi,spi*NCHAN:(spi+1)*NCHAN,3] = temp_weights[:,iLR]
				if a1[vi] == a2[vi]:
					weights[vi,:,:] = 0.0
			
			del temp,temp_vis,temp_weights
			
		##################### FOR .ms (USING CASA) ######################
		else:
			from casatools import ms as ms_casa
			from casatools import table as tb_casa
			from casatools import image as ia_casa
			
			tb = tb_casa()
			ms = ms_casa()
			ia = ia_casa()
			
		## Read correlation products:
			ms.open(visname)
			corrs = ms.range('CORR_NAMES')['corr_names']
			self.corrNames = [corrs[i][0] for i in range(len(corrs))]
			if len(corrs)!=4:
				raise Exception("Not a full-polarization dataset!")
			ms.close()
			
		## Assign unique indices to the pol. products:
			if 'XX' in self.corrNames: # linear correlators
				iRR = self.corrNames.index('XX')
				iRL = self.corrNames.index('XY')
				iLR = self.corrNames.index('YX')
				iLL = self.corrNames.index('YY')
				self.isRL = 0
			else:								# circular correlators
				iRR = self.corrNames.index('RR')
				iRL = self.corrNames.index('RL')
				iLR = self.corrNames.index('LR')
				iLL = self.corrNames.index('LL')
				self.isRL = 1
			
			stkOrder = [iRR,iLL,iRL,iLR] ## ORDER IS: RR, LL, RL, LR (or XX, YY, XY, YX)
			
		## Read antenna codes/names:  
			tb.open(os.path.join(visname,"ANTENNA"))
			self.antNames = list(tb.getcol("NAME"))
			tb.close()
			
		## Read frequencies:
			tb.open(os.path.join(visname,"SPECTRAL_WINDOW"))
			Nus = tb.getcol("CHAN_FREQ")
			tb.close()
			NCHAN,NIF = np.shape(Nus)		# number of channels and spws: Nus = [ch,spw]
			
		## All spws are joined into one array
			self.nSpw  = NIF
			self.nChan = NIF*NCHAN			# total number of channels
			self.FREQUENCY = np.zeros(self.nChan,dtype=dType)
			self.TWOPIFREQ = np.zeros(self.nChan,dtype=dType)
			for i in range(NIF):
				self.FREQUENCY[i*NCHAN:(i+1)*NCHAN] = Nus[:,i]
			self.TWOPIFREQ[:] = 2*np.pi*self.FREQUENCY[:]
		## FREQUENCY is an 1-D array organized by spw and chan, i.e.:
			## [spw0_ch0,...,spw0_chn, spw1_ch0,...,spw1_chn, ..., ..., spwN_ch0,...,spwN_chn]
			
		## Figure out list of usable antennas:
			self.usedAntennas = []
			if len(antennas)==0: # if empty, use all antennas
				self.usedAntennas = [str(ai) for ai in self.antNames]
			else:
				for ai in antennas:
					if ai not in self.antNames:
						raise Exception("Antenna %s not in measurement set!"%ai)
					self.usedAntennas.append(ai)
			
			self.nAnt = len(self.usedAntennas)			# number of used antennas
			self.nBas = self.nAnt*(self.nAnt-1)//2		# number of baselines
			
		## Ordered list of baselines:
			BASID = []
			for i in range(self.nAnt-1):			# from i = ant0 to i = ant(N-1)
				for j in range(i+1,self.nAnt):		# from ant_i to antN
					BASID.append([i,j,i*256+j])			# all baselines
			self.BASID = np.array(BASID,dtype=np.int32)
		## BASID is a 1-D list organized by antenna and baseline, i.e.:
			## [ant0_0Bas0,...,ant0_Last0Bas, ant1_1Bas0,...,ant1_Last1Bas, ..., ..., ant(N-1)_Nbas]
			
		## Read data and metadata:
			tb.open(visname)
			msvisib  = tb.getcol("DATA")
			msweight = tb.getcol('WEIGHT')
			ms_uvw   = tb.getcol("UVW")/2.99792458e8
			ms_a1    = tb.getcol("ANTENNA1")
			ms_a2    = tb.getcol("ANTENNA2")
			mstime   = tb.getcol("TIME")
			spw      = tb.getcol("DATA_DESC_ID")
			tb.close()
			
		## Initialize and fill data arrays, separating the spws
			nUsedvis = 0
			refspi   = 0
			badspw   = []
			goodspw  = []
			UsedVisSPW = {}
			for spi in np.unique(spw):
				tempUsedvis = []
				for i in range(len(self.usedAntennas)-1):
					for j in range(i+1,len(self.usedAntennas)):
						I = self.antNames.index(self.usedAntennas[i])
						J = self.antNames.index(self.usedAntennas[j])
						truevis = np.where(np.logical_and(np.logical_and(ms_a1==I,ms_a2==J),spw==spi))[0]
						tempUsedvis.extend(truevis)
						#print(I,J,len(tempUsedvis))
				print("spw %i: %i visibilities"%(spi,len(tempUsedvis)))
				UsedVisSPW[spi] = tempUsedvis
				# Compare visibilities and adjust goodspw/badspw
				if spi == np.unique(spw)[0]:  # First iteration setup
					print("First iteration: reference spw %i"%spi)
					goodspw.append(spi)
					nUsedvis = len(tempUsedvis)
					Usedvis = np.array(tempUsedvis)
					refspi = spi
				else:
					print()
					if len(tempUsedvis) == nUsedvis:
						goodspw.append(spi)
					elif len(tempUsedvis) > nUsedvis:
						nUsedvis = len(tempUsedvis)
						Usedvis = np.array(tempUsedvis)
						refspi = spi
						print('\nNew reference SPW: spw%i!\n'%spi)
						badspw.extend(goodspw)  # Move current good spws to bad
						goodspw = [spi]
					else:
						badspw.append(spi)
					
					if len(tempUsedvis) != nUsedvis:
						print('\nWARNING! Uneven number of visibilities for spw%i!\n'%spi)
			
			print('Will load %i visibilities'%(nUsedvis*(spi+1)))
			uvw     = np.zeros((nUsedvis,3),dtype=dType)
			a1      = np.zeros(nUsedvis,dtype=np.int32)
			a2      = np.zeros(nUsedvis,dtype=np.int32)
			time    = np.zeros(nUsedvis,dtype=np.float64)
			visib   = np.zeros((nUsedvis,self.nChan,4),dtype=dTypeVis)
			weights = np.zeros((nUsedvis,self.nChan,4),dtype=dType)
			
			### Fill spw independent info. using the reference antenna
			for vi,vis in enumerate(Usedvis):
				uvw[vi,0] = ms_uvw[0,vis]
				uvw[vi,1] = ms_uvw[1,vis]
				a1[vi]    = ms_a1[vis]
				a2[vi]    = ms_a2[vis]
				time[vi]  = mstime[vis]
				weights[vi,:,0] = msweight[iRR,vis]
				weights[vi,:,1] = msweight[iLL,vis]
				weights[vi,:,2] = msweight[iRL,vis]
				weights[vi,:,3] = msweight[iLR,vis]
				if a1[vi] == a2[vi]:
					weights[vi,:,:] = 0.0
			### Fill the visibilities: first of full spws
			for spi in goodspw:
				Usedvis = UsedVisSPW[spi]
				for vi,vis in enumerate(Usedvis):
					## ORDER IS: RR, LL, RL, LR (or XX, YY, XY, YX)
					visib[vi,spi*NCHAN:(spi+1)*NCHAN,0] = msvisib[iRR,:,vis]
					visib[vi,spi*NCHAN:(spi+1)*NCHAN,1] = msvisib[iLL,:,vis]
					visib[vi,spi*NCHAN:(spi+1)*NCHAN,2] = msvisib[iRL,:,vis]
					visib[vi,spi*NCHAN:(spi+1)*NCHAN,3] = msvisib[iLR,:,vis]
			### Fill visibilities of incomplete spws
			for spi in badspw:
				print("\n\n  WARNING: will not use spw%i: incomplete dataset! \n\n"%spi)
			### TODO TODO: implement a way to recover the data. Attempted, but not working...
			#	print("\nDoing incomplete spw%i"%spi)
			#	for i in range(len(self.usedAntennas)-1):
			#		print("Doing baselines with antenna %s"%(self.usedAntennas[i]))
			#		for j in range(i+1,len(self.usedAntennas)):
			#			I = self.antNames.index(self.usedAntennas[i])
			#			J = self.antNames.index(self.usedAntennas[j])
			#			for ti in np.unique(time):
			#				baslmask = np.logical_and(ms_a1==I,ms_a2==J)
			#				spwmask  = np.logical_and(baslmask,spw==spi)
			#				timemask = np.logical_and(spwmask,mstime==ti)
			#				truevis  = np.where(timemask)[0]
			#				if len(truevis) != 0:
			#					baslmask = np.logical_and(a1==I,a2==J)
			#					timemask = np.logical_and(baslmask,time==ti)
			#					vi_indx = np.where(timemask)[0]
			#					## ORDER IS: RR, LL, RL, LR (or XX, YY, XY, YX)
			#					visib[vi_indx,spi*NCHAN:(spi+1)*NCHAN,0] = msvisib[iRR,:,vis]
			#					visib[vi_indx,spi*NCHAN:(spi+1)*NCHAN,1] = msvisib[iLL,:,vis]
			#					visib[vi_indx,spi*NCHAN:(spi+1)*NCHAN,2] = msvisib[iRL,:,vis]
			#					visib[vi_indx,spi*NCHAN:(spi+1)*NCHAN,3] = msvisib[iLR,:,vis]
			#	del baslmask, spwmask, timemask, truevis
			
			## clear ms variables (rearranged separating spws)
			del msvisib, msweight, ms_uvw, ms_a1, ms_a2, mstime, spw, tempUsedvis, Usedvis, UsedVisSPW
		
	#################################################################
	## Integration times (unique set):
		self.TIME = np.unique(time)
		self.nTime = len(self.TIME)
		
		print('\nNumber of integration times:  %i'%self.nTime)
		print('Number of used antennas:  %i'%self.nAnt)
		print('Number of channels: %i (in %i spws)\n'%(self.nChan,NIF))
		
	## Prepare memory to store the data.
	# Dimensions for DATA/MODEL: [BASELINE_ID][TIME, FREQ_CHANNEL, CORR]
		self.DATA = [np.require(np.zeros((self.nTime,self.nChan,4),dtype=dTypeVis),requirements=['C','A']) for i in range(self.nBas)]
		self.MODEL = [np.require(np.zeros((self.nTime,self.nChan,4),dtype=dTypeVis),requirements=['C','A']) for i in range(self.nBas)]
	# Dimensions for UVW coords: [BASELINE_ID][TIME,uvw]
		self.UVW = [np.require(np.zeros((self.nTime,3),dtype=dType),requirements=['C','A']) for i in range(self.nBas)]
	# Dimensions for WEIGHTS: [BASELINE_ID][TIME, FREQ_CHANNEL]
		self.WEIGHTS = [np.require(np.zeros((self.nTime,self.nChan),dtype=dType),requirements=['C','A']) for i in range(self.nBas)]
		
	## Boolean that tells whether a baseline has observed a given integration time:
		# Dimensions for OBSERVED: [BASELINE_ID][TIME]
		self.OBSERVED = [np.require(np.zeros(self.nTime,dtype=bool),requirements=['C','A']) for i in range(self.nBas)]
		
	## Array to be filled with closure traces for a given quadruplet:
		# This array is the output of the getDataTraces C++ function
		# Dimensions for output: [BASELINE_ID][TIME, FREQ_CHANNEL]
		self.output = np.require(np.zeros((self.nTime,self.nChan),dtype=dTypeVis),requirements=['C','A'])
		
		self.output_WEIGHT = np.require(np.zeros((self.nTime,self.nChan),dtype=dType),requirements=['C','A'])
		
		timeMask = np.zeros(len(time),dtype=bool)		# mask for all integration times
		baselineMask = [np.copy(timeMask) for i in range(self.nBas)]	# each baseline observes all times 
		
		for i in range(self.nBas):		# mask for baselines: true for baselines of our selected antennas
			a1id = self.antNames.index(self.usedAntennas[self.BASID[i][0]])
			a2id = self.antNames.index(self.usedAntennas[self.BASID[i][1]])
			baselineMask[i][:] = np.logical_and(a1==a1id, a2==a2id)
			#print(a1id,a2id,np.sum(baselineMask[i]))
		
		self.baselineMaskcopy = [np.copy(bi) for bi in baselineMask]
		self.timecopy = np.copy(time)
		
		for ti,t in enumerate(self.TIME):    
			if ti%128==0:
				sys.stdout.write('\r Arranging time %i of %i'%(ti,self.nTime))
				sys.stdout.flush()
			
			timeMask[:] = time==t  # time mask for each unique integration times
			#print('for time %i: %i'%(ti,np.sum(timeMask)))
			for bi in range(self.nBas):
				datum = np.where(np.logical_and(timeMask,baselineMask[bi]))[0]
				self.OBSERVED[bi][ti] = False
				#if len(datum)>1:
				#	print(datum)
				for di in datum:   # for each unique integration time, for baselines of selected antennas
					if np.max(weights[di,:,:])>0.0:
						self.OBSERVED[bi][ti] = True
						self.UVW[bi][ti,:] = uvw[di,:]
						self.DATA[bi][ti,:,:] += visib[di,:,:]
						self.WEIGHTS[bi][ti,:] = 1.0 ## TODO: Set proper weights!
		
		## clear variables: free up RAM
		baselineMask.clear()
		del timeMask
		del visib, weights, uvw, a1, a2, time
		gc.collect()
		
	### End of data arrangement.
	#################################################################
	
	### Generate the list of quadruplets:
	## The baseline indices of each quadruplet will be stored here:
		self.QUADRUPLETS  = []
		
	## The antenna indices of each quadruplet will be stored here:
		self.quadAntennas = []
		
	## Fill the quadruplet information:
		for quad in itertools.combinations(range(self.nAnt),4):
			b01 = np.where(np.logical_and(self.BASID[:,0]==quad[0], self.BASID[:,1]==quad[1]))[0]
			b02 = np.where(np.logical_and(self.BASID[:,0]==quad[0], self.BASID[:,1]==quad[2]))[0]
			b03 = np.where(np.logical_and(self.BASID[:,0]==quad[0], self.BASID[:,1]==quad[3]))[0]
			b12 = np.where(np.logical_and(self.BASID[:,0]==quad[1], self.BASID[:,1]==quad[2]))[0]
			b13 = np.where(np.logical_and(self.BASID[:,0]==quad[1], self.BASID[:,1]==quad[3]))[0]
			b23 = np.where(np.logical_and(self.BASID[:,0]==quad[2], self.BASID[:,1]==quad[3]))[0]
			self.QUADRUPLETS.append(np.array([b01,b02,b03,b12,b13,b23], dtype=np.int32))
			self.quadAntennas.append(list(quad))
		
	##################
	## Initialize the closureTrace C++ module:
		CppTraces.setData(self.TIME, self.UVW, self.TWOPIFREQ, self.DATA, self.WEIGHTS, self.MODEL, self.OBSERVED, self.QUADRUPLETS, self.output, self.output_WEIGHT, self.NCPU, self.isRL, self.saveChiSq)
		
		
	####################################
	## Definition of functions to call functions within the C++ module
	####################################
	## Initialize RA and declination arrays
		UNITS = ['muas','mas','as','arcsec','deg','rad']
		SIZES = {'arcsec': np.pi/180./3600.,
					'as': np.pi/180./3600.,   
					'mas': np.pi/180./3600./1000.,
					'muas': np.pi/180./3600./1.e6,
					'deg': np.pi/180.,
					'rad': 1.0}
		# Get pixel size:
		for un in UNITS:
			if cell.endswith(un):
				VAL = cell.split(un)[0]
				try:
				  pixSize = float(VAL)*SIZES[un]
				  break
				except:
				  raise Exception("ERROR getting cell size of %s %s"%(VAL,un))
		
		self.RAs = np.linspace(pixSize*(npix-1)/2., -pixSize*(npix-1)/2.,npix).astype(dType)
		self.DECs = np.linspace(-pixSize*(npix-1)/2., pixSize*(npix-1)/2.,npix).astype(dType)
		self.RAs  -= pixSize/2.
		self.DECs += pixSize/2.
		#print((self.RAs[npix//2-1])/pixSize)
		#print((self.DECs[npix//2-1])/pixSize)
		#print((self.RAs[npix//2-1-50])/pixSize)
		#print((self.DECs[npix//2-1+20])/pixSize)
		
	## Initialize model Images for Stokes parameters
		self.I = np.zeros((npix,npix),dtype=dType)
		self.Q = np.zeros((npix,npix),dtype=dType)
		self.U = np.zeros((npix,npix),dtype=dType)
		self.V = np.zeros((npix,npix),dtype=dType)
	
	def loadModel(self,chanlist=[]):
		"""
			Load a full-Stokes Model Image and gets the Model Visibilities
		"""
		LoadChan = np.array(chanlist,dtype=np.int32)
		return CppTraces.loadModel(self.RAs,self.DECs, self.I,self.Q,self.U,self.V, LoadChan)
	
	##################
	def getDataTraces(self,quad=0):
		""" 
			Fills the "output" array with the DATA closure traces of the quadruplet specified by the "quad" index (default =0 is the first quadruplet in the list).
			"quad" can also be a list of four antenna codes.
		"""   
		try:
			qi = int(quad)
		except:
			try:
				ids = sorted([self.usedAntennas.index(q) for q in quad])
				qi = self.quadAntennas.index(ids)
			except:
				raise Exception("Wrong quadruplet!")
		toPrint = self.quadAntennas[qi]
		print("Processing quadruplet: %s-%s-%s-%s"%tuple(toPrint))
		CppTraces.getDataTraces(qi)
	
	def getModelTraces(self,quad=0):
		""" 
			Fills the "output" array with the MODEL closure traces of the quadruplet specified by the "quad" index (default =0 is the first quadruplet in the list).
		"""   
		try:
			qi = int(quad)
		except:
			try:
				ids = sorted([self.antNames.index(q) for q in quad])
				qi = self.quadAntennas.index(ids)
			except:
				raise Exception("Wrong quadruplet!")
		toPrint = self.quadAntennas[qi]
		print("Processing quadruplet: %s-%s-%s-%s"%tuple(toPrint))
		CppTraces.getModelTraces(qi)

	##################
	def getChi2(self):
		return CppTraces.getChi2()
	
	####################################

