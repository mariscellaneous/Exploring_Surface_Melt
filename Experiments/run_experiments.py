# IMPORTING MODULES

import time
start_time = time.time()



from netCDF4 import Dataset
import pickle
from pyproj import Proj, transform
import numpy as np
import numpy.matlib
from scipy.interpolate import RegularGridInterpolator
from datetime import date
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from smrt import make_snowpack, make_model, sensor_list, make_ice_column

from datetime import timedelta
import sys

from scipy import ndimage

from scipy.ndimage.filters import uniform_filter

from scipy.ndimage.filters import uniform_filter
from scipy.signal import convolve
from scipy import ndimage

initial_loc = int(sys.argv[1])


first_year_float = int(sys.argv[2]) 

first_month = int(sys.argv[3])

first_day = int(sys.argv[4])


last_year_float = int(sys.argv[5])

last_month = int(sys.argv[6])

last_day = int(sys.argv[7])

# first_year_float = 2012. 
# first_month = 7 
# first_day = 2

# last_year_float = 2019.
# last_month = 5
# last_day = 31

first_year_int = int(first_year_float)
first_month_cut = first_month
first_day_cut = first_day
last_year_int = int(last_year_float)
last_month_cut = last_month
last_day_cut = last_day





AWS13_sites = pickle.load(open('AWS_13_sitesV.p','rb'))

locations = ['AWS18','AWS5','AWS4','AWS11','AWS6','AWS14','AWS17','Dome_C','Point_Barnola','Kohnen','AWS16','AWS15','AWS19','Dome_C_bc','AWS16_bc']

file_loc = [ 0, 1, 2, 3, 4,5 ,6 ,7 ,8,9,10,11,12,7,10]
print('We are running loc # '+str(initial_loc)+' and site '+str(locations[initial_loc])+' in 18H')

days = AWS13_sites['days']
sites = AWS13_sites['sites']
lons = AWS13_sites['lons']
lats = AWS13_sites['lats']
TB18V_all = AWS13_sites['TB_18V']
TB18V_s = TB18V_all[file_loc[initial_loc],:]

TB18V_s[TB18V_s<40] = np.nan
# TB18V_s[np.isnan(TB18V_s)] = -888.



if initial_loc > 12:
    loc = initial_loc-13
    CFM_file = 'gsfc_v1_2_1_domec_aws16_biascorr_50m.nc'
else:
    loc = initial_loc 
    CFM_file = 'gsfc_v1_2_1_10132022_50m.nc'

CFM_object = Dataset(CFM_file)


# IMPORTING MODULES

from netCDF4 import Dataset
import pickle
from pyproj import Proj, transform
import numpy as np
import numpy.matlib
from scipy.interpolate import RegularGridInterpolator
from datetime import date
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from smrt import make_snowpack, make_model, sensor_list, make_ice_column

from datetime import timedelta

AWS13_sites = pickle.load(open('AWS_13_sitesH.p','rb'))

days = AWS13_sites['days']
sites = AWS13_sites['sites']
lons = AWS13_sites['lons']
lats = AWS13_sites['lats']
TB18H_all = AWS13_sites['TB_18H']
TB18H_s = TB18H_all[file_loc[initial_loc],:]



TB18H_s[TB18H_s<40] = np.nan

times = days.copy()
# for day in days:

#     times.append(datetime(int(day), 1, 1) + timedelta(days = (day % 1) * 365))

times=np.asarray(times)
first_year = np.min(days).year
last_year = np.max(days).year

nyears = last_year-first_year+2 


winters = np.arange(0,nyears+1)

melt_detection = np.zeros(TB18H_s.shape)*np.nan

 
thresholdP = []
thresholdZF = []
thresholdT = []

    

years = np.arange(first_year,last_year+1)

for ind,year in enumerate(years):
    if year == 2012:
        # grabbing the start and end index of austral melt season
        start_int = datetime(int(year),6,1)-times[0]
        end_int = datetime(int(year),9,1)-times[0]

        days_winter = ((times>datetime(int(year),6,1))&(times<datetime(int(year),9,1)))

        # Making list of thresholds based on P 
    
        thresholdP.append(np.nanmean(TB18H_s[days_winter])+20.)
    else:
        # grabbing the start and end index of austral melt season
        start_int = datetime(int(year)-1,6,1)-times[0]
        end_int = datetime(int(year)-1,9,1)-times[0]

        days_winter = ((times>datetime(int(year)-1,6,1))&(times<datetime(int(year)-1,9,1)))

        thresholdP.append(np.nanmean(TB18H_s[days_winter])+20.)


    # Making list of all the same thresholds based on ZF 
    thresholdZF.append(np.nanmean(TB18H_s)+30.)  
    
    days_year = ((times>datetime(int(year)-1,4,1))&(times<datetime(int(year),3,31)))

#     # adding a threshold
#     thresholdT.append(np.nanmean(TB18H_s[days_year]) + 30.)
        
  
        

TB_First_filter = np.zeros(years.shape[0])
TB_Second_filter = np.zeros(years.shape[0])
M = np.zeros(years.shape[0])
sigma = np.zeros(years.shape[0])
dayslist3=[]
times = np.asarray(times)
thresholdT=np.zeros(years.shape[0])

for ind,year in enumerate(years):
    start_int2 = datetime(int(year)-1,4,1)
    end_int2 = datetime(int(year),3,31)

    
    correct_points = ((times>start_int2)&(times<end_int2))    

    thresholdT[ind] = np.nanmean(TB18H_s[correct_points])

    TB_First_filter[ind]=np.nanmean(TB18H_s[correct_points][TB18H_s[correct_points]<(thresholdT[ind]+30.)])

    TB_Second_filter[ind]=np.nanmean(TB18H_s[correct_points][TB18H_s[correct_points]<(TB_First_filter[ind]+30.)])

    M[ind]=np.nanmean(TB18H_s[correct_points][TB18H_s[correct_points]<(TB_Second_filter[ind]+30.)])
    
    sigma[ind] = np.nanstd(TB18H_s[correct_points][TB18H_s[correct_points]<(TB_Second_filter[ind]+30.)])
    
    thresholdT[ind] = M[ind]+3.*sigma[ind]
    


for ind,year in enumerate(years):
    if year == 2012:
        thresholdT[ind] = thresholdT[ind+1]
        sigma[ind] = sigma[ind+1]
       
       
                   
                   
                   

melt_thresholdP=np.zeros(TB18H_s.shape)*np.nan
melt_thresholdZF=np.zeros(TB18H_s.shape)*np.nan
melt_thresholdT=np.zeros(TB18H_s.shape)*np.nan
melt_sigmaT = np.zeros(TB18H_s.shape)*np.nan 



        
for nyear,days1 in enumerate(years):

    # Grabbing the indices for the six months before and after austral melt midpoint
    start = datetime(int(years[nyear]-1),6,1)-times[0]
    end = datetime(int(years[nyear]),5,31)-times[0]

    if years[nyear] == 2012:
        melt_thresholdP[0:int(end.days+1)] = thresholdP[nyear]
    else:
        melt_thresholdP[int(start.days):int(end.days+1)] = thresholdP[nyear]
        
    start = datetime(int(years[nyear]-1),4,1)-times[0]
    end = datetime(int(years[nyear]),3,31)-times[0]
    
    melt_thresholdZF[:] = thresholdZF[nyear]
    
    start = datetime(int(years[nyear]-1),4,1)-times[0]
    end = datetime(int(years[nyear]),3,31)-times[0]
    
    if years[nyear] == 2012:
        melt_sigmaT[0:int(end.days+1)] = sigma[nyear]
        melt_thresholdT[0:int(end.days+1)] = thresholdT[nyear]
    else:
        melt_thresholdT[int(start.days):int(end.days+1)] = thresholdT[nyear]
        melt_sigmaT[int(start.days):int(end.days+1)] = sigma[nyear]




# times=np.asarray(times)
# times_new = times[((days>=datetime(first_year_cut,1,1))&(days<datetime(last_year_cut+1,1,1)))]
TB18V_s_new = TB18V_s[((days>=datetime(first_year_int,first_month,first_day))&(days<=datetime(last_year_int,last_month,last_day)))]
TB18H_s_new = TB18H_s[((days>=datetime(first_year_int,first_month,first_day))&(days<=datetime(last_year_int,last_month,last_day)))]

days_new = days[((days>=datetime(first_year_int,first_month,first_day))&(days<=datetime(last_year_int,last_month,last_day)))]

melt_thresholdP_snapped = melt_thresholdP[((days>=datetime(first_year_int,first_month,first_day))&(days<=datetime(last_year_int,last_month,last_day)))]
melt_thresholdZF_snapped = melt_thresholdZF[((days>=datetime(first_year_int,first_month,first_day))&(days<=datetime(last_year_int,last_month,last_day)))]
melt_thresholdT_snapped = melt_thresholdT[((days>=datetime(first_year_int,first_month,first_day))&(days<=datetime(last_year_int,last_month,last_day)))]


dry_snow_tb18h = []
wet_snow_tb18h = []

for point in np.arange(0,len(TB18H_s_new)):
    
    if TB18H_s_new[point]<=melt_thresholdP_snapped[point]:
        dry_snow_tb18h.append(TB18H_s_new[point])
        wet_snow_tb18h.append(-999.)
    elif TB18H_s_new[point]>melt_thresholdP_snapped[point]:
        dry_snow_tb18h.append(-999.)
        wet_snow_tb18h.append(TB18H_s_new[point])
    else:
        dry_snow_tb18h.append(np.nan)
        wet_snow_tb18h.append(np.nan)
        
dry_snow_tb18h_filled = np.asarray(dry_snow_tb18h.copy())
for point in np.arange(0,len(TB18H_s_new)):
    if dry_snow_tb18h[point]==-999:
        dry_snow_tb18h_filled[point-7:point+8]=-999.
    elif np.isnan(dry_snow_tb18h[point]):
        print('glop')      
        dry_snow_tb18h_filled[point] = np.nan



def x_intercept(x0,x1,y0,y1):
    m = (y1-y0)/(x1-x0)
    b=y1-m*x1
    return b

def round_off_rating(number,decimals=4): 
    return np.around(number * 2,decimals=decimals) / 2 

def forward_calc(grain_guess,estimate_tb,thickness,temperature,density,liquid,freq):

    snowpack2 = make_snowpack(thickness,"exponential", density = density, temperature = temperature,corr_length = grain_guess,liquid_water=liquid)
    m = make_model("iba", "dort")
    sensorH = sensor_list.amsr2(freq+'H')
    resH = m.run(sensorH, snowpack2)
#     print('18H brightness temp ',resH.TbH())
    difference = resH.TbH()-estimate_tb
    return resH.TbH(),difference

def calc_bounds(init_bounds,estimate_tb,thickness,temperature,density,freq):
    gr0 = density.copy()*np.nan
    gr1 = density.copy()*np.nan
    gr0 = init_bounds[0]
    gr1 = init_bounds[1]


    tb_h0, xx = forward_calc(gr0,estimate_tb,thickness,temperature,density,0,'18')
    tb_h1, xx = forward_calc(gr1,estimate_tb,thickness,temperature,density,0,'18')

    difference0 = tb_h0-estimate_tb
    difference1 = tb_h1-estimate_tb

    if ((difference0>0) & (difference1<0)).all() == False:
        print('Light warning: falls out of bounds of linear interpolation')
    
    offset = np.asarray([difference0,difference1])
    guess = x_intercept(difference0,difference1,gr0,gr1)
    
    tb_hguess, xx = forward_calc(guess,estimate_tb,thickness,temperature,density,0,'18')
    guess_offset = tb_hguess-estimate_tb
    
    return guess, guess_offset

def min_grain_diff(grain_guesses,estimate_tb,thickness,temperature,density,freq):
    difference = grain_guesses.copy()*np.nan
    tb_h = grain_guesses.copy()*np.nan


    for ind,grain_guess in enumerate(grain_guesses):
        tb_h[ind], xx = forward_calc(grain_guess,estimate_tb,thickness,temperature,density,0,'18')
        
        difference[ind] = tb_h[ind]-estimate_tb   
        
    grain_size_idx = np.argmin(np.abs(difference))
    grain_size = grain_guesses[grain_size_idx]
    final_offset = difference[grain_size_idx]
    tbh_final = tb_h[grain_size_idx]
    
    if ((difference[0]>0) & (difference[-1]<0)).any() == False:
        print('out of bounds2')
    return grain_size,final_offset,tbh_final


def calc_grain_size_estimate(initial_bounds,estimate_tb,thickness,temperature=280,density=400,uncertainty=50e-6,freq='18'):
    guess,offset = calc_bounds(initial_bounds,estimate_tb,thickness,temperature,density,freq)
    
    # if guess-uncertainty>0:
    #     grain_guesses = np.linspace(guess-uncertainty,guess+uncertainty,10)
    # else:
    #     grain_guesses = np.linspace(12.5e-s6,guess+uncertainty,10)
    # grain_size,final_offset,TBH=min_grain_diff(grain_guesses,estimate_tb,thickness,temperature,density,freq)
    # guess2,offset2 = calc_bounds(init_y_values2,estimate_tb,thickness,temperature,density,freq)
    return guess,offset

def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)

def calc_grain_size(estimate_tb,thickness,temperature=280,density=400,uncertainty=2.5e-6,freq='18',y_estimate=0):
    init_y_values = [ y_estimate-25e-7, y_estimate+25e-7]
    guess,offset = calc_bounds(init_y_values,estimate_tb,thickness,temperature,density,freq)

    if offset > 0.05:
        init_y_values2 = [guess-5e-7, guess+5e-7]
        guess_final,offset_final = calc_bounds(init_y_values2,estimate_tb,thickness,temperature,density,freq)
        return guess_final,offset_final,guess,offset
    else:
        return guess, offset, [], []
    #     guess_adj = np.around(guess,7)
#     grain_guesses = np.arange(guess_adj-10e-7,guess_adj+10e-7,2e-7)
#     grain_guesses = np.linspace(guess-uncertainty,guess+uncertainty,10)
#     print(grain_guesses)
#     grain_size,final_offset,TBH=min_grain_diff(grain_guesses,estimate_tb,thickness,temperature,density,freq)
#     print(grain_size)
#     print(final_offset)

    

# Changing Font Size
import matplotlib
font = {'size'   : 17}

matplotlib.rc('font', **font)

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


from datetime import datetime
import time

def year_fraction(date):
    start = datetime.date(datetime(date.year, 1, 1)).toordinal()
    year_length = datetime.date(datetime(date.year+1, 1, 1)).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

import scipy.io
mat = scipy.io.loadmat('cheat_sheet_CFM_Dates.mat')
CFM_years=np.squeeze(mat['dts_year'])
CFM_months = np.squeeze(mat['dts_month'])
CFM_days = np.squeeze(mat['dts_day'])
decyears = np.zeros(mat['dts_day'].shape)
decyears = np.squeeze(decyears)


for ind in np.arange(0,len(CFM_years)):
    year = CFM_years[ind]
    month = CFM_months[ind]
    day = CFM_days[ind]
    dt = datetime(year,month,day)
    decyears[ind] = year_fraction(dt)

if initial_loc>12:
    decyears = decyears[decyears>2009.]
    print(decyears)

first_year_cut = year_fraction(datetime(first_year_int,first_month,first_day))
last_year_cut = year_fraction(datetime(last_year_int,last_month,last_day))




# CFM_time = np.asarray(CFM_object['time']) # assign time data (format is Year.FractionalYear)


# Creating a 3D time mask for specificly selected years
mask = ((decyears>=first_year_cut) & (decyears<=last_year_cut))


time_true = np.squeeze(np.where(mask==True))


# Grabbing data only for years used
CFM_time_truncate = np.asarray(decyears[time_true])

CFM_density = np.asarray(CFM_object['density'])[:,time_true,:]
CFM_thickness = np.asarray(CFM_object['thickness'])[:,time_true,:]
CFM_temperature = np.asarray(CFM_object['temperature'])[:,time_true,:]
CFM_grain = np.asarray(CFM_object['grain_size'])[:,time_true,:]


# canitbemelt = np.zeros(len(days))


# Estimating Ccanitbemeltorr Length Boundary Conditions

predicted_CL = np.zeros(TB18H_s_new.shape)*np.nan


#-----


initial_bounds = [0.1e-3,0.5e-3]

try:
    day = np.where(((dry_snow_tb18h_filled>0)&(~np.isnan(dry_snow_tb18h_filled))))[0][0]
    good = ((CFM_thickness[loc,day,:]>0.) & (~np.isnan(CFM_thickness[loc,day,:])))
    if sum(good==True) < 1:
        print('Oh no! there is no good data for this day!')
    
    temp = CFM_temperature[loc,day,:][good]
    dens = CFM_density[loc,day,:][good]
    th = CFM_thickness[loc,day,:][good]
    tb = dry_snow_tb18h_filled[day]


    

    CLbounds,offset = calc_grain_size_estimate(initial_bounds,tb,np.asarray(th),temperature=temp,density=dens)

    secondary_bounds = [CLbounds-0.05e-3,CLbounds+0.05e-3]
except:
    secondary_bounds = initial_bounds
print('initial test day ',day)





# for day in np.arange(89,len(times4)-1,1):
for day in np.arange(0,len(days_new),20):
    
    
#     if day < 0365:
#         continue

    tb = dry_snow_tb18h_filled[day]

    if tb<0 or np.isnan(tb):
        
        continue
    

        
#     if melt_mask[day] == 0: 
#         print('Today is a melt day, but were continuing anyway...')
# #         continue    
    
    good = ((CFM_thickness[loc,day,:]>0.) & (~np.isnan(CFM_thickness[loc,day,:])))
    
    if sum(good==True) < 1:
        
        continue
        
    temp = CFM_temperature[loc,day,:][good]
    dens = CFM_density[loc,day,:][good]
    th = CFM_thickness[loc,day,:][good]
   
    
 
    
    CL,offset = calc_grain_size_estimate(secondary_bounds,tb,np.asarray(th),temperature=temp,density=dens)

    print('Secondary CL ',CL)
    print('Secondary Offset ',offset)

    
    predicted_CL[day] = CL



y_estimate=predicted_CL.copy()
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

nans, x= nan_helper(y_estimate)
y_estimate[nans]= np.interp(x(nans), x(~nans), y_estimate[~nans])

           
predicted_CL = np.zeros(TB18V_s_new.shape)*np.nan
offset_guesses = np.zeros(TB18V_s_new.shape)*np.nan
offsets = np.zeros(TB18V_s_new.shape)*np.nan

# for day in np.arange(89,len(times4)-1,1):
for day in np.arange(0,len(days_new),1):
    
#     if day < 365:
#         continue

    tb = dry_snow_tb18h_filled[day]

    if tb<0 or np.isnan(tb):
        continue
    

        
#     if melt_mask[day] == 0: 
#         print('Today is a melt day, but were continuing anyway...')
# #         continue    
    
    good = ((CFM_thickness[loc,day,:]>0.) & (~np.isnan(CFM_thickness[loc,day,:])))
    
    if sum(good==True) < 1:
        continue
        
    temp = CFM_temperature[loc,day,:][good]
    dens = CFM_density[loc,day,:][good]
    th = CFM_thickness[loc,day,:][good]
    
    
#     init_bounds = [150e-6,300e-6]
    
    CL,offset,CL_orig,offset_orig = calc_grain_size(tb,np.asarray(th),temperature=temp,density=dens,y_estimate=y_estimate[day])

    print('Third CL ',CL_orig)
    print('Third Offset: ',offset_orig)
    print('Final CL ', CL)
    print('Final Offset: ',offset)
    
    predicted_CL[day] = CL
    offsets[day] = offset

#     offset_guesses[day]=offset_guess
    







def window_stdev(X, window_size):
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)

y=predicted_CL.copy()
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

nans, x= nan_helper(y)
y[nans]= np.interp(x(nans), x(~nans), y[~nans])
# FILLING IN THE GAPS OF CORRELATION LENGTH (Y)


y_filt = ndimage.filters.median_filter(y,5)

y_std = 1*np.std(y)

y_down = y_filt-y_std
y_up = y_filt+y_std

running_std=window_stdev(y, 31.)
multiplier = 4
median_run_std = np.median(running_std[~np.isnan(predicted_CL)])
y_up_redone = y+multiplier*median_run_std
y_up_redone_filt = y_filt+multiplier*median_run_std
y_down_redone = y-multiplier*median_run_std
y_down_redone_filt = y_filt-multiplier*median_run_std


def window_stdev(X, window_size):
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)




TB = np.zeros(len(days_new))*np.nan
TB_up = np.zeros(len(days_new))*np.nan
TB_down = np.zeros(len(days_new))*np.nan

TB_filt = np.zeros(len(days_new))*np.nan
TB_up_filt = np.zeros(len(days_new))*np.nan
TB_down_filt = np.zeros(len(days_new))*np.nan


difference_up = np.zeros(len(days_new))*np.nan
difference_down = np.zeros(len(days_new))*np.nan
difference = np.zeros(len(days_new))*np.nan

difference_up_filt = np.zeros(len(days_new))*np.nan
difference_down_filt = np.zeros(len(days_new))*np.nan
difference_filt = np.zeros(len(days_new))*np.nan

for day in np.arange(0,len(days_new),1):

 

    good = ((CFM_thickness[loc,day,:]>0.) & (~np.isnan(CFM_thickness[loc,day,:])))
    
    if sum(good==True) < 1:
        print('boo')
        continue
        
    temp = CFM_temperature[loc,day,:][good]
    dens = CFM_density[loc,day,:][good]
    th = CFM_thickness[loc,day,:][good]
    tb = TB18H_s_new[day]
    

    
    freq = '18'
    
    TB[day],difference[day]=forward_calc(y[day],tb,th,temp,dens,0,freq)
    # TB_up[day],difference_up[day]=forward_calc(y_up_redone[day],tb,th,temp,dens,0,freq)
    TB_down[day],difference_down[day]=forward_calc(y_down_redone[day],tb,th,temp,dens,0,freq)
    TB_filt[day],difference_filt[day]=forward_calc(y_filt[day],tb,th,temp,dens,0,freq)
    # TB_up_filt[day],difference_up_filt[day]=forward_calc(y_up_redone_filt[day],tb,th,temp,dens,0,freq)
    TB_down_filt[day],difference_down_filt[day]=forward_calc(y_down_redone_filt[day],tb,th,temp,dens,0,freq)

    print(TB[day])

    
    
y_filt = ndimage.filters.median_filter(y,21)



def window_stdev(X, window_size):
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)


y_stdev = 3.*window_stdev(y,31)
y_stdev2 = 1*np.std(y)

y_stdev[np.isnan(predicted_CL)] = np.mean(y_stdev[~np.isnan(predicted_CL)])

from matplotlib.dates import MonthLocator, YearLocator

fig,(ax1,ax3,ax2) = plt.subplots(3,1,figsize=(16,12))
# times2=[]
# for day in days:

#     times2.append(datetime(int(day), 1, 1) + timedelta(days = (day % 1) * 365))


# ax1.set_xlim([datetime(2014,11,25),datetime(2018,12,31)])
ax1.plot(days_new[TB18H_s_new>0],TB18H_s_new[TB18H_s_new>0],color='k')
ax1.plot(days_new,melt_thresholdZF_snapped,'c--',linewidth=2,label='Zwally and Fiegles, 1994')
ax1.plot(days_new,melt_thresholdT_snapped,'b--',linewidth=2,label='Torinesi et al., 2003')
ax1.plot(days_new,melt_thresholdP_snapped,'--',linewidth=2,color='r',label='Picard et al., 2022')
ax1.set_ylabel('$T_B$ (18H) [K]')
ax1.minorticks_on()
ax1.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
ax1.grid(which='major', linestyle=':', linewidth='0.8', color='black')
ax1.set_xlim([datetime(first_year_int,first_month,first_day),datetime(last_year_int,last_month,last_day)])

ax1.legend(ncol=3,loc='upper left')

yloc = YearLocator()
mloc = MonthLocator()
ax1.xaxis.set_major_locator(yloc)
ax1.xaxis.set_minor_locator(mloc)
ax1.grid(True)
ax1.margins(y=0.2)
# ax1.set_ylim([140,300])


ax3.fill_between(days_new, y_down_redone*10**3,y_up_redone*10**3,color='silver',label='+/- 4$\sigma$ correl. length')
ax3.plot(days_new,y*10**3,'g-',label='SMRT Correlation Length')
# ax3.set_xlim([datetime(2014,11,25),datetime(2018,12,31)])
ax3.set_ylabel('Correl. Length [mm]')
ax3.minorticks_on() 
ax3.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
ax3.grid(which='major', linestyle=':', linewidth='0.8', color='black')
yloc = YearLocator()
mloc = MonthLocator()
ax3.xaxis.set_major_locator(yloc)
ax3.xaxis.set_minor_locator(mloc)
ax3.grid(True)
ax3.legend(ncol=2,loc='upper left')
# ax3.set_ylim([140,275])


ax3.plot(days_new,y_filt*10**3,'k-',label='Median Filter of Correl. Length')
ax3.set_xlim([datetime(first_year_int,first_month,first_day),datetime(last_year_int,last_month,last_day)])

melt_days=TB18H_s_new.copy()*np.nan
melt_daysflat=TB18H_s_new.copy()*np.nan

melt_days[TB18H_s_new>TB_down] = TB18H_s_new[TB18H_s_new>TB_down]
melt_daysflat[TB18H_s_new>TB_down] = 0.


ax2.fill_between(days_new, TB,TB_down,color='silver',label='- 4$\sigma$ correl. length')

ax2.plot(days_new[TB18H_s_new>0],TB18H_s_new[TB18H_s_new>0],color='k')
ax2.plot(days_new,TB,color='purple',label='SMRT Simulated Dry Snow')
# ax2.plot(times3[1:-4],melt_threshold_snapped[1:-4]-10.,'-',color='m',label='Hybrid method')
ax2.set_xlim([datetime(first_year_int,first_month,first_day),datetime(last_year_int,last_month,last_day)])
ax2.set_ylabel('$T_B$ (18H) [K]')
ax2.minorticks_on()
ax2.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
ax2.grid(which='major', linestyle=':', linewidth='0.8', color='black')

yloc = YearLocator()
mloc = MonthLocator()
ax2.xaxis.set_major_locator(yloc)
ax2.xaxis.set_minor_locator(mloc)
ax2.grid(True)
ax2.plot(days_new,melt_days,'.',color='steelblue',label='Melt days',markersize=10)
ax2.legend(ncol=3,loc='upper left')
# ax2.set_ylim([140,300])
ax2.margins(y=0.2)
# ax2.autoscale()
# ax2.set_ylim(top=1, bottom=-1)



# ax2.set_axisbelow(True)
# ax2.yaxis.grid(color='black', linestyle='dashed')
# ax2.xaxis.grid(color='black', linestyle='dashed')


# cat=pickle.load(open('aws18melt_data.p','rb'))

# ax4.bar(cat['dates_AWS'],cat['melt_sums'],color='navy',label='AWS Daily Melt')
# ax4.plot(days_new,melt_daysflat+3,'.',color='steelblue',label='SMRT Melt Days',markersize=10)
# ax4.set_xlim([datetime(2014,11,25),datetime(2018,12,31)])
# ax4.minorticks_on() 
# ax4.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
# ax4.grid(which='major', linestyle=':', linewidth='0.8', color='black')
# yloc = YearLocator()
# mloc = MonthLocator()
# ax4.set_ylim([0.,20.])
# ax4.xaxis.set_major_locator(yloc)
# ax4.xaxis.set_minor_locator(mloc)
# ax4.grid(True)
# ax4.legend()
# ax4.set_ylabel('Melt [mm w.e.]')
# # ax4.xaxis_date()

# ax3.set_ylim([140,275])
now = datetime.now() # current date and time

date_time = now.strftime("%m%d%Y_%H%M%S")


fig.savefig('OutputHybrid/hybridmethod18H'+str(initial_loc)+'_'+date_time+'.png',dpi=240)

##### SECOND FIGURE, WITH FILTERING

fig,(ax1,ax3,ax2) = plt.subplots(3,1,figsize=(16,12))
# times2=[]
# for day in days:

#     times2.append(datetime(int(day), 1, 1) + timedelta(days = (day % 1) * 365))


# ax1.set_xlim([datetime(2014,11,25),datetime(2018,12,31)])
ax1.plot(days_new[TB18H_s_new>0],TB18H_s_new[TB18H_s_new>0],color='k')
ax1.plot(days_new,melt_thresholdZF_snapped,'c--',linewidth=2,label='Zwally and Fiegles, 1994')
ax1.plot(days_new,melt_thresholdT_snapped,'b--',linewidth=2,label='Torinesi et al., 2003')
ax1.plot(days_new,melt_thresholdP_snapped,'--',linewidth=2,color='r',label='Picard et al., 2022')
ax1.set_ylabel('$T_B$ (18H) [K]')
ax1.minorticks_on()
ax1.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
ax1.grid(which='major', linestyle=':', linewidth='0.8', color='black')
ax1.set_xlim([datetime(first_year_int,first_month,first_day),datetime(last_year_int,last_month,last_day)])

ax1.legend(ncol=3,loc='upper left')

yloc = YearLocator()
mloc = MonthLocator()
ax1.xaxis.set_major_locator(yloc)
ax1.xaxis.set_minor_locator(mloc)
ax1.grid(True)
ax1.margins(y=0.2)
# ax1.set_ylim([140,300])


ax3.fill_between(days_new, y_down_redone_filt*10**3,y_up_redone_filt*10**3,color='silver',label='+/- 4$\sigma$ correl. length')
ax3.plot(days_new,y*10**3,'g-',label='SMRT Correlation Length')
# ax3.set_xlim([datetime(2014,11,25),datetime(2018,12,31)])
ax3.set_ylabel('Correl. Length [mm]')
ax3.minorticks_on() 
ax3.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
ax3.grid(which='major', linestyle=':', linewidth='0.8', color='black')
yloc = YearLocator()
mloc = MonthLocator()
ax3.xaxis.set_major_locator(yloc)
ax3.xaxis.set_minor_locator(mloc)
ax3.grid(True)
ax3.legend(ncol=2,loc='upper left')
# ax3.set_ylim([140,275])


ax3.plot(days_new,y_filt*10**3,'k-',label='Median Filter of Correl. Length')
ax3.set_xlim([datetime(first_year_int,first_month,first_day),datetime(last_year_int,last_month,last_day)])



melt_days_filt=TB18H_s_new.copy()*np.nan
melt_daysflat_filt=TB18H_s_new.copy()*np.nan

melt_days_filt[TB18H_s_new>TB_down_filt] = TB18H_s_new[TB18H_s_new>TB_down_filt]
melt_daysflat_filt[TB18H_s_new>TB_down_filt] = 0.


ax2.fill_between(days_new, TB_filt, TB_down_filt,color='silver',label='- 4$\sigma$ correl. length')

ax2.plot(days_new[TB18H_s_new>0],TB18H_s_new[TB18H_s_new>0],color='k')
ax2.plot(days_new,TB_filt,color='purple',label='SMRT Simulated Dry Snow')
# ax2.plot(times3[1:-4],melt_threshold_snapped[1:-4]-10.,'-',color='m',label='Hybrid method')
ax2.set_xlim([datetime(first_year_int,first_month,first_day),datetime(last_year_int,last_month,last_day)])
ax2.set_ylabel('$T_B$ (18H) [K]')
ax2.minorticks_on()
ax2.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
ax2.grid(which='major', linestyle=':', linewidth='0.8', color='black')

yloc = YearLocator()
mloc = MonthLocator()
ax2.xaxis.set_major_locator(yloc)
ax2.xaxis.set_minor_locator(mloc)
ax2.grid(True)
ax2.plot(days_new,melt_days_filt,'.',color='steelblue',label='Melt days',markersize=10)
ax2.legend(ncol=3,loc='upper left')
# ax2.set_ylim([140,300])
ax2.margins(y=0.2)
# ax2.autoscale()
# ax2.set_ylim(top=1, bottom=-1)



# ax2.set_axisbelow(True)
# ax2.yaxis.grid(color='black', linestyle='dashed')
# ax2.xaxis.grid(color='black', linestyle='dashed')


# cat=pickle.load(open('aws18melt_data.p','rb'))

# ax4.bar(cat['dates_AWS'],cat['melt_sums'],color='navy',label='AWS Daily Melt')
# ax4.plot(days_new,melt_daysflat+3,'.',color='steelblue',label='SMRT Melt Days',markersize=10)
# ax4.set_xlim([datetime(2014,11,25),datetime(2018,12,31)])
# ax4.minorticks_on() 
# ax4.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
# ax4.grid(which='major', linestyle=':', linewidth='0.8', color='black')
# yloc = YearLocator()
# mloc = MonthLocator()
# ax4.set_ylim([0.,20.])
# ax4.xaxis.set_major_locator(yloc)
# ax4.xaxis.set_minor_locator(mloc)
# ax4.grid(True)
# ax4.legend()
# ax4.set_ylabel('Melt [mm w.e.]')
# # ax4.xaxis_date()

# ax3.set_ylim([140,275])



fig.savefig('OutputHybrid/filterhybridmethod18H'+str(initial_loc)+'_'+date_time+'.png',dpi=240)


melt_thresholdP_snapped = melt_thresholdP[((days>=datetime(first_year_int,first_month,first_day))&(days<=datetime(last_year_int,last_month,last_day)))]
melt_thresholdZF_snapped = melt_thresholdZF[((days>=datetime(first_year_int,first_month,first_day))&(days<=datetime(last_year_int,last_month,last_day)))]
melt_thresholdT_snapped = melt_thresholdT[((days>=datetime(first_year_int,first_month,first_day))&(days<=datetime(last_year_int,last_month,last_day)))]



import pickle

pickle.dump({'offsets':offsets,'thresholdP':melt_thresholdP_snapped,'thresholdZF':melt_thresholdZF_snapped,'thresholdT':melt_thresholdT_snapped,'days':days_new,'TB18H':TB18H_s_new,'TB18V':TB18V_s_new,'y':y,'y_filt':y_filt,'y_up_redone':y_up_redone,'y_down_redone':y_down_redone,'y_stdev2':y_stdev2,'TB':TB,'TB_filt':TB_filt,'TB_up_filt':TB_up_filt,'TB_down_filt':TB_down_filt, 'predicted_CL':predicted_CL,'TB_up':TB_up,'TB_down':TB_down,'melt_days':melt_days,'melt_days_filt':melt_days_filt,'melt_daysflat':melt_daysflat,'melt_daysflat_filt':melt_daysflat_filt,'difference':difference,'difference_filt':difference_filt},open('OutputHybrid/Site'+str(initial_loc)+'_'+date_time+'corr_length_file18H.p','wb'))


print("--- %s seconds ---" % (time.time() - start_time))


