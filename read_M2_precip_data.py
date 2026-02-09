#!/usr/bin/env python
# coding: utf-8

# In[14]:


from scipy import stats
import numpy as np

r = len(xvalpig['temperature'])
r_values91pig = []
for depth in np.arange(r):
    temps = xvalpig['temperature'][depth,2::12]
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.squeeze(tbv_amsr91pig),temps)
    r_values91pig.append(r_value)

r = len(xvalpig['temperature'])
r_values37pig = []
for depth in np.arange(r):
    temps = xvalpig['temperature'][depth,2::12]
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.squeeze(tbv_amsr37pig),temps)
    r_values37pig.append(r_value)

r = len(xvalpig['temperature'])
r_values19pig = []
for depth in np.arange(r):
    temps = xvalpig['temperature'][depth,2::12]
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.squeeze(tbv_amsr19pig),temps)
    r_values19pig.append(r_value)


# In[15]:


from scipy import stats
import numpy as np

r = len(xvalkoh['temperature'])
r_values91koh = []
for depth in np.arange(r):
    temps = xvalkoh['temperature'][depth,2::4]
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.squeeze(tbv_amsr91koh),temps)
    r_values91koh.append(r_value)

r = len(xvalkoh['temperature'])
r_values37koh = []
for depth in np.arange(r):
    temps = xvalkoh['temperature'][depth,2::4]
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.squeeze(tbv_amsr37koh),temps)
    r_values37koh.append(r_value)

r = len(xvalkoh['temperature'])
r_values19koh = []
for depth in np.arange(r):
    temps = xvalkoh['temperature'][depth,2::4]
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.squeeze(tbv_amsr19koh),temps)
    r_values19koh.append(r_value)


# In[96]:


tbv_amsr91.shape
xval['temperature'][0,2::4].shape


# In[92]:


plt.plot(xval['temperature'][0,2::12])
plt.plot(np.squeeze(tbv_amsr91))


# In[60]:


plt.figure(figsize=(5,5))
x = np.zeros(10)
y = x + np.arange(10)
plt.plot(x,y,c=[0.3,0.3,0.3],lw=1)

plt.plot(r_values91pig,xvalpig['depth'][:,0],color='red',label='PIG 91 V')
plt.plot(np.max(r_values91pig[:200]),xvalpig['depth'][:,0][np.argmax(r_values91pig[:200])],'.',markersize=12,color='red')

plt.gca().invert_yaxis()

plt.plot(r_values37pig,xvalpig['depth'][:,0],color='blue',label='PIG 37 V')
plt.plot(np.max(r_values37pig[:200]),xvalpig['depth'][:,0][np.argmax(r_values37pig[:200])],'.',markersize=12,color='blue')


plt.plot(r_values19pig,xvalpig['depth'][:,0],color='green',label='PIG 19 V')

plt.plot(np.max(r_values19pig[:200]),xvalpig['depth'][:,0][np.argmax(r_values19pig[:200])],'.',markersize=12,color='green')


plt.plot(r_values91koh,xvalkoh['depth'][:,0],'r--',label='Kohnen 91 V')
plt.plot(np.max(r_values91koh[:200]),xvalkoh['depth'][:,0][np.argmax(r_values91koh[:200])],'v',markersize=10,color='r')

plt.gca().invert_yaxis()

plt.plot(r_values37koh,xvalkoh['depth'][:,0],'b--',label='Kohnen 37 V')
plt.plot(np.max(r_values37koh[:200]),xvalkoh['depth'][:,0][np.argmax(r_values37koh[:200])],'v',markersize=10,color='b')


plt.plot(r_values19koh,xvalkoh['depth'][:,0],'g--',label='Kohnen 19 V')

plt.plot(np.max(r_values19koh[:200]),xvalkoh['depth'][:,0][np.argmax(r_values19koh[:200])],'v',markersize=10,color='g')


plt.grid()
plt.legend(loc='lower left')
plt.ylim([4,0])
plt.xlim([0.,1.])


plt.xlabel('Correlation coefficient ($r$)',fontsize=13)
plt.ylabel('Depth [m]',fontsize=13)
plt.savefig('Correlation_with_depth.png',dpi=300)


# In[103]:


xval = loadmat('Testing Datasets/PIG_CFM_2017.mat')


# In[10]:


precip_set['lon']


# In[6]:


latm2


# In[ ]:


import xarray
import glob
files=glob.glob('Data2017MERRA2/MERRA2_400.statD*nc4')

temp_set=xarray.open_mfdataset(files[:])
latm2 = temp_set['lat']
lonm2 = temp_set['lon']
temp = temp_set['T2MMAX']
time = temp_set['time']


# In[12]:


# SETTINGS 

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import sys

freq =  '19'
pol = 'both'
loc_name = 'kohnen_fdm'
sampling_period = 200.  # how many days between samples
# LOADING 

if loc_name == 'pine_island_fdm':
    lat = -75.5414
    lon = -96.0856
elif loc_name == 'kohnen_fdm':
    print('this_Is_Kohnen_FDM')
    lat = -75.01
    lon = 0
elif loc_name == 'dome_c_fdm':
    lat = -75.1152
    lon = 123.0792

if freq == '91V':
    freqm = '89V'
    freql = '91v'  
elif freq == '37V':
    freqm = '37V'
    freql = '37v'
elif freq == '19V':
    freqm = '19V'
    freql = '19v'
elif freq == '37H':
    freqm = '37H'
    freql = '37h'
else:
    print('Sorry, not a recognized frequency/polarization')


import numpy as np
with open("/Users/mdattler/Desktop/Research/Microwave/FirnData/tbcube/tbcube_f17_2017_v5_s"+freq+"h.bin", "rb") as binary_file:
    outfile=np.fromfile(binary_file, dtype=np.uint16)
tbh_amsr = outfile/10.

with open("/Users/mdattler/Desktop/Research/Microwave/FirnData/tbcube/tbcube_f17_2017_v5_s"+freq+"v.bin", "rb") as binary_file:
    outfile=np.fromfile(binary_file, dtype=np.uint16)
tbv_amsr = outfile/10.


if freq == '91':
    tbh_amsr = tbh_amsr.reshape(365,332*2,316*2)
    tbv_amsr = tbv_amsr.reshape(365,332*2,316*2)
    skip = 2
else: 
    tbh_amsr = tbh_amsr.reshape(365,332,316)
    tbv_amsr = tbv_amsr.reshape(365,332,316)
    skip = 4

from pyproj import Proj, transform

inProj = Proj(init='epsg:3412')
outProj = Proj(init='epsg:4326')



PIGx,PIGy = transform(outProj,inProj,lon,lat)

# GRABBING GEOLOCATION DATA

if (freq == '19' or freq == '37'): # 25 km resolution
    with open("Geolocation Files/pss25lons_v3.dat", "rb") as binary_file:
        outfile=np.fromfile(binary_file, dtype=np.int32)
    lons = outfile/100000.
    lons = lons.reshape(332,316)

    with open("Geolocation Files/pss25lats_v3.dat", "rb") as binary_file:
        outfile=np.fromfile(binary_file, dtype=np.int32)
    lats = outfile/100000.
    lats = lats.reshape(332,316)

    X,Y = transform(outProj,inProj,lons,lats)
    SITEx,SITEy = transform(outProj,inProj,lon,lat)
    pointsx = ((X[0,:]<SITEx+12.5003*1000.) & (X[0,:]>SITEx-12.5003*1000.))
    pointsy = ((Y[:,0]<SITEy+12.5003*1000.) & (Y[:,0s]>SITEy-12.5003*1000.))


elif (freq == '91'): # 12 km resolution
    with open("Geolocation Files/pss12lons_v3.dat", "rb") as binary_file:
        outfile=np.fromfile(binary_file, dtype=np.int32)
    lons = outfile/100000.
    lons = lons.reshape(332*2,316*2)

    with open("Geolocation Files/pss12lats_v3.dat", "rb") as binary_file:
        outfile=np.fromfile(binary_file, dtype=np.int32)
    lats = outfile/100000.
    lats = lats.reshape(332*2,316*2)

    X,Y = transform(outProj,inProj,lons,lats)
    SITEx,SITEy = transform(outProj,inProj,lon,lat)

    pointsx = ((X[0,:]<SITEx+6.251*1000.) & (X[0,:]>SITEx-6.251*1000.))
    pointsy = ((Y[:,0]<SITEy+6.251*1000.) & (Y[:,0]>SITEy-6.251*1000.))

pointsx = np.where(pointsx==True)[0]
pointsy = np.where(pointsy==True)[0]

if len(pointsx) > 1:
    pointsx = pointsx[0]
    print('Note, your data is exactly in the center of two points')
if len(pointsy) > 1:
    pointsy = pointsy[0]
    print('Note, your data is exactly in the center of two points')

if freq == '19':
    tbv_amsr19koh = tbv_amsr[:,pointsy,pointsx]
    tbh_amsr19koh = tbv_amsr[:,pointsy,pointsx]
elif freq == '37':
    tbv_amsr37koh = tbv_amsr[:,pointsy,pointsx]
    tbh_amsr37koh = tbv_amsr[:,pointsy,pointsx]
elif freq == '91':
    tbv_amsr91koh = tbv_amsr[:,pointsy,pointsx]
    tbh_amsr91koh = tbh_amsr[:,pointsy,pointsx]

# tbv_amsr[:,pointsy,pointsx]


# In[31]:


plt.plot(tbv_amsr91[:,pointsy,pointsx])


# In[625]:


plt.imshow(XPRGR[0,:,:])


# In[77]:


from matplotlib import pyplot as plt
from mpl_toolkits import basemap as bm
from matplotlib import colors
import numpy as np
import numpy.ma as ma
from matplotlib.patches import Path, PathPatch
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt
import matplotlib.cm as cm 


fig,ax = plt.subplots(1,1,figsize=(12,12))

w = 5700000
h = 4700000
m = Basemap(width=w,height=h,
            resolution='l',projection='laea',\
            lat_ts=-71,lat_0=-90,lon_0=0.)

lons,lats=m(X+w/2,Y+h/2,inverse=True)
lons[np.where(lons>179.5)] = 179.8
lons[np.where(lons<-179.5)] = -179.8
plt.rc('font', family='serif')


day = 150


from mpl_toolkits.basemap import maskoceans
nc_new = maskoceans(lons,lats,XPRGR[day,:,:])
# m = Basemap(projection='stere',urcrnrlon=180.,urcrnrlat=-60.,llcrnrlon=-180,llcrnrlat=-90,resolution='l',lat_ts=-71.,lat_0=-90,lon_0=0,epsg=3031)

ax.scatter(w/2,h/2,c='grey',s=200,zorder=0)
bob = ax.pcolor(X[::,::]+w/2,Y[::,::]+h/2,nc_new,cmap=cm.get_cmap('Reds_r',2),vmax = 0., vmin=-0.1)
# ax.plot(x,y,'*',color='yellow',markersize=22,markeredgewidth=1.5,markeredgecolor='k')

cb=plt.colorbar(bob,ax=ax,shrink=0.68,orientation='vertical',extend='both')

cb.set_label('Melt',fontsize=25)
font_size = 16 # Adjust as appropriate.
cb.ax.tick_params(labelsize=19) 
# m.drawcoastlines()

m.drawmapboundary(fill_color='xkcd:lightblue')
# parallels = np.arange(-80,0.,40.)
# m.drawparallels(parallels,labels=[False,False,False,False])
# meridians = np.arange(0.,360.,30.)
# m.drawmeridians(meridians,labels=[False,False,False,False])


ax.set_xlim([0,6000000])
ax.set_ylim([0,5000000])



w = 5700000
h = 4700000
fig,ax2 = plt.subplots(1,1,figsize=(12,12))

m = Basemap(width=w,height=h,
            resolution='l',projection='laea',\
            lat_ts=-71,lat_0=-90,lon_0=0.)

lons,lats=m(X+w/2,Y+h/2,inverse=True)
lons[np.where(lons>179.5)] = 179.8
lons[np.where(lons<-179.5)] = -179.8
plt.rc('font', family='serif')


latm2 = temp_set['lat']
lonm2 = temp_set['lon']
temp = temp_set['T2MMAX']
time = temp_set['time']

x,y=np.meshgrid(np.asarray(lonm2),np.asarray(latm2))

(xm2,ym2)=m(x,y)

# from mpl_toolkits.basemap import maskoceans
# nc_new = maskoceans(lons,lats,XPRGR[0,:,:])
# # m = Basemap(projection='stere',urcrnrlon=180.,urcrnrlat=-60.,llcrnrlon=-180,llcrnrlat=-90,resolution='l',lat_ts=-71.,lat_0=-90,lon_0=0,epsg=3031)

# ax2.scatter(w/2,h/2,c='grey',s=200,zorder=0)
x[np.where(x>178.5)] = 179.3
x[np.where(x<-178.5)] = -179.3
nc_new = maskoceans(x,y,temp[day,:,:])
bob2 = ax2.pcolor(xm2,ym2,nc_new,cmap=cm.get_cmap('bwr',16),vmax = 260, vmin=284)
# ax.plot(x,y,'*',color='yellow',markersize=22,markeredgewidth=1.5,markeredgecolor='k')

cb=plt.colorbar(bob2,ax=ax2,shrink=0.68,orientation='vertical',extend='both')

cb.set_label('MERRA-2 Max 24 hr Temp. [K]',fontsize=25)
font_size = 16 # Adjust as appropriate.
cb.ax.tick_params(labelsize=19) 
# m.drawcoastlines()

m.drawmapboundary(fill_color='xkcd:lightblue')
# parallels = np.arange(-80,0.,40.)
# m.drawparallels(parallels,labels=[False,False,False,False])
# meridians = np.arange(0.,360.,30.)
# m.drawmeridians(meridians,labels=[False,False,False,False])

m.drawcoastlines()
ax2.set_xlim([0,6000000])
ax2.set_ylim([0,5000000])




# plt.plot(PIGx+w/2,PIGy+h/2,'*',c='k',markersize=20)
# plt.plot(PIGx+w/2,PIGy+h/2,'*',c='yellow',markersize=14)


# m.drawmapboundary(fill_color=[0.94,0.94,0.94])
# parallels = np.arange(-80,0.,40.)
# m.drawparallels(parallels,labels=[False,False,False,False])
# meridians = np.arange(0.,360.,30.)
# m.drawmeridians(meridians,labels=[False,False,False,False])






# plt.savefig('2017MeanBrightnessTemp'+freq+'.png',dpi=300)
# plt.savefig('forcolorbar'+freq+'.png',dpi=300)







# In[6]:


# TESTING OUT MELT DETECTION

XPRGR = (tbh_amsr19-tbv_amsr37)/(tbh_amsr19+tbv_amsr37)


# In[ ]:


np.random.seed(1234)

time_step = 0.02
period = 5.

time_vec = np.arange(0, 20, time_step)
sig = (np.sin(2 * np.pi / period * time_vec)
       + 0.5 * np.random.randn(time_vec.size))

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')


# In[17]:


from pyhht.visualization import plot_imfs
import numpy as np
from pyhht import EMD
t = np.linspace(0, 365, len(tbv_amsr[:,pointsy,pointsx]))
x = np.squeeze(tbv_amsr[:,pointsy,pointsx])-np.nanmean(xval['temperature'][0:1,2::12],axis=0)
# modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
decomposer = EMD(x)
imfs = decomposer.decompose()
f1=plt.figure(figsize=(7,7))
plot_imfs(x, imfs, t) 


from pyhht.visualization import plot_imfs
import numpy as np
from pyhht import EMD
t2 = np.linspace(0, 365, len(tbv_amsr[:,pointsy,pointsx]))
x2 = np.squeeze(tbh_amsr[:,pointsy,pointsx])
# modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
decomposer2 = EMD(x2)
imfs2 = decomposer2.decompose()
f1=plt.figure(figsize=(7,7))
plot_imfs(x2, imfs2, t2) 

from pyhht.visualization import plot_imfs
import numpy as np
from pyhht import EMD
t3 = np.linspace(0, 365, len(tbv_amsr[:,pointsy,pointsx]))
x3 = np.nanmean(xval['temperature'][0:1,2::12],axis=0)
# modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
decomposer3 = EMD(x3)
imfs3 = decomposer3.decompose()
f1=plt.figure(figsize=(7,7))
plot_imfs(x3, imfs3, t3) 


# In[587]:


mylist = list(np.mean(xval['temperature'][0:5,0::4],axis=0)-np.mean(xval['temperature'][20:23,0::4],axis=0))
N = 
cumsum, moving_aves = [0], []

for i, x in enumerate(mylist, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)


# In[523]:


plt.figure(figsize=(10,4))
plt.plot(np.linspace(0,365,len(xval['temperature'][0,:])),np.mean(xval['temperature'][0:1,:],axis=0))
plt.plot(x2*1.2+30)
# plt.xlim([100,225])
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8))
ax1.plot(np.linspace(0,365,len(precip)),precip,'-',color='xkcd:lightblue')
# plt.figure(figsize=(10,4))
ax2.plot(imfs2[0]+imfs2[1]-imfs3[0]-imfs3[1])


# In[490]:


residuals = x2*1.2+30.-np.linspace(0,365,len(xval['temperature'][0,2::4])),np.mean(xval['temperature'][0:1,::4],axis=0)
plt.plot(residuals)


# In[87]:


import numpy as np 
fig, (ax1,ax1a) = plt.subplots(2,1,figsize=(12,6))

color = 'tab:blue'
# ax2 = ax1.twinx()
precip = precip_set['PRECSNO'][:,1,0]
precip2 = precip_set['PRECSNO'][:,0,1]
# new_precip = np.asarray(precip_set['PRECSNO'][:,1,0]).copy()
# new_precip2 = np.asarray(precip_set['PRECSNO'][:,0,1]).copy()

# new_precip[precip>0.00003] = 1.
# new_precip[precip<=0.00003] = -1.
ax2_PLUS = ax1.twinx()
ax3_PLUS = ax1a.twinx()
# ax2_PLUS.plot(precip_set['time'][:],precip,'-',color='xkcd:lightblue')
# ax2_PLUS.plot(precip_set['time'][:],precip2,'-',color='xkcd:lightblue')
ax2_PLUS.set_ylabel('Precipitation [$kg m^{-2} s^{-1}$]',fontsize=14,color='tab:blue')
# ax3_PLUS.plot(precip_set['time'][:],precip,'-',color='xkcd:lightblue',alpha=0.7)
# ax3_PLUS.plot(precip_set['time'][:],precip2,'-',color='xkcd:lightblue',alpha=0.7)
days = []
for date in date_list:
    day=date+datetime.timedelta(days=1)
    days.append(day)

ax3_PLUS.plot(days,daily_precip,'-',color='xkcd:lightblue')
ax2_PLUS.plot(days,daily_precip,'-',color='xkcd:lightblue')

color = 'tab:red'
import datetime
base = datetime.datetime(2017,1,1)
date_list = [base + datetime.timedelta(days=x) for x in range(365)]
# ax2.plot(date_list,imfs[0]+imfs[1]+imfs[2]+imfs[3]+imfs[4]+imfs[5],color=color)

ax1.plot(date_list,imfs2[0]+imfs2[1]+imfs2[2]+imfs2[3],'-',color=[1.,0.6,0.6],linewidth=2,label='AMSR-2 ($T_B$)')
ax1.plot(date_list,imfs3[0]+imfs3[1]+imfs3[2]+imfs3[3],'-',color='green',linewidth=2,label='CFM ($T_P$)')

ax1.legend()

# ax1a.plot(date_list,imfs[0]+imfs[1]+imfs[2]+imfs[3],color='k',linewidth=2,label='Residuals ($T_B$ - $T_P$)')
ax1a.plot(date_list,imfs2[0]+imfs2[1]+imfs2[2]+imfs2[3]-(imfs3[0]+imfs3[1]+imfs3[2]+imfs3[3]),color='k',linewidth=2,label='Residuals ($T_B$ - $T_P$)')

# ax1.set_ylim([0.,3.])
ax1.set_ylim([-25,25.])
# ax1a.set_ylim([-45,-30])
# ax2.set_ylim([0,0.0007])
ax1.set_ylabel('$T_B$ and $T_P$ [K]',fontsize=14)
ax1a.set_ylabel('$T_B$ - $T_P$',fontsize=14)
ax1.set_xlim([datetime.datetime(2017,5,1),datetime.datetime(2017,11,1)])

ax1a.set_xlim([datetime.datetime(2017,5,1),datetime.datetime(2017,11,1)])

# ax3.plot(date_list,imfs[0]+imfs[1]+imfs[2]+imfs[3]-imfs2[0]-imfs2[1]-imfs2[2]-imfs2[3],color='k')
# ax3.set_ylabel('Residuals (K)')
# ax3.set_xlim([datetime.datetime(2017,10,1),datetime.datetime(2017,12,30)])
# ax3.set_ylim([-7,7])

ax1a.set_ylim([-25.,20])
plt.savefig('Recreating_Bindschadler.png')
plt.tight_layout()

plt.figure()
TB = imfs2[0]+imfs2[1]+imfs2[2]+imfs2[3];
TP = imfs3[0]+imfs3[1]+imfs3[2]+imfs3[3]
# plt.scatter(TB,TP)

# from scipy import stats
# slope, intercept, r_value, p_value, std_err = stats.linregress(TB,TP)


# plt.xlabel('$T_B$ [K]',fontsize=15)
# plt.ylabel('$T_P$ [K]',fontsize=15)

# print(r_value)
# print(p_value)


# In[117]:


import numpy as np 
fig, (ax0,ax1,ax1a) = plt.subplots(3,1,figsize=(12,9))

ax1.set_title('Pine Island Glacier')

color = 'tab:blue'
# ax2 = ax1.twinx()

# ax1.plot(date_list,x2,'-',color=[1.,0.6,0.6],linewidth=2,label='AMSR-2 ($T_B$)')
ax0.plot(date_list,x2,'-',color='orange',linewidth=2,label='AMSR-2 ($T_B$)')
ax0.set_ylabel('$T_B$ [K]',fontsize=14)
# ax1.set_ylim([200,265])
ax0.legend(loc='upper center')
precip = precip_set['PRECSNO'][:,1,0]
precip2 = precip_set['PRECSNO'][:,0,1]
# new_precip = np.asarray(precip_set['PRECSNO'][:,1,0]).copy()
# new_precip2 = np.asarray(precip_set['PRECSNO'][:,0,1]).copy()
ax0.set_ylim([200,225])
ax_PLUS2 = ax0.twinx()
ax_PLUS2.plot(precip_set['time'][:],precip,'-',color='xkcd:lightblue')
ax_PLUS2.plot(precip_set['time'][:],precip2,'-',color='xkcd:lightblue')
ax_PLUS2.set_ylabel('Precipitation [$kg m^{-2} s^{-1}$]',fontsize=14,color='tab:blue')
ax_PLUS2.plot(precip_set['time'][:],precip,'-',color='xkcd:lightblue',alpha=0.9)
ax_PLUS2.plot(precip_set['time'][:],precip2,'-',color='xkcd:lightblue',alpha=0.9)
days = []
for date in date_list:
    day=date+datetime.timedelta(hours=12)
    days.append(day)

# ax_PLUS2.plot(days,daily_precip,'-',color='xkcd:lightblue')
# ax_PLUS2.plot(days,daily_precip,'-',color='xkcd:lightblue')

ax_PLUS2.set_ylabel('Precipitation [$kg m^{-2} s^{-1}$]',fontsize=14,color='tab:blue')


# new_precip[precip>0.00003] = 1.
# new_precip[precip<=0.00003] = -1.
ax2_PLUS = ax1.twinx()
ax3_PLUS = ax1a.twinx()
ax2_PLUS.plot(precip_set['time'][:],precip,'-',color='xkcd:lightblue')
ax2_PLUS.plot(precip_set['time'][:],precip2,'-',color='xkcd:lightblue')
ax2_PLUS.set_ylabel('Precipitation [$kg m^{-2} s^{-1}$]',fontsize=14,color='tab:blue')
ax3_PLUS.plot(precip_set['time'][:],precip,'-',color='xkcd:lightblue',alpha=0.9)
ax3_PLUS.plot(precip_set['time'][:],precip2,'-',color='xkcd:lightblue',alpha=0.9)
days = []
for date in date_list:
    day=date+datetime.timedelta(hours=12)
    days.append(day)

# ax3_PLUS.plot(days,daily_precip,'-',color='xkcd:lightblue')
# ax2_PLUS.plot(days,daily_precip,'-',color='xkcd:lightblue')

ax3_PLUS.set_ylabel('Precipitation [$kg m^{-2} s^{-1}$]',fontsize=14,color='tab:blue')

color = 'tab:red'
import datetime
base = datetime.datetime(2017,1,1)
date_list = [base + datetime.timedelta(days=x) for x in range(365)]
# ax2.plot(date_list,imfs[0]+imfs[1]+imfs[2]+imfs[3]+imfs[4]+imfs[5],color=color)

# ax1.plot(date_list,x2,'-',color=[1.,0.6,0.6],linewidth=2,label='AMSR-2 ($T_B$)')
ax1.plot(date_list,x3,'-',color='green',linewidth=2,label='CFM ($T_P$)')
# ax1.set_ylim([200,265])
ax1.legend(loc='upper center')

# ax1a.plot(date_list,imfs[0]+imfs[1]+imfs[2]+imfs[3],color='k',linewidth=2,label='Residuals ($T_B$ - $T_P$)')
ax1a.plot(date_list,x2/x3,color='k',linewidth=2,label='Residuals ($T_B$ - $T_P$)')

# ax1.set_ylim([0.,3.])
# ax1.set_ylim([-25,25.])
# ax1a.set_ylim([-45,-30])
# ax2.set_ylim([0,0.0007])
ax1.set_ylabel('$T_P$ [K]',fontsize=14)
ax1a.set_ylabel('Emissivity ($T_B$/$T_P$)',fontsize=14)
ax1.set_xlim([datetime.datetime(2017,6,1),datetime.datetime(2017,8,1)])

ax1a.set_xlim([datetime.datetime(2017,6,1),datetime.datetime(2017,8,1)])

ax0.set_xlim([datetime.datetime(2017,6,1),datetime.datetime(2017,8,1)])

# ax3.plot(date_list,imfs[0]+imfs[1]+imfs[2]+imfs[3]-imfs2[0]-imfs2[1]-imfs2[2]-imfs2[3],color='k')
# ax3.set_ylabel('Residuals (K)')
# ax3.set_xlim([datetime.datetime(2017,10,1),datetime.datetime(2017,12,30)])
# ax3.set_ylim([-7,7])

plt.tight_layout()

# plt.figure()
# # TB = x2;
# # TP = x3;
# TB = imfs2[0]+imfs2[1]+imfs2[2]+imfs2[3];
# TP = imfs3[0]+imfs3[1]+imfs3[2]+imfs3[3]
# plt.figure()
# plt.scatter(daily_precip[1:],x2[:-1]/x3[:-1])
# plt.xlim([0,0.005])
# plt.ylim([0,0.90])
# from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(daily_precip[220:330],x2[214:324]/x3[214:324])


# # # plt.xlabel('$T_B$ [K]',fontsize=15)
# # # plt.ylabel('$T_P$ [K]',fontsize=15)

print(r_value)
print(p_value)





# In[53]:


plt.plot(daily_precip)


# In[47]:


daily_precip = np.zeros(365)*np.nan
N = 24
for ind,value in enumerate(daily_precip):
    daily_precip[ind] = sum(precip[(ind)*N-N:(ind*N)])


# In[278]:


from pyhht.visualization import plot_imfs
import numpy as np
from pyhht import EMD
t = np.linspace(0, 365, len(xval['temperature'][0,:]))
x = np.squeeze(np.nanmean(xval['temperature'][0:5,:],axis=0))
# modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
decomposer = EMD(x)
imfs = decomposer.decompose()
f1=plt.figure(figsize=(7,7))
plot_imfs(x, imfs, t) 


# In[282]:


fig, ax1 = plt.subplots(figsize=(12,4))

color = 'tab:blue'
# ax1.plot(precip_set['time'],np.asarray(precip_set['PRECSNO'][:,1,0]),color=color)
# ax1.plot(precip_set['time'],np.asarray(precip_set['PRECSNO'][:,0,1]),color=color)


new_precip = np.asarray(precip_set['PRECSNO'][:,1,0]).copy()

new_precip[precip>0.00003] = 1.
new_precip[precip<=0.00003] = -1.
ax1.plot(precip_set['time'][:],new_precip,'*')

# ax1.plot(np.linspace(0,364,len(np.asarray(precip_set['PRECTOT'][:,0,1]))),np.nansum(np.asarray(precip_set['PRECTOT'][:,0,1]),np.asarray(precip_set['PRECSNO'][:,1,0])),color=color)
# ax1.plot(np.arange(0,364),model_reformat,color=color)
ax1.set_ylabel('MERRA-2 Snowfall [kg m-2 s-1]',color=color)

ax2 = ax1.twinx()
color = 'tab:red'
date_list = [base + datetime.timedelta(days=x) for x in range(365)]
# ax2.plot(date_list,tbv_amsr[:,pointsy,pointsx],color=color)
ax2.plot(date_list2,imfs[1]+imfs[2]+imfs[3]+imfs[4]+imfs[5],color=color)

# ax1.set_xlim([250,300])
ax1.set_ylim([0.,3.])
ax2.set_ylim([-10.,30.])
ax1.set_xlim([datetime.datetime(2017,1,1),datetime.datetime(2017,4,30)])
ax2.set_ylabel('Filtered HHT CFM $T$ (K)',color=color)

# ax2.set_xlim([250,300])
ax1.set_xlim([datetime.datetime(2017,10,1),datetime.datetime(2017,11,30)])


# In[51]:


from scipy import interpolate
days = np.linspace(0,364,len(np.asarray(precip_set['PRECSNO'][:,1,0])))
values = np.asarray(precip_set['PRECSNO'][:,1,0])
f = interpolate.interp1d(days,values)
model_reformat = f(np.arange(0,364))


# In[3]:


from scipy.io import loadmat
xvalkoh = loadmat('Testing Datasets/Khonen_CFM_2017.mat')
xvalpig = loadmat('Testing Datasets/PIG_CFM_2017.mat')

