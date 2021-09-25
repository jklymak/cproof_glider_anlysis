
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import datetime
import matplotlib.units as munits
import jmkfigure
converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter

lon0 = -129
lat0 =  51.5
def get_xy_lonlat(lon, lat, lon0, lat0):
    x = (lon - lon0) * np.cos(np.deg2rad(lat0)) * 60 * 1.852
    y = (lat - lat0) * 60 * 1.852
    return x, y

deploy_name = 'dfo-bb046-20200717'

Waypoints={}
Waypoints['QCS02'] = {'lon': -128 - 17/60 - 34 /3600, 'lat': 51 + 41/60 + 39/3600}
Waypoints['SS5'] = {'lon': -128 - 30/60, 'lat': 51 + 28/60}
Waypoints['MPA1'] = {'lon': -128 - 43/60 -44 / 3600, 'lat': 51 + 23/60}
Waypoints['MPA2'] = {'lon': -129 - 55 / 3600, 'lat': 51 + 19/60 + 14 / 3600}
Waypoints['Shelf'] = {'lon': -129 - 49 /60 - 51 / 3600, 'lat': 51 + 5/60 + 13 / 3600}

wps = np.zeros((len(Waypoints.keys()), 2))
for nn, k in enumerate(Waypoints.keys()):
    print('k', k)
    wp = Waypoints[k]
    x, y  = get_xy_lonlat(wp['lon'], wp['lat'], lon0, lat0)
    wps[nn, 0] = x
    wps[nn, 1] = y
print(wps)


def get_xy(ds, lat0, lon0):
    x, y = get_xy_lonlat(ds.longitude, ds.latitude, lon0, lat0)
    ds['x'] = x
    ds.x.attrs['units'] = f'km east of {lon0}'
    ds['y'] = y
    ds.y.attrs['units'] = f'km north of {lat0}'
    return ds


def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    dsq = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / dsq

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = np.sqrt(dx*dx + dy*dy)

    return dist

def get_alongx(wps, x, y):

    distline = np.zeros(len(wps)-1)

    for j in range(0, len(wps)-1):
        distline[j] = np.sqrt((wps[j+1, 0] - wps[j, 0])**2 + (wps[j+1, 1] - wps[j, 1])**2)

    distline = np.hstack([0, np.cumsum(distline)])
    print(wps)
    print(distline)
    ind = np.zeros(len(x))
    for j in range(len(x)):
        thedist = np.Inf
        for i in range(len(wps)-1):
            dd = dist(wps[i][0], wps[i][1], wps[i+1][0], wps[i+1][1], x[j], y[j])
            if dd < thedist:
                thedist = dd
                ind[j] = i

    alongx = x * 0.
    for i in range(len(x)):
        # get the distance along the line....
        indd = int(ind[i])
        x0 = wps[indd][0]
        x1 = wps[indd+1][0]
        y0 = wps[indd][1]
        y1 = wps[indd+1][1]

        xp = x[i] - x0
        yp = y[i] - y0
        dot = xp * (x1 - x0) + yp * (y1 - y0)
        dot = dot / np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        alongx[i] = dot + distline[indd]
    return alongx, ind, distline

proj = ccrs.Mercator(central_longitude=-129.25, min_latitude=50.5, max_latitude=52)


fig = plt.figure(figsize=(8, 5), constrained_layout=False)

ax = fig.add_subplot(projection = proj, facecolor='0.5')

with xr.open_dataset('/Users/jklymak/gliderdata/bathy/british_columbia_3_msl_2013.nc') as topo0:
    topo0 = topo0.isel(lon=slice(6500, 13000, 6), lat=slice(2000, 5500, 6))
    # ax = axs['map']
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.3, color='gray')
    gl.ylabels_right=False
    gl.xlabels_top=False
    pc = ax.pcolormesh(topo0.lon, topo0.lat, topo0.Band1, transform=ccrs.PlateCarree(),
                  rasterized=True, alpha=1, vmin=-500, vmax=0, zorder=0)
    #ax.set_xlim(-131, -127.5)
    ax.set_extent((-130.5, -128, 50.85, 51.75))
    with xr.open_dataset('/Users/jklymak/gliderdata/deployments/dfo-bb046/dfo-bb046-20200717/L2geo-gridfiles/dfo-bb046-20200717_grid.nc') as ds:
        turns = eval(ds.attrs['turn_indices'])
        cols = ['C1', 'C3', 'C4', 'C5']
        for ii in range(len(turns) - 1):
            ind = range(turns[ii], turns[ii+1])
            time = f'{ds.time.values[ind[0]]}'[:16]
            ax.plot(ds.longitude[ind], ds.latitude[ind], '.', markersize=2,
                    transform=ccrs.PlateCarree(), zorder=5, color=cols[ii],
                    label=f'Line {ii+1}: {time}')
        ax.legend(loc=2)
        ds['alongx'], ind, distline = get_alongx(wps, ds.x, ds.y)

    start = f'{ds.time.values[0]}'[:16]
    stop = f'{ds.time.values[-1]}'[:16]
    ax.set_title(f'{deploy_name}: {start} - {stop}', fontsize='medium', loc='left')
    for nn, k in enumerate(Waypoints.keys()):
        wp = Waypoints[k]
        x, y  = get_xy_lonlat(wp['lon'], wp['lat'], lon0, lat0)
        wps[nn, 0] = x
        wps[nn, 1] = y
        print(wp['lon'], wp['lat'], x, y)
        ax.plot(wp['lon'], wp['lat'],  'o', color='0.75', zorder=1, markersize=10, transform=ccrs.PlateCarree(),
               alpha=0.5)
        ax.text(wp['lon'], wp['lat'], f'    {k} {distline[nn]:1.1f} km',
                transform=ccrs.PlateCarree(), fontsize=8, fontstyle='italic', color='1', alpha=0.5,
                verticalalignment='center')
        # print(wp.lon, wp.lat, 'd')

fig.colorbar(pc, ax=ax, shrink=0.6, extend='both')

now = np.datetime64(datetime.datetime.now()) + np.timedelta64(7, 'h')
st = f'{now}'
st = st[:-10]
#fig.canvas.draw()
#fig.savefig('figs/MissionMap.png', dpi=200)
jmkfigure.jmkprint('MissionMap', 'plotJustDeploymentMap.py', dpi=200)
plt.show()
