import matplotlib.pyplot as plt
from spacepy.pybats import IdlFile
import numpy as np
import spiceypy as spice
import pandas as pd
from datetime import datetime, timedelta



spice.furnsh('./kernels/de430.bsp')
spice.furnsh('./kernels/naif0012.tls')
spice.furnsh('./kernels/pck00010.tpc')
spice.furnsh('./kernels/pck00011.tpc')
spice.furnsh('./kernels/earth_000101_240326_240101.bpc')
spice.furnsh('./kernels/earth_000101_240326_240101.cmt')
spice.furnsh('./kernels/mars_iau2000_v1.tpc')
Rs_km = 696300  # km

# %%
data_path = '/Users/ephe/THL8/RayTracing/run_1023/'
file_type = 'box_mhd_4_'
n_iter = 10000
n_time = None

filename = file_type + 'n' + str(int(n_iter)).zfill(8)

print('Reading file: ' + data_path + filename + '.out')
data_box = IdlFile(data_path + filename + '.out')
var_list = list(data_box.keys())
unit_list = data_box.meta['header'].split()[1:]
print('Variables: ', var_list)
print('Units: ', unit_list)

# %%
mp = 1.6726e-24  # g

Vax = np.array(data_box['Bx']) * 1e-5 / np.sqrt(4. * np.pi * data_box['Rho']) * 1e3  # m/s
Vay = np.array(data_box['By']) * 1e-5 / np.sqrt(4. * np.pi * data_box['Rho']) * 1e3
Vaz = np.array(data_box['Bz']) * 1e-5 / np.sqrt(4. * np.pi * data_box['Rho']) * 1e3

Vax[np.isnan(Vax)] = 0.
Vay[np.isnan(Vay)] = 0.
Vaz[np.isnan(Vaz)] = 0.

Vswx = np.array(data_box['Ux']) * 1e3  # m /s
Vswy = np.array(data_box['Uy']) * 1e3
Vswz = np.array(data_box['Uz']) * 1e3
Vswx[np.isnan(Vswx)] = 0.
Vswy[np.isnan(Vswy)] = 0.
Vswz[np.isnan(Vswz)] = 0.

gamma = 5. / 3.
Cs = np.sqrt(gamma * data_box['P'] / data_box['Rho']) * 1e-2  # m/s
Cs[np.isnan(Cs)] = 0.

gridx_Rs = np.array(data_box['x'])  # Rs
gridy_Rs = np.array(data_box['y'])
gridz_Rs = np.array(data_box['z'])

GRIDX, GRIDY, GRIDZ = np.meshgrid(gridx_Rs,gridy_Rs,gridz_Rs,indexing='ij')

Bx,By,Bz = np.array(data_box['Bx']),np.array(data_box['By']),np.array(data_box['Bz'])
Br = (Bx*GRIDX+By*GRIDY+Bz*GRIDZ)/np.sqrt(GRIDX**2+GRIDY**2+GRIDZ**2)

Vswr = (Vswx*GRIDX+Vswy*GRIDY+Vswz*GRIDZ)/np.sqrt(GRIDX**2+GRIDY**2+GRIDZ**2)/1.e3 #km/s

d_i = 2.28e7 / np.sqrt(data_box['Rho'] / mp) * 1e-2  # m
omega_i = 1.32e3 * np.sqrt(data_box['Rho'] / mp)  # rad/sec


import pyvista as pv





dimensions = data_box['grid']
spacing = (abs(gridx_Rs[1] - gridx_Rs[0]), abs(gridy_Rs[1] - gridy_Rs[0]), abs(gridz_Rs[1] - gridz_Rs[0]))
origin = (gridx_Rs[0], gridy_Rs[0], gridz_Rs[0])
box_grid = pv.UniformGrid(dimensions=(dimensions[0], dimensions[1], dimensions[2]), spacing=spacing,
                          origin=origin)
lgRho = np.log10(data_box['Rho'])
lgRho[np.isinf(lgRho)] = np.nan
box_grid.point_data['lg(Rho)'] = lgRho.ravel('F')

box_grid.point_data['Br'] = Br.ravel(order='F')
isos_br = box_grid.contour(scalars='Br',isosurfaces=1,rng=[0.,0.])

box_grid.point_data['Vr'] = Vswr.ravel(order='F')
box_grid.point_data['Vx'] = Vswx.ravel(order='F')
box_grid.point_data['Vy'] = Vswy.ravel(order='F')
box_grid.point_data['Vz'] = Vswz.ravel(order='F')



def create_epoch(range_dt, step_td):
    beg_dt = range_dt[0]
    end_dt = range_dt[1]
    return [beg_dt + n * step_td for n in range((end_dt - beg_dt) // step_td)]

def get_body_pos(bodyName, epochDt, coord='IAU_SUN'):
    epochEt = spice.datetime2et(epochDt)
    bodyPos, _ = spice.spkpos(bodyName, epochEt, coord, 'NONE', 'SUN')
    return bodyPos


startDt = datetime(2021,9,25)
endDt = datetime(2021,10,23)
stepDt = timedelta(hours=4)
epochDt = create_epoch([startDt, endDt], stepDt)
earthPos = np.array(get_body_pos('EARTH', epochDt, ))
marsPos = np.array(get_body_pos('MARS BARYCENTER', epochDt))
POS_type='EM'
if POS_type == 'EM':
    vecPOSn = np.array((earthPos - marsPos) / np.linalg.norm((earthPos - marsPos).T))
elif POS_type == 'ES':
    vecPOSn = np.array(earthPos / np.linalg.norm(earthPos.T))
projPos = np.zeros_like(earthPos)
for i in range(len(epochDt)):
    OE = earthPos[i]
    OM = marsPos[i]
    Nvec = vecPOSn[i]
    OP = (np.dot(OE, Nvec) * OM - np.dot(OM, Nvec) * OE) / (np.dot(OE, Nvec) - np.dot(OM, Nvec))
    projPos[i] = OP/Rs_km
# %%


result_path = 'export/five_cases/'

p = pv.Plotter(window_size=[2048,1536],theme=pv.themes.DarkTheme())
for i in [3,4]:

    result_name = 'Slow_Forward_Case'+str(i+1)+'_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path+result_name)
    pos_list = np.stack([result_df['pos_x_Rs'],result_df['pos_y_Rs'],result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'],result_df['By_G'],result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'],result_df['k_y_1/m'],result_df['k_z_1/m']]).T
    pos_line = pv.lines_from_points(pos_list)
    p.add_mesh(pos_line.tube(radius=0.2), color='pink',label='Slow_Forward')
    casePos = pos_list[0]

    result_name = 'Fast_Forward_Case' + str(i + 1) + '_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path + result_name)
    pos_list = np.stack([result_df['pos_x_Rs'], result_df['pos_y_Rs'], result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'], result_df['By_G'], result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'], result_df['k_y_1/m'], result_df['k_z_1/m']]).T
    pos_line = pv.lines_from_points(pos_list)
    p.add_mesh(pos_line.tube(radius=0.2), color='lightgreen',label='Fast_Forward')

    # result_name = 'Alfven_Forward_Case' + str(i + 1) + '_xh=0.01_kh=1e-10).csv'
    # result_df = pd.read_csv(result_path + result_name)
    # pos_list = np.stack([result_df['pos_x_Rs'], result_df['pos_y_Rs'], result_df['pos_z_Rs']]).T
    # B_list = np.stack([result_df['Bx_G'], result_df['By_G'], result_df['Bz_G']]).T
    # k_list = np.stack([result_df['k_x_1/m'], result_df['k_y_1/m'], result_df['k_z_1/m']]).T
    # pos_line = pv.lines_from_points(pos_list)
    # p.add_mesh(pos_line.tube(radius=0.2), color='lightblue',label='Alfven_Forward')

    result_name = 'Slow_Backward_Case'+str(i+1)+'_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path+result_name)
    pos_list = np.stack([result_df['pos_x_Rs'],result_df['pos_y_Rs'],result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'],result_df['By_G'],result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'],result_df['k_y_1/m'],result_df['k_z_1/m']]).T
    error_ind = np.where(result_df['omega_i_Hz'] > 2.e6)[0][0]
    # pos_list[error_ind:,:]=np.nan
    # B_list[error_ind:,:] = np.nan
    # k_list[error_ind:,:] = np.nan

    pos_line = pv.lines_from_points(pos_list)
    p.add_mesh(pos_line.tube(radius=0.2),color='red',label='Slow_Backward')
    minr_ind = np.argmin(np.array(result_df['pos_r_Rs']))


    print('footpoint_rlonlat(deg): ',result_df['pos_r_Rs'][minr_ind],np.rad2deg(result_df['pos_lon_rad'][minr_ind]),np.rad2deg(result_df['pos_lat_rad'][minr_ind]))

    result_name = 'Fast_Backward_Case' + str(i + 1) + '_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path + result_name)
    pos_list = np.stack([result_df['pos_x_Rs'], result_df['pos_y_Rs'], result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'], result_df['By_G'], result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'], result_df['k_y_1/m'], result_df['k_z_1/m']]).T
    pos_line = pv.lines_from_points(pos_list)
    p.add_mesh(pos_line.tube(radius=0.2), color='green',label='Fast_Backward')

    # result_name = 'Alfven_Backward_Case' + str(i + 1) + '_xh=0.01_kh=1e-10).csv'
    # result_df = pd.read_csv(result_path + result_name)
    # pos_list = np.stack([result_df['pos_x_Rs'], result_df['pos_y_Rs'], result_df['pos_z_Rs']]).T
    # B_list = np.stack([result_df['Bx_G'], result_df['By_G'], result_df['Bz_G']]).T
    # k_list = np.stack([result_df['k_x_1/m'], result_df['k_y_1/m'], result_df['k_z_1/m']]).T
    # pos_line = pv.lines_from_points(pos_list)
    # p.add_mesh(pos_line.tube(radius=0.2), color='blue',label='Alfven_Backward')

sun_sphere = pv.Sphere(1.5)
sun_sphere = sun_sphere.sample(box_grid)

projpos_line = pv.lines_from_points(projPos)
projpos_line = projpos_line.sample(box_grid)

p.add_mesh(projpos_line.tube(radius=0.2),color='white')
p.add_mesh(box_grid.slice(normal=[0,0,1]),clim=[-22,-15],cmap='jet',opacity=0.5)
p.add_mesh(sun_sphere, scalars='Br',cmap='seismic',clim=[-1.,1.])
p.show_axes()
# p.add_legend([['Slow_Forward','pink'],['Fast_Forward','lightgreen'],#['Alfven_Forward','lightblue'],
#               ['Slow_Backward','red'],['Fast_Backward','green'],])#['Alfven_Backward','blue']])
p.show_grid()
# p.show()
# p.camera.position = (-20,60,30)

case_dt = datetime(2021,10,4,8)
case_ind = np.argmin(abs((np.array(epochDt)-case_dt)/timedelta(days=1)))
p.camera.focal_point=casePos
p.camera.position = casePos*1.5+np.array([0,0,2])
# p.camera.position = casePos*3.+np.array([0,0,7])
# p.camera.position = np.array([-20,0,20])

p.show(auto_close=False)

p.save_graphic('zoom_case45.pdf')

p.close()

# %%
obs_time = np.array([datetime(2021,10,1,7,12),datetime(2021,10,1,7,28),
                     datetime(2021,10,12,9,55),
                     datetime(2021,10,23,4,45),datetime(2021,10,23,6,5)])
obs_Vr = np.array([85.77,136.29,161.55,90.82,141.34])
obs_Vx = np.array([-9.366585582,-40.50130796,-127.9369512,8.775763386,10.52712678])
obs_Vy = np.array([0.184111948,0.901671455,-96.03392846,27.93541434,32.00513614])
obs_Vz = np.array([85.25700362,130.1244943,-22.52547862,85.97491334,137.2661661])
#vp_x	vp_y	vp_z
# -9.366585582	0.184111948	85.25700362
# -40.50130796	0.901671455	130.1244943
# -127.9369512	-96.03392846	-22.52547862
# 8.775763386	27.93541434	85.97491334
# 10.52712678	32.00513614	137.2661661
plt.figure()
# plt.plot(epochDt,projpos_line['Vr'].ravel(),label='simu_Vr')
plt.plot(epochDt,projpos_line['Vx'].ravel()/1000,label='model_Vx')
plt.plot(epochDt,projpos_line['Vy'].ravel()/1000,label='model_Vy')
plt.plot(epochDt,projpos_line['Vz'].ravel()/1000,label='model_Vz')
plt.xlabel('Time')
plt.ylabel('Solar Wind Velocity [km/s]')
plt.scatter(obs_time,obs_Vx,label='obs_Vx')
plt.scatter(obs_time,obs_Vy,label='obs_Vy')
plt.scatter(obs_time,obs_Vz,label='obs_Vz')
plt.legend()
plt.title('Solar Wind Velocity')
plt.show()

