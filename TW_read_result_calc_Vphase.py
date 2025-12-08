import matplotlib.pyplot as plt
from spacepy.pybats import IdlFile
import numpy as np
import spiceypy as spice
import pandas as pd
from datetime import datetime, timedelta
from ray_tracer import get_Vsw



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
dt = 60 * 1.  # s
Nt = 4000  # steps




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


result_path = 'export/TW_coro_results/'
color_list = ['#1f77b4',  # 蓝色
              '#ff7f0e',  # 橙色
              '#2ca02c',  # 绿色
              '#d62728',  # 红色
              '#9467bd']  # 紫色
plt.figure(dpi=300,figsize=(8,8))
for i in range(5):
    result_name = 'Slow_Forward_Case'+str(i+1)+'_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path+result_name)
    pos_list = np.stack([result_df['pos_x_Rs'],result_df['pos_y_Rs'],result_df['pos_z_Rs']]).T
    casePos = pos_list[0]
    B_list = np.stack([result_df['Bx_G'],result_df['By_G'],result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'],result_df['k_y_1/m'],result_df['k_z_1/m']]).T
    r_list = np.array(result_df['pos_r_Rs'])
    V_list_slow_forward = []
    V_list_slow_forward_pos = []
    V_sw_r_lst = []
    V_sw_lst = get_Vsw(pos_list)/1.e3
    for i_pos, pos in enumerate(pos_list[:-1]):
        V_pos = (pos_list[i_pos+1]-pos)/dt*Rs_km
        V_pos_r = np.dot(V_pos, pos) / np.linalg.norm(pos)
        V_list_slow_forward_pos.append(V_pos_r)
        V_sw = V_sw_lst[:,i_pos]
        V_sw_r = np.dot(V_sw, pos) / np.linalg.norm(pos)
        V_sw_r_lst.append(V_sw_r)
        if i_pos==0:
            print(V_sw_r)
        V_relative = V_pos-V_sw
        V_relative_r = np.dot(V_relative, pos) / np.linalg.norm(pos)
        V_list_slow_forward.append(V_relative_r)
    plt.subplot(2, 1, 1)
    plt.plot(r_list[:-1], V_sw_r_lst,'--',color=color_list[i],label='BG Vr Case'+str(i+1))
    plt.scatter(r_list[:-1], V_list_slow_forward_pos,s=1,marker='+',color=color_list[i],label='Traced Vr Case'+str(i+1))
    plt.subplot(2, 1, 2)
    plt.scatter(r_list[:-1],V_list_slow_forward,s=1,marker='o',color=color_list[i],label='Relative Vr Case'+str(i+1))


    result_name = 'Slow_Backward_Case'+str(i+1)+'_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path+result_name)
    pos_list = np.stack([result_df['pos_x_Rs'],result_df['pos_y_Rs'],result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'],result_df['By_G'],result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'],result_df['k_y_1/m'],result_df['k_z_1/m']]).T
    if sum(result_df['omega_i_Hz'] > 2.e6)>0:
        error_ind = np.where(result_df['omega_i_Hz'] > 2.e6)[0][0]

    r_list = np.array(result_df['pos_r_Rs'])
    V_list_slow_backward = []
    V_list_slow_backward_pos = []
    V_sw_r_lst = []
    V_sw_lst = get_Vsw(pos_list)/1.e3
    for i_pos, pos in enumerate(pos_list[:-1]):
        V_pos = (pos-pos_list[i_pos + 1]) / dt * Rs_km
        V_pos_r = np.dot(V_pos, pos) / np.linalg.norm(pos)
        V_list_slow_backward_pos.append(V_pos_r)
        V_sw = V_sw_lst[:, i_pos]
        V_sw_r = np.dot(V_sw, pos) / np.linalg.norm(pos)
        V_sw_r_lst.append(V_sw_r)
        V_relative = V_pos - V_sw
        V_relative_r = np.dot(V_relative, pos) / np.linalg.norm(pos)
        V_list_slow_backward.append(V_relative_r)
    plt.subplot(2, 1, 1)
    plt.plot(r_list[:-1], V_sw_r_lst, '--', color=color_list[i])
    plt.scatter(r_list[:-1], V_list_slow_backward_pos,s=1,marker='+', color=color_list[i])
    plt.subplot(2, 1, 2)
    plt.scatter(r_list[:-1], V_list_slow_backward, s=1, marker='o', color=color_list[i])

plt.subplot(2,1,1)
plt.xlim([1,20])

# plt.xlim([1,20])
plt.legend(ncol=2)
plt.xlabel('r [Rs]')
plt.ylabel('Velocity [km/s]')
plt.subplot(2,1,2)
plt.xlim([1,20])
plt.ylim([0.,120])
plt.xlabel('r [Rs]')
plt.legend(ncol=2)
plt.ylabel('Velocity [km/s]')
plt.title('Relative Velocity')
plt.tight_layout()
plt.show()

sun_sphere = pv.Sphere(1.5)
sun_sphere = sun_sphere.sample(box_grid)

projpos_line = pv.lines_from_points(projPos)
projpos_line = projpos_line.sample(box_grid)


case_dt = datetime(2021,10,4,8)
case_ind = np.argmin(abs((np.array(epochDt)-case_dt)/timedelta(days=1)))
# p.camera.focal_point=casePos
# p.camera.position = casePos*1.5+np.array([0,0,2])
# p.camera.position = casePos*3.+np.array([0,3,7])
# p.camera.position = casePos*3.+np.array([0,5,7])
# p.camera.position = np.array([-20,0,20])


# %%
# obs_time = np.array([datetime(2021,10,1,7,12),datetime(2021,10,1,7,28),
#                      datetime(2021,10,12,9,55),
#                      datetime(2021,10,23,4,45),datetime(2021,10,23,6,5)])
# obs_Vr = np.array([85.77,136.29,161.55,90.82,141.34])
# obs_Vx = np.array([-9.366585582,-40.50130796,-127.9369512,8.775763386,10.52712678])
# obs_Vy = np.array([0.184111948,0.901671455,-96.03392846,27.93541434,32.00513614])
# obs_Vz = np.array([85.25700362,130.1244943,-22.52547862,85.97491334,137.2661661])
# #vp_x	vp_y	vp_z
# # -9.366585582	0.184111948	85.25700362
# # -40.50130796	0.901671455	130.1244943
# # -127.9369512	-96.03392846	-22.52547862
# # 8.775763386	27.93541434	85.97491334
# # 10.52712678	32.00513614	137.2661661
# plt.figure()
# # plt.plot(epochDt,projpos_line['Vr'].ravel(),label='simu_Vr')
# plt.plot(epochDt,projpos_line['Vx'].ravel()/1000,label='model_Vx')
# plt.plot(epochDt,projpos_line['Vy'].ravel()/1000,label='model_Vy')
# plt.plot(epochDt,projpos_line['Vz'].ravel()/1000,label='model_Vz')
# plt.xlabel('Time')
# plt.ylabel('Solar Wind Velocity [km/s]')
# plt.scatter(obs_time,obs_Vx,label='obs_Vx')
# plt.scatter(obs_time,obs_Vy,label='obs_Vy')
# plt.scatter(obs_time,obs_Vz,label='obs_Vz')
# plt.legend()
# plt.title('Solar Wind Velocity')
# plt.show()

