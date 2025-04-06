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


plt.figure(dpi=200,figsize=(6,2),)
plt.subplots_adjust(right=0.65,left=0.1,bottom=0.2,top=0.8)
# plt.tight_layout()
colors = ['#4575B4', '#D73027', '#FC8D59', '#91BFDB', '#FEE090']
for i in range(5):
    # plt.subplot(2,1,1)
    result_name = 'Slow_Forward_Case'+str(i+1)+'_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path+result_name)
    pos_list = np.stack([result_df['pos_x_Rs'],result_df['pos_y_Rs'],result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'],result_df['By_G'],result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'],result_df['k_y_1/m'],result_df['k_z_1/m']]).T
    pos_r_list = np.array(result_df['pos_r_Rs'])
    theta_kb_list = np.linspace(0,0,len(result_df))
    for j in range(len(result_df)):
        theta_kb_list[j] = np.rad2deg(np.arccos(np.dot(B_list[j], k_list[j]) / np.linalg.norm(B_list[j]) / np.linalg.norm(k_list[j])))
    # theta_kb_list[result_df['omega_i_H]

    plt.plot(pos_r_list, theta_kb_list,c=colors[i],linestyle='--',label='Forward_Case'+str(i+1),linewidth=2)






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
    pos_r_list = np.array(result_df['pos_r_Rs'])
    theta_kb_list = np.linspace(0, 0, len(result_df))
    for j in range(len(result_df)):
        theta_kb_list[j] = np.rad2deg(
            np.arccos(np.dot(B_list[j], k_list[j]) / np.linalg.norm(B_list[j]) / np.linalg.norm(k_list[j])))
    error_ind = np.where(result_df['omega_i_Hz']>2.e6)[0][0]
    # theta_kb_list[result_df['omega_i_Hz']>2.e6] = np.nan
    theta_kb_list[error_ind:]=np.nan
    # error_ind = np.argwhere(theta_kb_list==np.nan)
    plt.plot(pos_r_list, theta_kb_list,c=colors[i],label='Backward_Case'+str(i+1))
    # plt.subplot(2, 1, 2)
    # plt.plot(pos_r_list, result_df['omega_i_Hz'])

plt.xlabel('r (Rs)')
plt.ylabel(r'$\theta_{kb}$')
plt.xlim([1,20])
plt.ylim([10,170])
plt.yticks(np.arange(0,180,30))
plt.xticks(np.arange(1,21,2))
plt.legend(ncol=1,bbox_to_anchor=(1.05, 1.3),
         loc='upper left',frameon=False,shadow=False,
         borderaxespad=0.)
plt.title(r'$\theta_{kb}$ evolution of slow mode')
plt.grid(True,linestyle=':')
plt.show()


