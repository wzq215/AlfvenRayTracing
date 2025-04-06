from spacepy.pybats import IdlFile
import PIL.Image
import numpy as np
import spiceypy as spice
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.offline


spice.furnsh('./kernels/de430.bsp')
spice.furnsh('./kernels/naif0012.tls')
spice.furnsh('./kernels/pck00010.tpc')
spice.furnsh('./kernels/pck00011.tpc')
spice.furnsh('./kernels/earth_000101_240326_240101.bpc')
spice.furnsh('./kernels/earth_000101_240326_240101.cmt')
spice.furnsh('./kernels/mars_iau2000_v1.tpc')
Rs_km = 696300  # km

# %%
data_path = '/Users/ephe/THL8/Test_SC230315_2304/output_SC_230315/SC/'
data_path = '/Users/ephe/THL8/RayTracing/output_01/SC/'
file_type = 'box_mhd_4_'
n_iter = 5900
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

Bx,By,Bz = np.array(data_box['Bx']),np.array(data_box['By']),np.array(data_box['Bz'])
Br = (Bx*gridx_Rs+By*gridy_Rs+Bz*gridz_Rs)/np.sqrt(gridx_Rs**2+gridy_Rs**2+gridz_Rs**2)

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

p = pv.Plotter()
for i in [4]:
    fig=plt.figure()

    result_name = 'Slow_Forward_Case'+str(i+1)+'_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path+result_name)
    pos_list = np.stack([result_df['pos_x_Rs'],result_df['pos_y_Rs'],result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'],result_df['By_G'],result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'],result_df['k_y_1/m'],result_df['k_z_1/m']]).T

    plt.subplot(3, 1, 1)
    plt.plot(result_df['pos_r_Rs'], result_df['omega_Hz'],color='pink',label='Slow+Forward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('omega [Hz]')

    plt.subplot(3, 1, 2)
    plt.plot(result_df['pos_r_Rs'],
             np.linalg.norm(k_list,axis=1)*np.array([np.float(strtmp[1:-1]) for strtmp in result_df['d_i_m'].values])
             ,color='pink',label='Slow+Forward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('k*d_i')

    theta_kb_lst = np.array(
        [np.rad2deg(np.arccos(np.dot(k_list[i], B_list[i]) / np.linalg.norm(k_list[i]) / np.linalg.norm(B_list[i]))) for
         i in range(len(k_list))])
    plt.subplot(3, 1, 3)
    plt.plot(result_df['pos_r_Rs'], theta_kb_lst,color='pink',label='Slow+Forward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('theta_kb [deg]')

    result_name = 'Slow_Backward_Case'+str(i+1)+'_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path+result_name)
    pos_list = np.stack([result_df['pos_x_Rs'],result_df['pos_y_Rs'],result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'],result_df['By_G'],result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'],result_df['k_y_1/m'],result_df['k_z_1/m']]).T
    # pos_line = pv.lines_from_points(pos_list)
    # p.add_mesh(pos_line.tube(radius=0.2),color='red')

    plt.subplot(3, 1, 1)
    plt.plot(result_df['pos_r_Rs'], result_df['omega_Hz'],color='red',label='Slow+Backward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('omega [Hz]')
    plt.xlim([0,25])


    plt.subplot(3, 1, 2)
    plt.plot(result_df['pos_r_Rs'], np.linalg.norm(k_list,axis=1)*np.array([np.float(strtmp[1:-1]) for strtmp in result_df['d_i_m'].values])
             ,color='red',label='Slow+Backward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('k*d_i')
    plt.xlim([0, 25])

    theta_kb_lst = np.array([np.rad2deg(np.arccos(np.dot(k_list[i],B_list[i])/np.linalg.norm(k_list[i])/np.linalg.norm(B_list[i]))) for i in range(len(k_list))])
    plt.subplot(3, 1, 3)
    plt.plot(result_df['pos_r_Rs'], theta_kb_lst,color='red',label='Slow+Backward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('theta_kb [deg]')
    plt.xlim([0, 25])
    plt.ylim([0,180])

    # plt.suptitle('Slow Mode (Case '+str(i+1)+')')

    result_name = 'Fast_Forward_Case' + str(i + 1) + '_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path + result_name)
    pos_list = np.stack([result_df['pos_x_Rs'], result_df['pos_y_Rs'], result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'], result_df['By_G'], result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'], result_df['k_y_1/m'], result_df['k_z_1/m']]).T

    plt.subplot(3, 1, 1)
    plt.plot(result_df['pos_r_Rs'], result_df['omega_Hz'], color='lightgreen', label='Fast+Forward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('omega [Hz]')

    plt.subplot(3, 1, 2)
    plt.plot(result_df['pos_r_Rs'],
             np.linalg.norm(k_list, axis=1) * np.array([np.float(strtmp[1:-1]) for strtmp in result_df['d_i_m'].values])
             , color='lightgreen', label='Fast+Forward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('k*d_i')

    theta_kb_lst = np.array(
        [np.rad2deg(np.arccos(np.dot(k_list[i], B_list[i]) / np.linalg.norm(k_list[i]) / np.linalg.norm(B_list[i]))) for
         i in range(len(k_list))])
    plt.subplot(3, 1, 3)
    plt.plot(result_df['pos_r_Rs'], theta_kb_lst, color='lightgreen', label='Fast+Forward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('theta_kb [deg]')

    result_name = 'Fast_Backward_Case' + str(i + 1) + '_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path + result_name)
    pos_list = np.stack([result_df['pos_x_Rs'], result_df['pos_y_Rs'], result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'], result_df['By_G'], result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'], result_df['k_y_1/m'], result_df['k_z_1/m']]).T
    # pos_line = pv.lines_from_points(pos_list)
    # p.add_mesh(pos_line.tube(radius=0.2),color='red')

    plt.subplot(3, 1, 1)
    plt.plot(result_df['pos_r_Rs'], result_df['omega_Hz'], color='green', label='Fast+Backward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('omega [Hz]')
    plt.xlim([0, 25])

    plt.subplot(3, 1, 2)
    plt.plot(result_df['pos_r_Rs'],
             np.linalg.norm(k_list, axis=1) * np.array([np.float(strtmp[1:-1]) for strtmp in result_df['d_i_m'].values])
             , color='green', label='Fast+Backward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('k*d_i')
    plt.xlim([0, 25])

    theta_kb_lst = np.array(
        [np.rad2deg(np.arccos(np.dot(k_list[i], B_list[i]) / np.linalg.norm(k_list[i]) / np.linalg.norm(B_list[i]))) for
         i in range(len(k_list))])
    plt.subplot(3, 1, 3)
    plt.plot(result_df['pos_r_Rs'], theta_kb_lst, color='green', label='Fast+Backward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('theta_kb [deg]')
    plt.xlim([0, 25])
    plt.ylim([0, 180])

    result_name = 'Alfven_Forward_Case' + str(i + 1) + '_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path + result_name)
    pos_list = np.stack([result_df['pos_x_Rs'], result_df['pos_y_Rs'], result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'], result_df['By_G'], result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'], result_df['k_y_1/m'], result_df['k_z_1/m']]).T

    plt.subplot(3, 1, 1)
    plt.plot(result_df['pos_r_Rs'], result_df['omega_Hz'], color='lightblue', label='Alfven+Forward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('omega [Hz]')

    plt.subplot(3, 1, 2)
    plt.plot(result_df['pos_r_Rs'],
             np.linalg.norm(k_list, axis=1) * np.array([np.float(strtmp[1:-1]) for strtmp in result_df['d_i_m'].values])
             , color='lightblue', label='Alfven+Forward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('k*d_i')

    theta_kb_lst = np.array(
        [np.rad2deg(np.arccos(np.dot(k_list[i], B_list[i]) / np.linalg.norm(k_list[i]) / np.linalg.norm(B_list[i]))) for
         i in range(len(k_list))])
    plt.subplot(3, 1, 3)
    plt.plot(result_df['pos_r_Rs'], theta_kb_lst, color='lightblue', label='Alfven+Forward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('theta_kb [deg]')

    result_name = 'Alfven_Backward_Case' + str(i + 1) + '_xh=0.01_kh=1e-10).csv'
    result_df = pd.read_csv(result_path + result_name)
    pos_list = np.stack([result_df['pos_x_Rs'], result_df['pos_y_Rs'], result_df['pos_z_Rs']]).T
    B_list = np.stack([result_df['Bx_G'], result_df['By_G'], result_df['Bz_G']]).T
    k_list = np.stack([result_df['k_x_1/m'], result_df['k_y_1/m'], result_df['k_z_1/m']]).T
    # pos_line = pv.lines_from_points(pos_list)
    # p.add_mesh(pos_line.tube(radius=0.2),color='red')

    plt.subplot(3, 1, 1)
    plt.plot(result_df['pos_r_Rs'], result_df['omega_Hz'], color='blue', label='Alfven+Backward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('omega [Hz]')
    plt.xlim([1, 25])
    # plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(result_df['pos_r_Rs'],
             np.linalg.norm(k_list, axis=1) * np.array([np.float(strtmp[1:-1]) for strtmp in result_df['d_i_m'].values])
             , color='blue', label='Alfven+Backward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('k*d_i')
    plt.xlim([1, 25])
    plt.legend()

    theta_kb_lst = np.array(
        [np.rad2deg(np.arccos(np.dot(k_list[i], B_list[i]) / np.linalg.norm(k_list[i]) / np.linalg.norm(B_list[i]))) for
         i in range(len(k_list))])
    plt.subplot(3, 1, 3)
    plt.plot(result_df['pos_r_Rs'], theta_kb_lst, color='blue', label='Alfven+Backward')
    plt.xlabel('Radius [Rs]')
    plt.ylabel('theta_kb [deg]')
    plt.xlim([1, 25])
    plt.ylim([0, 180])
    # plt.legend()

    plt.suptitle('Case '+str(i+1))



    plt.show()
    # result_name = 'Fast_Backward_Case' + str(i + 1) + '_xh=0.01_kh=1e-10).csv'
    # result_df = pd.read_csv(result_path + result_name)
    # pos_list = np.stack([result_df['pos_x_Rs'], result_df['pos_y_Rs'], result_df['pos_z_Rs']]).T
    # B_list = np.stack([result_df['Bx_G'], result_df['By_G'], result_df['Bz_G']]).T
    # k_list = np.stack([result_df['k_x_1/m'], result_df['k_y_1/m'], result_df['k_z_1/m']]).T
    # pos_line = pv.lines_from_points(pos_list)
    # p.add_mesh(pos_line.tube(radius=0.2), color='green')
    #
    # result_name = 'Alfven_Backward_Case' + str(i + 1) + '_xh=0.01_kh=1e-10).csv'
    # result_df = pd.read_csv(result_path + result_name)
    # pos_list = np.stack([result_df['pos_x_Rs'], result_df['pos_y_Rs'], result_df['pos_z_Rs']]).T
    # B_list = np.stack([result_df['Bx_G'], result_df['By_G'], result_df['Bz_G']]).T
    # k_list = np.stack([result_df['k_x_1/m'], result_df['k_y_1/m'], result_df['k_z_1/m']]).T
    # pos_line = pv.lines_from_points(pos_list)
    # p.add_mesh(pos_line.tube(radius=0.2), color='blue')
    # p.add_arrows(pos_list, B_list, mag=1e3, color='black')
    # p.add_arrows(pos_list, k_list, mag=1e3, color='blue')

    # result_name = 'Slow_Backward_Case'+str(i+1)+'_reverse_xh=0.01_kh=1e-10).csv'
    # result_df = pd.read_csv(result_path+result_name)
    # pos_list = np.stack([result_df['pos_x_Rs'],result_df['pos_y_Rs'],result_df['pos_z_Rs']]).T
    # B_list = np.stack([result_df['Bx_G'],result_df['By_G'],result_df['Bz_G']]).T
    # k_list = np.stack([result_df['k_x_1/m'],result_df['k_y_1/m'],result_df['k_z_1/m']]).T
    # pos_line = pv.lines_from_points(pos_list)
    # p.add_mesh(pos_line.tube(radius=0.1),color='blue')

projpos_line = pv.lines_from_points(projPos)
p.add_mesh(projpos_line.tube(radius=0.2),color='black')
# p.add_mesh_slice_orthogonal(box_grid, clim=[-22, -15], cmap='jet')
p.add_mesh_slice(box_grid,normal=[0,0,1],clim=[-22,-15],cmap='jet',opacity=0.5)
# p.add_volume(box_grid,scalars='lg(Rho)',clim=[-22,-15],)
p.add_mesh(pv.Sphere(1.))
# p.add_mesh(isos_br,opacity=0.5)
# p.add_mesh_slice_orthogonal(box_grid, scalars='Br',clim=[-0.1,0.1])
# p.show_grid()
p.show_axes()
p.show()


