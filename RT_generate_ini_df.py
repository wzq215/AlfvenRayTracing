import numpy as np

import pandas as pd
import pyvista as pv
from spacepy.pybats import IdlFile

data_path = '/Users/ephe/THL8/Test_SC230315_2304/output_SC_230315/SC/'
# data_path = '/Users/ephe/THL8/RayTracing/output_01/SC/'
file_type = 'box_mhd_4_'
n_iter = 5900
n_time = None

filename = file_type + 'n' + str(int(n_iter)).zfill(8)

print('Reading file: ' + data_path + filename + '.out')
data_box = IdlFile(data_path + filename + '.out')
var_list = list(data_box.keys())
unit_list = data_box.meta['header'].split()[1:]

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
box_grid.point_data['Bx'] = Bx.ravel(order='F')
box_grid.point_data['By'] = By.ravel(order='F')
box_grid.point_data['Bz'] = Bz.ravel(order='F')
box_grid.point_data['Vswx'] = Vswx.ravel(order='F')
box_grid.point_data['Vswy'] = Vswy.ravel(order='F')
box_grid.point_data['Vswz'] = Vswz.ravel(order='F')
sph_msh = pv.Sphere(radius=2,theta_resolution=4,phi_resolution=4)
sph_msh = sph_msh.sample(box_grid)

ini_df = pd.DataFrame()
ini_df['posx_ini'] = np.zeros_like(sph_msh.points[:,0])+2.
ini_df['posy_ini'] = np.zeros_like(sph_msh.points[:,0])+2.
ini_df['posz_ini'] = np.zeros_like(sph_msh.points[:,0])+0.5

ini_df['kx_ini'] = sph_msh.points[:,0]/2.
ini_df['ky_ini'] = sph_msh.points[:,1]/2.
ini_df['kz_ini'] = sph_msh.points[:,2]/2.

ini_df['Ux_ini'] = sph_msh['Vswx']
ini_df['Uy_ini'] = sph_msh['Vswy']
ini_df['Uz_ini'] = sph_msh['Vswz']

print(ini_df)
ini_df.to_csv('./RT_src.csv')


