from spacepy.pybats import IdlFile
import numpy as np

Rs = 696300  #km

# %%
data_path = '/Users/ephe/THL8/Test_SC230315_2304/output_SC_230315/SC/'
file_type = 'box_mhd_4_'
n_iter = 6000
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

Vax = np.array(data_box['Bx']) * 1e-5 / np.sqrt(4. * np.pi * data_box['Rho']) / Rs  # Rs/s
Vay = np.array(data_box['By']) * 1e-5 / np.sqrt(4. * np.pi * data_box['Rho']) / Rs
Vaz = np.array(data_box['Bz']) * 1e-5 / np.sqrt(4. * np.pi * data_box['Rho']) / Rs

Vax[np.isnan(Vax)]=0.
Vay[np.isnan(Vay)]=0.
Vaz[np.isnan(Vaz)]=0.

Vswx = np.array(data_box['Ux']) / Rs  # Rs/s
Vswy = np.array(data_box['Uy']) / Rs
Vswz = np.array(data_box['Uz']) / Rs
Vswx[np.isnan(Vswx)]=0.
Vswy[np.isnan(Vswy)]=0.
Vswz[np.isnan(Vswz)]=0.
gamma = 5. / 3.
Cs = np.sqrt(gamma * data_box['P'] / data_box['Rho']) * 1e-5 / Rs  # Rs/s
Cs[np.isnan(Cs)]=0.
gridx = np.array(data_box['x'])  # Rs
gridy = np.array(data_box['y'])
gridz = np.array(data_box['z'])


# %%
def calc_omega(Va_vec, Vsw_vec, Cs, k_vec, mode='Alfven'):
    if mode == 'Alfven':
        omega = np.dot(k_vec, (Vsw_vec + Va_vec))
    elif mode == 'Fast':
        Va_norm = np.linalg.norm(Va_vec)
        # Vsw_norm = np.linalg.norm(Vsw_vec)
        k_norm = np.linalg.norm(k_vec)
        k_dot_Va = np.dot(k_vec, Va_vec)
        k_dot_Vsw = np.dot(k_vec, Vsw_vec)
        omega = np.sqrt((Cs ** 2 + Va_norm ** 2) * k_norm ** 2 / 2.
                        + np.sqrt(
            (Cs ** 2 + Va_norm ** 2) ** 2 * k_norm ** 4 - 4 * k_norm ** 2 * Cs ** 2 * k_dot_Va ** 2) / 2.
                        + k_dot_Vsw ** 2)
    elif mode == 'Slow':
        Va_norm = np.linalg.norm(Va_vec)
        # Vsw_norm = np.linalg.norm(Vsw_vec)
        k_norm = np.linalg.norm(k_vec)
        k_dot_Va = np.dot(k_vec, Va_vec)
        k_dot_Vsw = np.dot(k_vec, Vsw_vec)
        omega = np.sqrt((Cs ** 2 + Va_norm ** 2) * k_norm ** 2 / 2.
                        - np.sqrt(
            (Cs ** 2 + Va_norm ** 2) ** 2 * k_norm ** 4 - 4 * k_norm ** 2 * Cs ** 2 * k_dot_Va ** 2) / 2.
                        + k_dot_Vsw ** 2)
    else:
        return None

    return omega


from scipy.interpolate import interpn

def get_Va(pos):
    return np.array([interpn((gridx,gridy,gridz),Vax,pos),
                     interpn((gridx,gridy,gridz),Vay,pos),
                     interpn((gridx,gridy,gridz),Vaz,pos)])


def get_Vsw(pos):
    return np.array([interpn((gridx,gridy,gridz),Vswx,pos),
                     interpn((gridx,gridy,gridz),Vswy,pos),
                     interpn((gridx,gridy,gridz),Vswz,pos)])

def get_Cs(pos):
    return np.array(interpn((gridx,gridy,gridz),Cs,pos))


def get_derivative(pos, k, xh, kh, mode='Alfven'):
    pos = np.array(pos)
    k = np.array(k)
    omega_x0k0 = calc_omega(get_Va(pos),
                            get_Vsw(pos),
                            get_Cs(pos),
                            k,mode=mode,)
    print('omega_x0k0: ', omega_x0k0)

    domega_dx_k0_x = (calc_omega(get_Va(pos + [xh,0,0]),
                                 get_Vsw(pos + [xh, 0, 0]),
                                 get_Cs(pos + [xh, 0, 0]),
                                 k,mode=mode,)
                      - calc_omega(get_Va(pos - [xh,0,0]),
                                   get_Vsw(pos - [xh, 0, 0]),
                                   get_Cs(pos - [xh, 0, 0]),
                                   k,mode=mode,)) / (2 * xh)

    domega_dx_k0_y = (calc_omega(get_Va(pos + [0,xh,0]),
                                 get_Vsw(pos + [0, xh, 0]),
                                 get_Cs(pos + [0, xh, 0]),
                                 k,mode=mode,)
                      - calc_omega(get_Va(pos - [0,xh,0]),
                                   get_Vsw(pos - [0, xh, 0]),
                                   get_Cs(pos - [0, xh, 0]),
                                   k,mode=mode,)) / (2 * xh)

    domega_dx_k0_z = (calc_omega(get_Va(pos + [0,0,xh]),
                                 get_Vsw(pos + [0,0,xh]),
                                 get_Cs(pos + [0,0,xh]),
                                 k,mode=mode,)
                      - calc_omega(get_Va(pos - [0,0,xh]),
                                   get_Vsw(pos - [0,0,xh]),
                                   get_Cs(pos - [0,0,xh]),
                                   k,mode=mode,)) / (2 * xh)

    domega_dk_x0_kx = (calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                  k + [kh, 0, 0],mode=mode)
                       - calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                    k - [kh, 0, 0],mode=mode)) / (2 * kh)
    domega_dk_x0_ky = (calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                  k + [0, kh, 0],mode=mode)
                       - calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                    k - [0, kh, 0],mode=mode)) / (2 * kh)
    domega_dk_x0_kz = (calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                  k + [0, 0, kh],mode=mode)
                       - calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                    k - [0, 0, kh],mode=mode)) / (2 * kh)
    print('domega_dx: ', [domega_dx_k0_x, domega_dx_k0_y, domega_dx_k0_z])
    print('domega_dk: ', [domega_dk_x0_kx, domega_dk_x0_ky, domega_dk_x0_kz])

    return np.array([domega_dx_k0_x, domega_dx_k0_y, domega_dx_k0_z]).squeeze(), \
        np.array([domega_dk_x0_kx, domega_dk_x0_ky, domega_dk_x0_kz]).squeeze(), \
        omega_x0k0

def step_on(pos_tmp, k_tmp, xh, kh, dt,mode='Alfven',direction='Forward'):
    dwdx, dwdk, omega= get_derivative(pos_tmp, k_tmp, xh, kh,mode=mode)
    dx = np.array(dwdk) * dt
    dk = -np.array(dwdx) * dt
    print('dx: ', dx)
    print('dk: ', dk)
    if direction == 'Forward':
        pos_new = pos_tmp + dx
        k_new = k_tmp + dk
    elif direction == 'Backward':
        pos_new = pos_tmp - dx
        k_new = k_tmp - dk
    else:
        return None
    return pos_new, k_new, omega

# %%
pos_ini = np.array([1.,1.,0.])
k_ini = np.array([1., 1., 1.])
xh = 0.5
kh = 0.5

pos_tmp = pos_ini
k_tmp = k_ini
pos_list = []
k_list = []
omega_list = []
pos_list.append(pos_tmp)
k_list.append(k_tmp)
dt = 60*10  # s
for nt in range(200):
    print('-----------------Nt = ' + str(nt) + '------------------')
    pos_new, k_new, omega = step_on(pos_tmp, k_tmp, xh, kh, dt,mode='Fast',direction='Forward')
    print('pos_tmp: ', pos_tmp, 'k_tmp: ', k_tmp)
    pos_tmp = pos_new
    k_tmp = k_new
    pos_list.append(pos_tmp)
    k_list.append(k_tmp)
    omega_list.append(omega)
    if abs(pos_tmp[0])>24. or abs(pos_tmp[1])>24. or abs(pos_tmp[2])>24.:
        print('Boundary Reached.')
        break

pos_list = np.array(pos_list)
k_list = np.array(k_list)
omega_list = np.array(omega_list)

import pandas as pd
result = {'pos_x_Rs':pos_list[:,0],'pos_y_Rs':pos_list[:,1],'pos_z_Rs':pos_list[:,2],
          'k_x':k_list[:,0],'k_y':k_list[:,1],'k_z':k_list[:,2],
          'omega':omega_list}
df = pd.DataFrame(result)
df.to_csv('RESULT/test.csv')

# %%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(omega_list)
plt.show()

# %%
import pyvista as pv
pos_line = pv.lines_from_points(pos_list)

dimensions = data_box['grid']
spacing = (abs(gridx[1] - gridx[0]), abs(gridy[1] - gridy[0]), abs(gridz[1] - gridz[0]))
origin = (gridx[0], gridy[0], gridz[0])
box_grid = pv.UniformGrid(dimensions=(dimensions[0], dimensions[1], dimensions[2]), spacing=spacing,
                                   origin=origin)
lgRho = np.log10(data_box['Rho'])
lgRho[np.isinf(lgRho)] = np.nan
box_grid.point_data['lg(Rho)'] = lgRho.ravel('F')
# %%
p = pv.Plotter()
p.add_mesh(pos_line.tube(radius=0.1))
p.add_arrows(pos_list,k_list,mag=0.5)
p.add_mesh_slice_orthogonal(box_grid,clim=[-22,-15],cmap='jet')
p.show_grid()
p.show_axes()
p.show()

