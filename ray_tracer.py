from spacepy.pybats import IdlFile
import numpy as np
from scipy.interpolate import interpn

Rs_km = 696300  # km

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

Vax = np.array(data_box['Bx']) * 1e-5 / np.sqrt(4. * np.pi * data_box['Rho']) * 1e3  # m/s
Vay = np.array(data_box['By']) * 1e-5 / np.sqrt(4. * np.pi * data_box['Rho']) * 1e3
Vaz = np.array(data_box['Bz']) * 1e-5 / np.sqrt(4. * np.pi * data_box['Rho']) * 1e3

Vax[np.isnan(Vax)] = 0.
Vay[np.isnan(Vay)] = 0.
Vaz[np.isnan(Vaz)] = 0.

Vswx = np.array(data_box['Ux']) * 1e3  # m/s
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

d_i = 2.28e7 / np.sqrt(data_box['Rho'] / mp) * 1e-2  # m
omega_i = 1.32e3 * np.sqrt(data_box['Rho'] / mp)  # rad/sec



# %%
def calc_omega(Va_vec, Vsw_vec, Cs, k_vec, mode='Alfven'):
    if mode == 'Alfven':
        omega_ = np.dot(k_vec, (Vsw_vec + Va_vec))  # 1/sec

    elif mode == 'Fast':
        Va_norm = np.linalg.norm(Va_vec)
        k_norm = np.linalg.norm(k_vec)
        k_dot_Va = np.dot(k_vec, Va_vec)
        k_dot_Vsw = np.dot(k_vec, Vsw_vec)
        omega_ = np.sqrt((Cs ** 2 + Va_norm ** 2) * k_norm ** 2 / 2.
                         + np.sqrt(
            (Cs ** 2 + Va_norm ** 2) ** 2 * k_norm ** 4 - 4 * k_norm ** 2 * Cs ** 2 * k_dot_Va ** 2) / 2.
                         + k_dot_Vsw ** 2)
    elif mode == 'Slow':
        Va_norm = np.linalg.norm(Va_vec)
        k_norm = np.linalg.norm(k_vec)
        k_dot_Va = np.dot(k_vec, Va_vec)
        k_dot_Vsw = np.dot(k_vec, Vsw_vec)
        omega_ = np.sqrt((Cs ** 2 + Va_norm ** 2) * k_norm ** 2 / 2.
                         - np.sqrt(
            (Cs ** 2 + Va_norm ** 2) ** 2 * k_norm ** 4 - 4 * k_norm ** 2 * Cs ** 2 * k_dot_Va ** 2) / 2.
                         + k_dot_Vsw ** 2)
    else:
        return None

    return omega_

def get_B(pos):
    return np.array([interpn((gridx_Rs, gridy_Rs, gridz_Rs), data_box['Bx'], pos),
                     interpn((gridx_Rs, gridy_Rs, gridz_Rs), data_box['By'], pos),
                     interpn((gridx_Rs, gridy_Rs, gridz_Rs), data_box['Bz'], pos)])


def get_Va(pos):
    return np.array([interpn((gridx_Rs, gridy_Rs, gridz_Rs), Vax, pos),
                     interpn((gridx_Rs, gridy_Rs, gridz_Rs), Vay, pos),
                     interpn((gridx_Rs, gridy_Rs, gridz_Rs), Vaz, pos)])


def get_Vsw(pos):
    return np.array([interpn((gridx_Rs, gridy_Rs, gridz_Rs), Vswx, pos),
                     interpn((gridx_Rs, gridy_Rs, gridz_Rs), Vswy, pos),
                     interpn((gridx_Rs, gridy_Rs, gridz_Rs), Vswz, pos)])


def get_Cs(pos):
    return np.array(interpn((gridx_Rs, gridy_Rs, gridz_Rs), Cs, pos))


def get_derivative(pos, k, xh, kh, mode='Alfven'):
    pos = np.array(pos)  # Rs
    k = np.array(k)  # 1/m
    omega_x0k0 = calc_omega(get_Va(pos),
                            get_Vsw(pos),
                            get_Cs(pos),
                            k, mode=mode, )
    # print('omega_x0k0: ', omega_x0k0)

    domega_dx_k0_x = (calc_omega(get_Va(pos + [xh, 0, 0]),
                                 get_Vsw(pos + [xh, 0, 0]),
                                 get_Cs(pos + [xh, 0, 0]),
                                 k, mode=mode, )
                      - calc_omega(get_Va(pos - [xh, 0, 0]),
                                   get_Vsw(pos - [xh, 0, 0]),
                                   get_Cs(pos - [xh, 0, 0]),
                                   k, mode=mode, )) / (2 * xh * Rs_km * 1e3)  # 1/(m*s)

    domega_dx_k0_y = (calc_omega(get_Va(pos + [0, xh, 0]),
                                 get_Vsw(pos + [0, xh, 0]),
                                 get_Cs(pos + [0, xh, 0]),
                                 k, mode=mode, )
                      - calc_omega(get_Va(pos - [0, xh, 0]),
                                   get_Vsw(pos - [0, xh, 0]),
                                   get_Cs(pos - [0, xh, 0]),
                                   k, mode=mode, )) / (2 * xh * Rs_km * 1e3)

    domega_dx_k0_z = (calc_omega(get_Va(pos + [0, 0, xh]),
                                 get_Vsw(pos + [0, 0, xh]),
                                 get_Cs(pos + [0, 0, xh]),
                                 k, mode=mode, )
                      - calc_omega(get_Va(pos - [0, 0, xh]),
                                   get_Vsw(pos - [0, 0, xh]),
                                   get_Cs(pos - [0, 0, xh]),
                                   k, mode=mode, )) / (2 * xh * Rs_km * 1e3)

    domega_dk_x0_kx = (calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                  k + [kh, 0, 0], mode=mode)
                       - calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                    k - [kh, 0, 0], mode=mode)) / (2 * kh)  # m/s
    domega_dk_x0_ky = (calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                  k + [0, kh, 0], mode=mode)
                       - calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                    k - [0, kh, 0], mode=mode)) / (2 * kh)
    domega_dk_x0_kz = (calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                  k + [0, 0, kh], mode=mode)
                       - calc_omega(get_Va(pos), get_Vsw(pos), get_Cs(pos),
                                    k - [0, 0, kh], mode=mode)) / (2 * kh)
    # print('domega_dx [m/s]: ', [domega_dx_k0_x, domega_dx_k0_y, domega_dx_k0_z])  # m/s
    # print('domega_dk [1/ms]: ', [domega_dk_x0_kx, domega_dk_x0_ky, domega_dk_x0_kz])  # 1/(m*s)

    return np.array([domega_dx_k0_x, domega_dx_k0_y, domega_dx_k0_z]).squeeze(), \
        np.array([domega_dk_x0_kx, domega_dk_x0_ky, domega_dk_x0_kz]).squeeze(), \
        omega_x0k0


def step_on(pos_tmp, k_tmp, xh, kh, dt, mode='Alfven', direction='Forward'):
    dwdx, dwdk, omega = get_derivative(pos_tmp, k_tmp, xh, kh, mode=mode)
    dx = np.array(dwdk) * dt  # m
    dk = -np.array(dwdx) * dt  # 1/m
    print('dx [Rs]: ', dx/(Rs_km*1e3))
    print('dk [1/m]: ', dk)
    if direction == 'Forward':
        pos_new = pos_tmp + dx/(Rs_km*1e3)
        k_new = k_tmp + dk
    elif direction == 'Backward':
        pos_new = pos_tmp - dx/(Rs_km*1e3)
        k_new = k_tmp - dk
    else:
        return None
    return pos_new, k_new, omega

def step_on_RK4(pos_tmp, k_tmp, xh, kh, dt, mode='Alfven', direction='Forward'):
    if direction == 'Backward':
        dt = -dt

    dwdx_1, dwdk_1, omega_0 = get_derivative(pos_tmp, k_tmp, xh, kh, mode=mode)
    dwdx_2, dwdk_2, _ = get_derivative(pos_tmp + np.array(dwdk_1) * dt/(Rs_km*1e3)/2,
                                       k_tmp - np.array(dwdx_1) * dt/2,
                                       xh, kh, mode=mode)
    dwdx_3, dwdk_3, _ = get_derivative(pos_tmp + np.array(dwdk_2) * dt/(Rs_km*1e3)/2,
                                       k_tmp - np.array(dwdx_2) * dt/2,
                                       xh, kh, mode=mode)
    dwdx_4, dwdk_4, _ = get_derivative(pos_tmp + np.array(dwdk_3) * dt/(Rs_km*1e3),
                                       k_tmp - np.array(dwdx_3) * dt,
                                       xh, kh, mode=mode)
    dx = (dwdk_1+2*dwdk_2+2*dwdk_3+dwdk_4)*dt/6/(Rs_km*1e3)
    dk = -(dwdx_1+2*dwdx_2+2*dwdx_3+dwdx_4)*dt/6
    print('dx [Rs]: ', dx)
    print('dk [1/m]: ', dk)
    pos_new = pos_tmp + dx
    k_new = k_tmp + dk

    return pos_new, k_new, omega_0

def step_on_adapt_RK4(pos_tmp, k_tmp, xh, kh, dt, mode='Alfven', direction='Forward',error=0.001):
    if direction == 'Backward':
        dt = -dt

    dwdx_1, dwdk_1, omega_0 = get_derivative(pos_tmp, k_tmp, xh, kh, mode=mode)
    dwdx_2, dwdk_2, _ = get_derivative(pos_tmp + np.array(dwdk_1) * dt/(Rs_km*1e3)/2,
                                       k_tmp - np.array(dwdx_1) * dt/2,
                                       xh, kh, mode=mode)
    dwdx_3, dwdk_3, _ = get_derivative(pos_tmp + np.array(dwdk_2) * dt/(Rs_km*1e3)/2,
                                       k_tmp - np.array(dwdx_2) * dt/2,
                                       xh, kh, mode=mode)
    dwdx_4, dwdk_4, _ = get_derivative(pos_tmp + np.array(dwdk_3) * dt/(Rs_km*1e3),
                                       k_tmp - np.array(dwdx_3) * dt,
                                       xh, kh, mode=mode)
    dx = (dwdk_1+2*dwdk_2+2*dwdk_3+dwdk_4)*dt/6/(Rs_km*1e3)
    dk = -(dwdx_1+2*dwdx_2+2*dwdx_3+dwdx_4)*dt/6

    _,_,omega_new = get_derivative(pos_tmp+dx, k_tmp + dk, xh, kh, mode=mode)
    n_half = 0
    print(abs((omega_new-omega_0)/omega_0))
    while abs((omega_new-omega_0)/omega_0) > error:
        n_half += 1
        print('Do Half')
        dx = dx / 2.
        dk = dk / 2.
        _, _, omega_new = get_derivative(pos_tmp+dx, k_tmp+dk,xh,kh,mode=mode)

    print('dx [Rs]: ', dx)
    print('dk [1/m]: ', dk)
    print('Half Times: ', n_half)
    pos_new = pos_tmp + dx
    k_new = k_tmp + dk

    return pos_new, k_new, omega_0


# %%
# ++++++++++++++++++++++++ User Define +++++++++++++++++++++++++++++++++++++
pos_ini = np.array([15., 0., 0.])  # Rs
k_ini = np.array([5., 0., 0.])*1e-5  # 1/m
xh = 0.1  # Rs
kh = 1e-6  # 1/m
mode = 'Alfven'
direction = 'Backward'
error = 1.e-3
dt = 60*10.   # s
Nt = 300  # steps
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

B_ini = get_B(pos_ini).squeeze() # G
d_i_ini = interpn((gridx_Rs, gridy_Rs, gridz_Rs), d_i, pos_ini)  # m
omega_i_ini = interpn((gridx_Rs, gridy_Rs, gridz_Rs), omega_i, pos_ini)  # rad/sec
kdi_ini = np.linalg.norm(k_ini) * d_i_ini  #
theta_kb_ini = np.rad2deg(np.arccos(np.dot(B_ini, k_ini) / np.linalg.norm(B_ini) / np.linalg.norm(k_ini)))  # deg



print('========INITIAL VALUE==========')
print('Pos [Rs]: ', pos_ini)
print('k_vec [1/m]: ',k_ini)
print('B [Gs]: ', B_ini)
print('d_i [m]: ', d_i_ini)
print('w_i [1/s]: ', omega_i_ini)
print('k*d_i: ', kdi_ini)
print('theta_kb [deg]: ', theta_kb_ini)
print('Mode: ', mode+'+'+direction)
print('===============================')

pos_tmp = pos_ini
k_tmp = k_ini

pos_list = []
k_list = []
omega_list = []
B_list = []
di_list = []
kdi_list = []
theta_kb_list = []
omegai_list = []

pos_list.append(pos_ini)
k_list.append(k_ini)
B_list.append(B_ini)
di_list.append(d_i_ini)
kdi_list.append(kdi_ini)
theta_kb_list.append(theta_kb_ini)
omegai_list.append(omega_i_ini)
# %%

for nt in range(Nt):
    print('-----------------Nt = ' + str(nt) + '------------------')
    pos_new, k_new, omega = step_on_adapt_RK4(pos_tmp, k_tmp, xh, kh, dt, mode=mode, direction=direction,error=error)
    print('pos_tmp: ', pos_tmp, 'k_tmp: ', k_tmp)
    pos_tmp = pos_new
    k_tmp = k_new

    pos_list.append(pos_tmp)
    k_list.append(k_tmp)
    omega_list.append(omega)

    B_tmp = get_B(pos_new).squeeze()  # Gs
    di_tmp = interpn((gridx_Rs, gridy_Rs, gridz_Rs), d_i, pos_new)  # m
    wi_tmp = interpn((gridx_Rs, gridy_Rs, gridz_Rs), omega_i, pos_new)  # rad/sec
    theta_kb_tmp = np.rad2deg(np.arccos(np.dot(B_tmp,k_tmp)/np.linalg.norm(B_tmp)/np.linalg.norm(k_tmp)))

    B_list.append(B_tmp)
    di_list.append(di_tmp)
    kdi_list.append(np.linalg.norm(k_tmp)*di_tmp)
    omegai_list.append(wi_tmp)
    theta_kb_list.append(theta_kb_tmp)

    R_tmp = np.linalg.norm(pos_tmp)
    if R_tmp < 1. or R_tmp > 24.:
        print('Boundary Reached.')
        break

omega_last = calc_omega(get_Va(pos_tmp),
                        get_Vsw(pos_tmp),
                        get_Cs(pos_tmp),
                        k_tmp, mode=mode, )  # rad/sec
omega_list.append(omega_last)
pos_list = np.array(pos_list)
k_list = np.array(k_list)
omega_list = np.array(omega_list).squeeze()
B_list = np.array(B_list)
omegai_list = np.array(omegai_list).squeeze()
# %%
import pandas as pd

result = {'pos_x_Rs': pos_list[:, 0], 'pos_y_Rs': pos_list[:, 1], 'pos_z_Rs': pos_list[:, 2],
          'Bx_G': B_list[:, 0], 'By_G': B_list[:, 1], 'Bz_G': B_list[:, 2],
          'k_x_1/m': k_list[:, 0], 'k_y_1/m': k_list[:, 1], 'k_z_1/m': k_list[:, 2],
          'omega_Hz': omega_list, 'd_i_m': di_list, 'omega_i_Hz': omegai_list}
df = pd.DataFrame(result)
df.to_csv('RESULT/test.csv')

# %%
import matplotlib.pyplot as plt

pos_r_Rs = np.linalg.norm(pos_list,axis=1)
plt.figure()

plt.subplot(2,2,1)
plt.plot(pos_r_Rs, omega_list)
plt.xlabel('Radius [Rs]')
plt.ylabel('omega [Hz]')

plt.subplot(2,2,2)
plt.plot(pos_r_Rs,kdi_list)
plt.xlabel('Radius [Rs]')
plt.ylabel('k*d_i')

plt.subplot(2,2,3)
plt.plot(pos_r_Rs,theta_kb_list)
plt.xlabel('Radius [Rs]')
plt.ylabel('theta_kb [deg]')

plt.subplot(2,2,4)
plt.plot(kdi_list,omega_list/omegai_list)
plt.xlabel('k*d_i')
plt.ylabel('omega/omega_i')

plt.suptitle('pos_ini='+str(pos_ini)+'Rs k_ini='+str(k_ini)+'m^-1\n mode='+mode+' direction='+direction)
plt.show()

# %%
import pyvista as pv

pos_line = pv.lines_from_points(pos_list)

dimensions = data_box['grid']
spacing = (abs(gridx_Rs[1] - gridx_Rs[0]), abs(gridy_Rs[1] - gridy_Rs[0]), abs(gridz_Rs[1] - gridz_Rs[0]))
origin = (gridx_Rs[0], gridy_Rs[0], gridz_Rs[0])
box_grid = pv.UniformGrid(dimensions=(dimensions[0], dimensions[1], dimensions[2]), spacing=spacing,
                          origin=origin)
lgRho = np.log10(data_box['Rho'])
lgRho[np.isinf(lgRho)] = np.nan
box_grid.point_data['lg(Rho)'] = lgRho.ravel('F')
# %%
p = pv.Plotter()
p.add_mesh(pos_line.tube(radius=0.1))
p.add_arrows(pos_list, B_list, mag=1e3,color='black')
p.add_arrows(pos_list, k_list, mag=1e3,color='blue')
p.add_mesh_slice_orthogonal(box_grid, clim=[-22, -15], cmap='jet')
p.show_grid()
p.show_axes()
p.show()
