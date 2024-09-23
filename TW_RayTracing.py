import numpy as np
import pandas as pd

from ray_tracer import *


if __name__ == '__main__':
    Rs2km = 696300
    # so_positions, _ = spice.spkp('SOLO, times, 'SPP_HCI, 'NONE,' 'SUNso_po = so_.T / AUax.plot([0,[1,[2,c='gray,alpha=0.8so_post,, _ = spice.spk('SOLO, obs, 'SPP_HCI, BO'NONE, 'SUNso_psos = pso_pos.T/AU))
    # df = pd.read_csv('Stations_20211004T080000.csv')
    #
    # JS_x_Rs = df['P_in_HG_x'][df.StationName == 'JS'].values[0] / Rs2km
    # JS_y_Rs = df['P_in_HG_y'][df.StationName == 'JS'].values[0] / Rs2km
    # JS_z_Rs = df['P_in_HG_z'][df.StationName == 'JS'].values[0] / Rs2km
    #
    # BD_x_Rs = df['P_in_HG_x'][df.StationName == 'BD'].values[0] / Rs2km
    # BD_y_Rs = df['P_in_HG_y'][df.StationName == 'BD'].values[0] / Rs2km
    # BD_z_Rs = df['P_in_HG_z'][df.StationName == 'BD'].values[0] / Rs2km
    #
    # YG_x_Rs = df['P_in_HG_x'][df.StationName == 'YG'].values[0] / Rs2km
    # YG_y_Rs = df['P_in_HG_y'][df.StationName == 'YG'].values[0] / Rs2km
    # YG_z_Rs = df['P_in_HG_z'][df.StationName == 'YG'].values[0] / Rs2km
    #
    # HH_x_Rs = df['P_in_HG_x'][df.StationName == 'HH'].values[0] / Rs2km
    # HH_y_Rs = df['P_in_HG_y'][df.StationName == 'HH'].values[0] / Rs2km
    # HH_z_Rs = df['P_in_HG_z'][df.StationName == 'HH'].values[0] / Rs2km
    #
    # JS_BD_mid = np.array([(JS_x_Rs + BD_x_Rs) / 2, (JS_y_Rs + BD_y_Rs) / 2, (JS_z_Rs + BD_z_Rs) / 2])
    # JS_BD_vec = np.array([JS_x_Rs - BD_x_Rs,JS_y_Rs - BD_y_Rs,JS_z_Rs - BD_z_Rs])
    #
    # # BD-JS为向外（+）
    # # HH-YG为向外（+）
    # Vsw_JS_BD_mid = get_Vsw(JS_BD_mid).squeeze()/1e3 # km/s
    # Vsw_along_JS_BD_vec = np.dot(Vsw_JS_BD_mid, JS_BD_vec)/np.linalg.norm(JS_BD_vec)
    # print(Vsw_JS_BD_mid)
    # print(Vsw_along_JS_BD_vec)
    #
    #
    # YG_HH_mid = np.array([(YG_x_Rs + HH_x_Rs) / 2, (YG_y_Rs + HH_y_Rs) / 2, (YG_z_Rs + HH_z_Rs) / 2])
    # YG_HH_vec = np.array([YG_x_Rs - HH_x_Rs, YG_y_Rs - HH_y_Rs, YG_z_Rs - HH_z_Rs])
    #
    # Vsw_YG_HH_mid = get_Vsw(YG_HH_mid).squeeze() / 1e3  # km/s
    # Vsw_along_YG_HH_vec = np.dot(Vsw_YG_HH_mid, YG_HH_vec) / np.linalg.norm(YG_HH_vec)
    # print(Vsw_YG_HH_mid)
    # print(Vsw_along_YG_HH_vec)


    # ++++++++++++++++++++++++ User Define +++++++++++++++++++++++++++++++++++++

    df = pd.read_csv('TW_wave_vector',sep='\t')
    for i in range(len(df)):
    # i = 0


        pos_ini = np.array([df['station_x'][i]/Rs2km,df['station_y'][i]/Rs2km,df['station_z'][i]/Rs2km])  # Rs
        k_ini = np.array([df['wv_x'][i],df['wv_y'][i],df['wv_z'][i]]) # 1/m
        xh = 0.01  # Rs
        kh = 1e-6  # 1/m
        mode = 'Slow'
        direction = 'Backward'
        error = 1.e-3
        dt = 60 * 10.  # s
        Nt = 500  # steps
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # for theta in np.linspace(0, np.pi * 2, 5):
        #     pos_ini = np.array([15. * np.cos(theta), 15. * np.sin(theta), 0.])
        #     k_ini = np.array([5. * np.cos(theta+np.pi/2), 5. * np.sin(theta+np.pi/2), 0.]) * 1.e-5
        ray_tracer(pos_ini, k_ini,visualize=False,mode=mode,direction=direction,xh=xh,kh=kh,error=error,dt=dt,Nt=Nt)
