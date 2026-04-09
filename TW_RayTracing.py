from ray_tracer import *


if __name__ == '__main__':
    Rs2km = 696300

    # ++++++++++++++++++++++++ User Define +++++++++++++++++++++++++++++++++++++
    df = pd.read_csv('TW_wave_vector',sep='\t')
    for i in range(4):
        pos_ini = np.array([df['station_x'][i]/Rs2km,df['station_y'][i]/Rs2km,df['station_z'][i]/Rs2km])  # Rs
        k_ini = np.array([df['wv_x'][i],df['wv_y'][i],df['wv_z'][i]])*1e-3 # 1/m
        xh = 0.01  # Rs
        kh = 1e-10  # 1/m
        mode = 'Slow'
        direction = 'Forward'
        error = 1.e-3
        dt = 60 * 1.  # s
        Nt = 4000  # steps

        for mode in ['Slow','Fast','Alfven']:
            for direction in ['Forward', 'Backward']:
                result_df = ray_tracer(pos_ini, k_ini,visualize=False,mode=mode,direction=direction,xh=xh,kh=kh,error=error,dt=dt,Nt=Nt,
                               result_tag='Case'+str(i+1),export_result_path='export/TW_coro_results/',export_fig_path='export/TW_coro_figures/')
        plt.figure()
        result_df.plot(subplots=True)
        plt.show()
