from TGLC.ffi import *

if __name__ == '__main__':
    sector = 17
    for i in range(16):
        cut_ffi(sector=sector, camera=1 + i // 4, ccd=1 + i % 4, path=f'/home/tehan/data/sector{sector}/')
    # input_files = glob(f'/mnt/d/TESS_Sector_17/*2-3-????-?_ffic.fits')
    # with fits.open(input_files[0], mode='denywrite') as hdul:
    #     wcs = WCS(hdul[1].header)
    #     print(wcs.all_world2pix(np.array([359.79338510868735, 39.314255560169116]).reshape((1, 2)), 0))
    # with open('/mnt/d/TESS_Sector_17/2-3/source_0_0.pkl', 'rb') as input_:
    #     source = pickle.load(input_)
    # epsf(source, factor=2, target='17', sector=17, local_directory='/mnt/d/TESS_Sector_17/lc/')
    # with open('/mnt/d/TESS_Sector_17/3-2/source_4_5.pkl', 'rb') as input_:
    #     source = pickle.load(input_)
    #     epsf(source, factor=2, target='17', sector=17,
    #          local_directory='/mnt/d/TESS_Sector_17/lc/')
    # pool = Pool(os.cpu_count())
    # for _ in tqdm(pool.imap_unordered(cnn_prediction, pos), total=len(pos)):
    #     pass
    # pool.close()

    # with open('/mnt/c/users/tehan/desktop/sector24/2-2/source_21_2.pkl', 'rb') as input_:
    #     source_1 = pickle.load(input_)
    #     plt.imshow(source.flux[0])
    #     plt.scatter(source_1.gaia['sector_0_x'][0:10], source_1.gaia['sector_0_y'][0:10],
    #                 s=10 * source.gaia['tess_flux_ratio'][0:10], c='r')
    #     plt.show()
