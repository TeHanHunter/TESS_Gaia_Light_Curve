import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from lightkurve.correctors import RegressionCorrector, DesignMatrix
from lightkurve.correctors import PLDCorrector
import warnings
from astropy.table import Table
from astropy.io import ascii
from tqdm import trange


def get_crowdsap(star, sector):
    lc = lk.search_lightcurve(star, mission='TESS', exptime=120, sector=sector)[0].download()
    crowdsap = lc.meta['CROWDSAP']
    return (1 - crowdsap) / crowdsap

def get_crowdsap_tpf(star, sector):
    tpf = lk.search_targetpixelfile(star, sector=sector).download(quality_bitmask='hard')
    crowdsap = tpf.hdu[1].header['CROWDSAP']
    return (1 - crowdsap) / crowdsap

if __name__ == '__main__':
    contamrt = ascii.read('/Users/tehan/Documents/TGLC/contamination_ratio.dat')
    folder = '/Users/tehan/Documents/TGLC/'
    contamrt_min = 0.

    # difference_tglc = ascii.read(f'{folder}deviation_TGLC.dat')
    # # d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    # # d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # # print(len(d_qlp))
    # d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.1)]
    # d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # # print(len(d_tglc))
    # # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    # difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    # ground = [156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
    #           445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 289661991, 464300749, 151483286, 335590096,
    #           17865622, 193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
    #           395393265, 310002617, 220076110, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137, 243641947,
    #           419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
    #           240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
    #           239816546, 361343239] + [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
    #                                    268301217, 455784423]
    # contamrt_ground = []
    # for i in range(len(d_tglc)):
    #     star_sector = d_tglc['Star_sector'][i]
    #     try:
    #         if contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]] > contamrt_min:
    #             if int(star_sector.split('_')[1]) in ground:
    #                 difference_tglc.add_row(d_tglc[i])
    #                 contamrt_ground.append(contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]])
    #     except IndexError:
    #         print(star_sector)

    difference_tglc = ascii.read(f'{folder}deviation_TGLC.dat')
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.1)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    no_ground = [428699140, 201248411, 172518755, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                 271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                 351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                 219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                 148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                 29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                 172370679, 116483514, 350153977, 37770169, 162802770, 212957629, 393831507, 207110080, 190496853,
                 404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                 394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                 151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 1003831, 83092282,
                 264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] + [
                    370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                    464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                    320004517, 163539739, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                    408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                    407126408, 55650590, 335630746, 55525572, 342642208, 394357918]
    contamrt_no_ground = []
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        try:
            if contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]] > contamrt_min:
                if int(star_sector.split('_')[1]) in no_ground:
                    difference_tglc.add_row(d_tglc[i])
                    contamrt_no_ground.append(contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]])
        except IndexError:
            pass
            # difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])

    contamrt_spoc = []
    for i in trange(len(difference_tglc)):
        try:
            star = difference_tglc['Star_sector'][i].split('_')[1]
            sector = int(difference_tglc['Star_sector'][i].split('_')[2])
            contam_rt = get_crowdsap(f'TIC {star}', sector)
            contamrt_spoc.append(contam_rt)
        except:
            try:
                contam_rt = get_crowdsap_tpf(f'TIC {star}', sector)
                contamrt_spoc.append(contam_rt)
            except:
                contamrt_spoc.append(100)

    difference_tglc['contamrt_spoc'] = contamrt_spoc
    difference_tglc.write(f'{folder}deviation_TGLC_spoc_crowdsap_ng.dat', format='ascii', overwrite=True)
