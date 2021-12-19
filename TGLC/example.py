from TGLC.ffi import *
from TGLC.ePSF import *
from TGLC.local_background import *
from TGLC.target_lightcurve import *

target = 'NGC_7654'  # Target identifier or coordinates
local_directory = f'/mnt/c/users/tehan/desktop/{target}/'
# local_directory = os.path.join(os.getcwd(), f'{target}/')
if not os.path.exists(local_directory):
    os.makedirs(local_directory)
size = 90  # int
source = ffi(target=target, size=size, local_directory=local_directory)
print(source.sector_table)
# source.select_sector(sector=18)
flatten_lc = epsf(factor=4, target=target, sector=source.sector, local_directory=local_directory)