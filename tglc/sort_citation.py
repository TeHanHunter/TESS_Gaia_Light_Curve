import re
import json
from collections import OrderedDict
import numpy as np
import pickle
from astropy.io import ascii
from uncertainties import ufloat
# ========== CONFIGURATION ==========
input_table = 'new_input_table.tex'
output_entries = "entries.tex"
output_refs = "references.tex"

tic_list_1 = [156648452, 454248975, 445751830, 86263325, 194795551, 193641523, 394050135, 409794137, 243641947,
              470171739, 306648160, 460984940, 33595516, 458419328, 147977348, 16005254, 280655495, 375506058,
              269333648, 279947414, 154293917, 100389539, 401125028, 285048486, 428787891, 452006073, 198008005,
              464300749, 335590096, 447061717, 44792534, 23769326, 395393265, 241249530, 20182780, 70524163, 361343239,
              376524552, 250111245, 68007716, 39414571, 336128819, 417646390, 69356857, 66561343, 263179590, 154220877,
              139375960, 239816546, 119585136, 271893367, 88992642, 394722182, 388076422, 395171208, 439366538, 7548817,
              446549906, 178162579, 151483286, 58825110, 259172249, 396562848, 268301217, 165464482, 124379043,
              150098860, 179317684, 95660472, 310002617, 178709444, 90850770, 419411415, 281408474, 8599009, 144700903,
              240823272, 166184428, 258920431]
tic_list_2 = [428699140, 157698565, 140691463, 271478281, 332558858, 126606859, 387690507, 460205581, 370133522,
              219344917, 408636441, 103633434, 441765914, 147950620, 232982558, 342642208, 257060897, 289580577,
              206541859, 417931300, 166648874, 261867566, 10837041, 296739893, 321669174, 183120439, 383390264,
              293954617, 280206394, 154872375, 329148988, 391949880, 198356533, 328934463, 232612416, 141488193,
              306955329, 158588995, 410214986, 441738827, 322807371, 220479565, 393831507, 190496853, 394137592,
              394561119, 183985250, 142937186, 318753380, 34077285, 71431780, 232540264, 237913194, 404518509,
              348835438, 303432813, 219857012, 151825527, 268532343, 358070912, 437856897, 229742722, 305424003,
              362249359, 293607057, 37749396, 437011608, 286916251, 119584412, 287080092, 394357918, 124573851,
              237922465, 29857954, 219850915, 21535395, 376637093, 365102760, 97766057, 287145649, 1167538, 144193715,
              219016883, 230001847, 219854519, 260708537, 246965431, 201177276, 39699648, 289164482, 333657795,
              55525572, 199376584, 52005579, 420112589, 70513361, 229510866, 154089169, 445903569, 256722647, 179034327,
              468574941, 21832928, 12421862, 49428710, 393818343, 349095149, 352413427, 172370679, 350153977, 118327550,
              209464063, 232976128, 307958020, 207141131, 232967440, 348755728, 267263253, 300038935, 257527578,
              353475866, 159873822, 142394656, 368287008, 422756130, 62483237, 224297258, 160390955, 466206508,
              305739565, 120826158, 158025009, 91987762, 218795833, 83092282, 76923707, 271971130, 429358906, 394918211,
              139285832, 470381900, 63898957, 142387023, 417931607, 317548889, 99869022, 290348383, 372172128,
              328081248, 262530407, 219854185, 35009898, 367858035, 233087860, 198485881, 54002556, 231702397,
              277634430, 407591297, 271169413, 407126408, 445805961, 268334473, 22233480, 380887434, 70899085,
              287256467, 148673433, 335630746, 282485660, 200723869, 332534326, 382602147, 320004517, 149845414,
              198241702, 9348006, 237232044, 29960110, 392476080, 37770169, 116483514, 212957629, 360156606, 207110080,
              298663873, 58542531, 404505029, 49254857, 439456714, 464646604, 192790476, 441462736, 169249234,
              158002130, 153065527, 456862677, 389900760, 4672985, 339672028, 176956893, 142381532, 219041246,
              266980320, 260004324, 288735205, 158241252, 237104103, 73540072, 159418353, 159781361, 204650483,
              321857016, 343628284, 350618622, 436873727]
tic_list_3 = [27916356, 137683938, 405717754, 271354351, 123495874, 268159158, 123233041, 122596693,
              27774415, 164786087, 159098316, 268924036, 158388163, 138430864, 164892194, 272366748,
              273874849, 137899948, 158170594, 399794420, 63452790, 378085713, 122441491, 137903329,
              299220166, 271042217, 269269546, 171974763, 27318774, 159725995, 27990610]
folder='/Users/tehan/Documents/TGLC/'
difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025.dat')
tics_fit = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc['Star_sector']]
# Update for tic_list_1 loop
rors_1 = []
ror_errs_1 = []
fp_1 = []
tic_1 = []
for i, tic in enumerate(tic_list_1):
    ror = []
    ror_err = []
    weights = []
    ror_lit = 0.
    for j in range(len(difference_tglc)):
        if tics_fit[j] == tic and difference_tglc['rhat'][j] == 1.0:
            value = float(difference_tglc['value'][j])
            err = float((difference_tglc['err1'][j] - difference_tglc['err2'][j]) / 2)
            if err > 0:
                ror.append(value)
                ror_err.append(err)
                weights.append(1 / err**2)
                ror_lit = difference_tglc['pl_ratror'][j]

    if weights:
        ror_avg_val = np.average(ror, weights=weights)
        ror_avg_err_val = np.sqrt(1 / np.sum(weights))
        ror_avg_u = ufloat(ror_avg_val, ror_avg_err_val)
    else:
        ror_avg_u = ufloat(np.nan, np.nan)
    rors_1.append(ror_avg_u.nominal_value)
    ror_errs_1.append(ror_avg_u.std_dev)
    fp = 1 - (ror_lit / ror_avg_u) if not (np.isnan(ror_avg_u.nominal_value) or ror_avg_u.nominal_value == 0) else ufloat(np.nan, np.nan)
    fp_1.append(fp)
    tic_1.append(tic)

# Similar update for tic_list_2 loop
rors_2 = []
ror_errs_2 = []
fp_2 = []
tic_2 = []
for i, tic in enumerate(tic_list_2):
    ror = []
    ror_err = []
    weights = []
    ror_lit = 0.
    for j in range(len(difference_tglc)):
        if tics_fit[j] == tic and difference_tglc['rhat'][j] == 1.0:
            value = float(difference_tglc['value'][j])
            err = float((difference_tglc['err1'][j] - difference_tglc['err2'][j]) / 2)
            if err > 0:
                ror.append(value)
                ror_err.append(err)
                weights.append(1 / err**2)
                ror_lit = difference_tglc['pl_ratror'][j]

    if weights:
        ror_avg_val = np.average(ror, weights=weights)
        ror_avg_err_val = np.sqrt(1 / np.sum(weights))
        ror_avg_u = ufloat(ror_avg_val, ror_avg_err_val)
    else:
        ror_avg_u = ufloat(np.nan, np.nan)
    rors_2.append(ror_avg_u.nominal_value)
    ror_errs_2.append(ror_avg_u.std_dev)
    fp = 1 - (ror_lit / ror_avg_u) if not (np.isnan(ror_avg_u.nominal_value) or ror_avg_u.nominal_value == 0) else ufloat(np.nan, np.nan)
    fp_2.append(fp)
    tic_2.append(tic)

print(len(tic_2))

difference_tglc_kepler = ascii.read(f'{folder}deviation_TGLC_2024_kepler.dat')
tics_fit_kepler = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_kepler['Star_sector']]

# 2. Process tic_list_3 data
rors_3 = []
ror_errs_3 = []
fp_3 = []
tic_3 = []
for i, tic in enumerate(tic_list_3):
    ror = []
    ror_err = []
    weights = []
    ror_lit = 0.
    for j in range(len(difference_tglc_kepler)):
        if tics_fit_kepler[j] == tic and difference_tglc_kepler['rhat'][j] == 1.0:
            value = float(difference_tglc_kepler['value'][j])
            err = float((difference_tglc_kepler['err1'][j] - difference_tglc_kepler['err2'][j]) / 2)
            if err > 0:
                ror.append(value)
                ror_err.append(err)
                weights.append(1 / err ** 2)
                ror_lit = difference_tglc_kepler['pl_ratror'][j]

    if weights:
        ror_avg_val = np.average(ror, weights=weights)
        ror_avg_err_val = np.sqrt(1 / np.sum(weights))
        ror_avg_u = ufloat(ror_avg_val, ror_avg_err_val)
    else:
        ror_avg_u = ufloat(np.nan, np.nan)

    rors_3.append(ror_avg_u.nominal_value)
    ror_errs_3.append(ror_avg_u.std_dev)
    fp = 1 - (ror_lit / ror_avg_u) if not (np.isnan(ror_avg_u.nominal_value) or ror_avg_u.nominal_value == 0) else ufloat(np.nan, np.nan)
    fp_3.append(fp)
    tic_3.append(tic)

# print(tic_g)
# ===================================
def parse_table(input_table, convert):
    with open(input_table, 'r') as f:
        content = f.read()

    pattern = r'(\d+) & (.*?) & (\\cite\{TIC_\d+\})'
    matches = re.findall(pattern, content)
    entries = []
    citation_order = OrderedDict()
    for tic_str, pipeline, citation in matches:
        tic = int(tic_str)
        key = re.search(r'{TIC_(\d+)}', citation).group(0)[1:-1]
        key = convert[key]
        if tic not in citation_order:
            citation_order[key] = tic
        entries.append({'tic': tic, 'pipeline': pipeline, 'citation': citation, 'key': key})
    return entries, citation_order

def split_and_sort_tables(entries, tic_list_1, tic_list_2, tic_list3):
    table1_entries = [e for e in entries if e['tic'] in tic_list_1]
    table2_entries = [e for e in entries if e['tic'] in tic_list_2]
    table3_entries = [e for e in entries if e['tic'] in tic_list_3]
    table1_sorted = sorted(table1_entries, key=lambda x: x['tic'])
    table2_sorted = sorted(table2_entries, key=lambda x: x['tic'])
    table3_sorted = sorted(table3_entries, key=lambda x: x['tic'])
    return table1_sorted, table2_sorted, table3_sorted

def reorder_citations(entries_file, refs_file, citation_order):
    with open(entries_file, 'r') as f:
        entries = f.readlines()
    with open(refs_file, 'r') as f:
        refs = f.readlines()

    entry_dict = {}
    seen = set()
    for line in entries:
        match = re.search(r'\\noindent\s+\d+\.\s+([\w\s-]+),\s+[A-Z].*?\(\d{4}\)', line)
        if match:
            last_name = match.group(1)
            year = re.search(r'\((\d{4})\)', line).group(1)
            key = f"{last_name}{year}"
            if key in seen:
                key = key + 'b'
            entry_dict[key] = line
            seen.add(key)

    ref_dict = {}
    for line in refs:
        key_match = re.search(r'\\reference\{(.*?)\}', line)
        if key_match:
            key = key_match.group(1)
            ref_dict[key] = line

    sorted_entries = []
    sorted_refs = []
    seen = set()
    start = 30
    for key in citation_order:
        if key in entry_dict and key not in seen:
            match = re.search(r'\\noindent\s+(\d+)\.\s+[\w\s-]+,\s+[A-Z].*?\(\d{4}\)', entry_dict[key])
            sorted_entries.append(entry_dict[key].replace(f'{match.group(1)}.', f'{start}.'))
            seen.add(key)
            sorted_refs.append(ref_dict[key].replace(f'^{match.group(1)}', '^{' + f'{start}' + '}'))
            sorted_refs.append(ref_dict[f"{key}-r"].replace('{' + f'{match.group(1)}', '{' + f'{start}'))
            start +=1

    with open('entries_sorted.tex', 'w') as f:
        f.writelines(sorted_entries)
    with open('references_sorted.tex', 'w') as f:
        f.writelines(sorted_refs)

def generate_latex_tables(table1, table2, table3, convert):
    # Table 1 (Gaia radii)
    table1_tex = [
        r'\begin{longtable}{ccccc}',
        r'\caption{TICs with Gaia Radii} \label{tab:table1} \\',
        r'\hline',
        r'TIC & Photometry & Literature & $p_{\text{TGLC}}$ & $f_p$ \\',
        r'\hline',
        r'\endfirsthead',
        r'\hline',
        r'TIC & Photometry & Literature & $p_{\text{TGLC}}$ & $f_p$ \\',
        r'\hline',
        r'\endhead',
        r'\hline\endfoot'
    ]
    for entry in table1:
        tic = entry['tic']
        try:
            idx = tic_1.index(tic)
        except ValueError:
            print(f"TIC {tic} not found in tic_1. Skipping.")
            continue
        val = ufloat(rors_1[idx], ror_errs_1[idx])
        fp_val = fp_1[idx] * 100  # Convert to percentage
        key = entry['citation'].split('{')[1].split('}')[0]
        citation = '\cite{' + f'{convert[key]}' + '}'
        if not np.isnan(fp_val.nominal_value):
            table1_tex.append(f"{tic} & {entry['pipeline']} & {citation} & ${val:.1uL}$ & ${fp_val.nominal_value:.2f}\\%$ \\\\")

    # Table 2 (Non-Gaia radii)
    table2_tex = [
        r'\begin{longtable}{ccccc}',
        r'\caption{TICs with Non-Gaia Radii} \label{tab:table2} \\',
        r'\hline',
        r'TIC & Photometry & Literature & $p_{\text{TGLC}}$ & $f_p$ \\',
        r'\hline',
        r'\endfirsthead',
        r'\hline',
        r'TIC & Photometry & Literature & $p_{\text{TGLC}}$ & $f_p$ \\',
        r'\hline',
        r'\endhead',
        r'\hline\endfoot'
    ]

    for entry in table2:
        tic = entry['tic']
        try:
            idx = tic_2.index(tic)
        except ValueError:
            print(f"TIC {tic} not found in tic_2. Skipping.")
            continue
        val = ufloat(rors_2[idx], ror_errs_2[idx])
        fp_val = fp_2[idx] * 100  # Convert to percentage
        key = entry['citation'].split('{')[1].split('}')[0]
        citation = '\cite{' + f'{convert[key]}' + '}'
        if not np.isnan(fp_val.nominal_value):
            table2_tex.append(f"{tic} & {entry['pipeline']} & {citation} & ${val:.1uL}$ & ${fp_val.nominal_value:.2f}\\%$ \\\\")

    # Table 3 (Kepler)
    table3_tex = [
        r'\begin{longtable}{cccc}',
        r'\caption{Additional TICs (Kepler)} \label{tab:table3} \\',
        r'\hline',
        r'TIC & Literature & $p_{\text{TGLC}}$ & $f_p$ \\',
        r'\hline',
        r'\endfirsthead',
        r'\hline',
        r'TIC & Literature & $p_{\text{TGLC}}$ & $f_p$ \\',
        r'\hline',
        r'\endhead',
        r'\hline\endfoot'
    ]

    for entry in table3:
        tic = entry['tic']
        try:
            idx = tic_3.index(tic)
        except ValueError:
            print(f"TIC {tic} not found in tic_3. Skipping.")
            continue

        val = ufloat(rors_3[idx], ror_errs_3[idx])
        fp_val = fp_3[idx] * 100  # Convert to percentage
        key = entry['citation'].split('{')[1].split('}')[0]
        citation = '\cite{' + f'{convert[key]}' + '}'

        table3_tex.append(
            f"{tic} & {citation} & "
            f"${val:.1uL}$ & ${fp_val.nominal_value:.1f}\\%$ \\\\"
        )

    return '\n'.join(table1_tex), '\n'.join(table2_tex), '\n'.join(table3_tex)

if __name__ == "__main__":
    with open("dictionary.json", "r") as file:
        convert = json.load(file)
    entries, citation_order = parse_table(input_table, convert)
    table1, table2, table3 = split_and_sort_tables(entries, tic_list_1, tic_list_2, tic_list_3)
    reorder_citations(output_entries, output_refs, citation_order)
    table1_tex, table2_tex, table3_tex = generate_latex_tables(table1, table2, table3, convert)
    with open("table1.tex", 'w') as f:
        f.write(table1_tex)
    with open("table2.tex", 'w') as f:
        f.write(table2_tex)
    with open("table3.tex", 'w') as f:
        f.write(table3_tex)