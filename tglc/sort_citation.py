import re
import json
from collections import OrderedDict
import numpy as np
import pickle
from astropy.io import ascii
from uncertainties import ufloat
import csv


# ========== CONFIGURATION ==========
input_table = 'new_input_table.tex'
output_entries = "entries.tex"
output_refs = "references.tex"
ground_old = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 155867025, 198008005, 178162579, 464300749, 151483286,
               335590096,
               193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
               395393265, 310002617, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137,
               243641947,
               419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
               240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
               239816546, 361343239] +
              [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
               268301217, 455784423] +
              [452006073, 306648160, 165464482, 23769326, 470171739,
               166184428, 259172249, 69356857, 58825110, 154220877,
               119585136, 388076422, 178709444, 241249530, 446549906,
               269333648, 401125028, 439366538])
ground = [16005254, 20182780, 33595516, 44792534, 88992642,
          119585136, 144700903, 150098860, 154220877, 179317684, 193641523,
          243641947, 250111245, 259172249, 271893367, 285048486, 335590096,
          376524552, 388076422, 394050135, 396562848, 409794137,
          419411415, 428787891, 445751830, 447061717, 458419328, 460984940,
          464300749]

ground_diff = list(set(ground_old) - set(ground))

ground_diff = list(set(ground_old) - set(ground))
tic_list_1 = ground
tic_list_2 = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
              271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
              351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
              219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
              148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
              29960110, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
              172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
              404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 169249234, 159873822,
              394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
              151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
              264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
             [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
              464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
              320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
              408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
              407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
             [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
              352413427, 169765334, 39699648, 305739565, 391903064, 237913194, 160390955, 257060897, 365102760,
              393818343, 153065527, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
              120826158, 99869022, 456862677, 219850915, 380887434, 232612416, 271169413, 232976128, 49254857,
              198241702, 282485660, 224297258, 303432813, 391949880, 437011608, 198356533, 232982558, 237232044,
              343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 58542531, 102734241, 268334473,
              159418353, 18318288, 219857012, 35009898, 287080092, 124573851, 289580577, 367858035, 277634430,
              9348006, 219344917, 21535395, 34077285, 286916251, 322807371, 142381532, 142387023, 46432937,
              348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
              166648874, 368287008, 389900760, 159781361, 21832928, 8348911, 289164482, 158241252, 467651916,
              201177276, 307958020, 382602147, 317548889, 268532343, 407591297, 1167538, 328081248, 328934463,
              429358906, 37749396, 305424003, 63898957]
              + [209459275,130924120,419523962,163539739] # temp
              + ground_diff)
tic_list_3 = [27916356, 137683938, 405717754, 271354351, 123495874, 268159158, 123233041, 122596693,
              27774415, 164786087, 159098316, 268924036, 158388163, 138430864, 164892194, 272366748,
              273874849, 137899948, 158170594, 399794420, 63452790, 378085713, 122441491,
              299220166, 271042217, 269269546, 171974763, 27318774, 159725995, 27990610]
tic_list_4 = [428699140, 156648452, 157698565, 140691463, 271478281, 332558858, 126606859, 387690507, 460205581, 454248975, 24358417, 370133522, 219344917, 408636441, 103633434, 441765914, 147950620, 232982558, 194795551, 342642208, 257060897, 289580577, 206541859, 261867566, 10837041, 296739893, 321669174, 183120439, 154872375, 293954617, 198356533, 332534326, 329148988, 280206394, 328934463, 232612416, 306955329, 158588995, 410214986, 441738827, 322807371, 220479565, 209459275, 393831507, 190496853, 130924120, 394137592, 470171739, 394561119, 306648160, 183985250, 142937186, 318753380, 232540264, 237913194, 404518509, 303432813, 219857012, 151825527, 268532343, 358070912, 437856897, 229742722, 147977348, 280655495, 375506058, 362249359, 269333648, 293607057, 279947414, 286916251, 119584412, 154293917, 394357918, 124573851, 287080092, 237922465, 29857954, 100389539, 219850915, 376637093, 401125028, 365102760, 97766057, 287145649, 1167538, 230001847, 246965431, 452006073, 219854519, 201177276, 289164482, 333657795, 198008005, 199376584, 420112589, 154089169, 229510866, 445903569, 97568467, 256722647, 179034327, 468574941, 12421862, 393818343, 349095149, 23769326, 124029677, 172370679, 350153977, 241249530, 118327550, 232976128, 70524163, 307958020, 361343239, 207141131, 232967440, 348755728, 267263253, 353475866, 257527578, 163539739, 159873822, 142394656, 368287008, 422756130, 68007716, 62483237, 224297258, 39414571, 466206508, 305739565, 160390955, 158025009, 91987762, 336128819, 417646390, 218795833, 83092282, 271971130, 69356857, 429358906, 76923707, 66561343, 394918211, 263179590, 139285832, 142387023, 417931607, 139375960, 317548889, 99869022, 290348383, 372172128, 328081248, 239816546, 262530407, 219854185, 367858035, 233087860, 198485881, 419523962, 54002556, 231702397, 277634430, 271169413, 394722182, 395171208, 445805961, 407126408, 268334473, 439366538, 70899085, 22233480, 7548817, 446549906, 178162579, 151483286, 58825110, 148673433, 335630746, 282485660, 200723869, 268301217, 165464482, 124379043, 382602147, 320004517, 198241702, 9348006, 237232044, 29960110, 392476080, 95660472, 310002617, 37770169, 360156606, 207110080, 298663873, 178709444, 404505029, 49254857, 439456714, 464646604, 192790476, 441462736, 169249234, 158002130, 456862677, 169765334, 4672985, 281408474, 142381532, 176956893, 219041246, 339672028, 266980320, 8599009, 260004324, 288735205, 237104103, 73540072, 240823272, 166184428, 258920431, 159418353, 159781361, 204650483, 321857016, 343628284, 350618622, 436873727]

folder='/Users/tehan/Documents/TGLC/'
difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025_updated.dat', format='csv')
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

    pattern = r'(\d+) & (.*?) & \\cite\{([^\}]+)\}'
    matches = re.findall(pattern, content)
    entries = []
    citation_order = OrderedDict()

    for tic_str, pipeline, citation_key in matches:
        tic = int(tic_str)
        if citation_key.startswith('TIC_'):
            key = convert.get(citation_key, citation_key)  # Convert TIC_... to custom key
        else:
            key = citation_key  # Keep author-year as is

        if key not in citation_order:
            citation_order[key] = tic

        entries.append({
            'tic': tic,
            'pipeline': pipeline.strip(),
            'citation': f'\\cite{{{citation_key}}}',
            'key': key
        })

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


def process_pipeline(pipeline_str):
    """Process pipeline string to split into two columns"""
    # Remove LaTeX commands and dollar signs, keep bracketed content
    cleaned = re.sub(r'\$?\\[a-zA-Z]+\$?', '', pipeline_str)  # Remove \commands
    cleaned = re.sub(r'\$\s*', '', cleaned)  # Remove dollar signs
    cleaned = re.sub(r'\{([^}]*)\}', r'\1', cleaned)  # Remove brackets but keep content

    # Split by '+' and clean
    parts = [p.strip() for p in cleaned.split('+')]
    phot1 = parts[0] if len(parts) > 0 else ''
    phot2 = parts[1] if len(parts) > 1 else ''
    return phot1, phot2


def generate_csv(table_entries, tic_list, rors, ror_errs, fp_list, convert, filename):
    """Generate a CSV file for a given table"""
    csv_rows = []
    for entry in table_entries:
        tic = entry['tic']
        try:
            idx = tic_list.index(tic)
        except ValueError:
            continue

        phot1, phot2 = process_pipeline(entry['pipeline'])
        literature = entry['key']

        p_val = rors[idx]
        p_err = ror_errs[idx]
        fp = fp_list[idx]

        # Skip entries with NaN values
        if (np.isnan(p_val) or np.isnan(p_err) or np.isnan(fp.nominal_value) or np.isnan(fp.std_dev)):
            continue

        fp_val = fp.nominal_value
        fp_err = fp.std_dev

        csv_rows.append([
            tic,
            phot1,
            phot2,
            literature,
            p_val,
            p_err,
            fp_val,
            fp_err
        ])

    # Updated headers to include f_p_error
    headers = ['TIC', 'Photometry1', 'Photometry2', 'Literature',
               'p_TGLC_value', 'p_TGLC_error', 'f_p_value', 'f_p_error']

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

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
        if key.startswith('TIC_'):
            citation = f'\\cite{{{convert.get(key, key)}}}'
        else:
            citation = f'\\cite{{{key}}}'
        if not np.isnan(fp_val.nominal_value):
            fp_str = f"{fp_val.nominal_value:.1f}\\% $\\pm$ {fp_val.std_dev:.1f}\\%"
            table1_tex.append(
                f"{tic} & {entry['pipeline']} & {citation} & ${val:.1uL}$ & {fp_str} \\\\"
            )

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
    tics_ = []
    for entry in table2:
        tic = entry['tic']
        tics_.append(tic)
        try:
            idx = tic_2.index(tic)
        except ValueError:
            print(f"TIC {tic} not found in tic_2. Skipping.")
            continue
        val = ufloat(rors_2[idx], ror_errs_2[idx])
        fp_val = fp_2[idx] * 100
        key = entry['citation'].split('{')[1].split('}')[0]
        if key.startswith('TIC_'):
            citation = f'\\cite{{{convert.get(key, key)}}}'
        else:
            citation = f'\\cite{{{key}}}'
        if not np.isnan(fp_val.nominal_value):
            fp_str = f"{fp_val.nominal_value:.1f}\\% $\\pm$ {fp_val.std_dev:.1f}\\%"
            table2_tex.append(
                f"{tic} & {entry['pipeline']} & {citation} & ${val:.1uL}$ & {fp_str} \\\\"
            )
    print(len(set(tics_) - set(tic_list_4)))
    print(set(tic_list_4) - set(tics_))

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
        fp_val = fp_3[idx] * 100
        key = entry['citation'].split('{')[1].split('}')[0]
        if key.startswith('TIC_'):
            citation = f'\\cite{{{convert.get(key, key)}}}'
        else:
            citation = f'\\cite{{{key}}}'
        fp_str = f"{fp_val.nominal_value:.1f}\\% $\\pm$ {fp_val.std_dev:.1f}\\%"
        table3_tex.append(
            f"{tic} & {citation} & ${val:.1uL}$ & {fp_str} \\\\"
        )

    generate_csv(table1, tic_1, rors_1, ror_errs_1, fp_1, convert, 'table1.csv')
    generate_csv(table2, tic_2, rors_2, ror_errs_2, fp_2, convert, 'table2.csv')
    generate_csv(table3, tic_3, rors_3, ror_errs_3, fp_3, convert, 'table3.csv')

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