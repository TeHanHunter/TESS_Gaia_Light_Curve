import ads
import re
import math
import pkg_resources
import numpy as np
from astropy.io import ascii
token = 'EJaIdttWb1wgv0A3AQnJSbNvftJiKHBkaqCIQ0PI'
import requests

# Load data
t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.02.05_22.52.50.csv'))
tics = [int(s[4:]) for s in t['tic_id']]
html = t['disc_refname']

difference_tglc = ascii.read('/Users/tehan/Documents/TGLC/deviation_TGLC_common.dat')
used_tics = list(set([int(s.split('_')[1]) for s in difference_tglc['Star_sector']]))
idx = [np.where(np.array(tics) == used_tics[i])[0][0] for i in range(len(used_tics)) if used_tics[i] in tics]

# Prepare lists for star names and URLs
star_names = t['tic_id'][idx].tolist()
html_list = html[idx].tolist()
paper_urls = [re.search(r'href=(https://[^\s]+)', tag).group(1) for tag in html_list]

# Helper function to extract bibcode from ADS URL
def extract_bibcode(url):
    match = re.search(r'/abs/([A-Za-z0-9.&]+)/', url)
    return match.group(1) if match else None

# Function to fetch BibTeX entry and citation key for a given bibcode
def fetch_bibtex_and_citekey(bibcode, star_name):
    bibtex_entry = requests.get(f"https://api.adsabs.harvard.edu/v1/export/bibtex/{bibcode}".format(bibcode=bibcode),
                      headers={'Authorization': 'Bearer ' + token}).text
    bibtex_entry = re.sub(r'\{[^,]*,', f'{{{star_name},', bibtex_entry, count=1)
    # print(bibtex_entry)
    return bibtex_entry

# Collect data for LaTeX table, and keep track of successes and failures
entries = []
successful_entries = 0
bibcode_used = []
tic_used = []
with open("citations.bib", "w") as bibfile:
    for star_name, url in zip(star_names, paper_urls):
        bibcode = extract_bibcode(url)
        if bibcode:
            # check if paper is already cited
            if bibcode not in bibcode_used:
                star_name_ = star_name.replace(' ', '_')
                bibcode_used.append(bibcode)
                tic_used.append(star_name_)
                bibtex_entry = fetch_bibtex_and_citekey(bibcode, star_name_)
                if bibtex_entry and star_name:
                    bibfile.write(bibtex_entry + "\n\n")
            else:
                star_name_ = tic_used[np.where(np.array(bibcode_used) == bibcode)[0][0]]
            entries.append((star_name.split(' ')[-1], star_name_))
            successful_entries += 1
            print(f"Added citation for {bibcode}")
        else:
            entries.append((star_name.split(' ')[-1], ""))  # Invalid URL format
            print(f"Invalid URL format for: {url}")
print(len(bibcode_used))
print(len(tic_used))
# Calculate success rate
total_entries = len(star_names)
success_rate = (successful_entries / total_entries) * 100

# Generate three-column LaTeX table with longtable environment
num_rows = math.ceil(len(entries) / 3)
latex_table = "\\begin{longtable}{llllll}\n\\caption{Star citations} \\\\\n\\hline\n"
latex_table += "TIC & Citation & TIC & Citation & TIC & Citation \\\\\n\\hline\n\\endfirsthead\n"
latex_table += "\\hline\nTIC & Citation & TIC & Citation & TIC & Citation \\\\\n\\hline\n\\endhead\n\\hline\\endfoot\n"

# Fill the table in three columns
for i in range(num_rows):
    row = ""
    for j in range(3):
        idx = i + j * num_rows
        if idx < len(entries):
            star_name, cite_key = entries[idx]
            citation = f"\\cite{{{cite_key}}}" if cite_key else ""
            row += f"{star_name} & {citation} & "
        else:
            row += " & & "  # Empty cell if no more entries
    latex_table += row.rstrip(" & ") + " \\\\\n"  # Remove trailing "&" and add newline

latex_table += "\\hline\n\\end{longtable}"

# Save the LaTeX table to a .tex file
with open("citations_table.tex", "w") as texfile:
    texfile.write(latex_table)

# Output success rate
print(f"Success Rate: {success_rate:.2f}%")
print("BibTeX file 'citations.bib' and LaTeX table 'citations_table.tex' created successfully.")