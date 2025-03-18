import json
import bibtexparser
from bibtexparser.bparser import BibTexParser

# ========== CONFIGURATION ==========
input_bib = "g_ng_citation.bib"
output_entries = "entries.tex"
output_refs = "references.tex"
start_num = 1
# ===================================

# Expanded journal name mapping (update/add entries here)
journal_mapping = {
    'apjl': 'The Astrophysical Journal Letters',
    'apjlett': 'The Astrophysical Journal Letters',
    'apjs': 'The Astrophysical Journal Supplement Series',
    'aj': 'The Astronomical Journal',
    'mnras': 'Monthly Notices of the Royal Astronomical Society',
    'aap': 'Astronomy & Astrophysics',
    'apj': 'The Astrophysical Journal',
    'pasp': 'Publications of the Astronomical Society of the Pacific',
    'nat': 'Nature',
    'pasa': 'Publications of the Astronomical Society of Australia',
    'pasj': 'Publications of the Astronomical Society of Japan'
    # Add more mappings as needed
}

def get_journal_name(journal_field):
    # Remove ALL braces and backslashes, then lowercase
    cleaned = journal_field.replace('{', '').replace('}', '').replace('\\', '').lower()
    return journal_mapping.get(cleaned, journal_field)


def process_first_name(first):
    parts = first.split()
    initials = []
    for part in parts:
        clean_part = part.split('{')[0]
        if not clean_part:
            if len(part) > 1:
                initial = part[1]
            else:
                initial = part[0]
        else:
            initial = clean_part[0]
        initials.append(f"{initial}.")
    return ' '.join(initials)


def parse_author(author_str):
    author_str = author_str.strip('{}')
    if ',' in author_str:
        last, first = author_str.split(',', 1)
        last = last.strip().strip('{}')
        first = first.strip()
    else:
        last = author_str.strip('{}')
        first = ''
    return last, first


def format_author_list(authors):
    if not authors:
        return ''
    elif len(authors) == 1:
        last, first = authors[0]
        initials = process_first_name(first)
        return f"{last}, {initials}"
    elif len(authors) == 2:
        a1 = f"{authors[0][0]}, {process_first_name(authors[0][1])}"
        a2 = f"{authors[1][0]}, {process_first_name(authors[1][1])}"
        return f"{a1} & {a2}"
    else:
        a1 = f"{authors[0][0]}, {process_first_name(authors[0][1])}"
        return f"{a1} et al."
# ... [Keep other functions like process_first_name, format_author_list, etc.] ...

def process_bib_entries(entries, start_num):
    processed = []
    used_name = set()
    for entry in entries:
        authors = [parse_author(a) for a in entry.get('author', '').split(' and ')]
        formatted_authors = format_author_list(authors)
        title = entry.get('title', '').strip('{}')
        journal = get_journal_name(entry.get('journal', ''))  # Key fix here
        volume = entry.get('volume', '')
        pages = entry.get('pages', entry.get('eid', ''))
        year = entry.get('year', '')
        key = f"{authors[0][0]}{year}" if authors else f"Unknown{year}"
        suffix = ''
        while key + suffix in used_name:
            if not suffix:
                suffix = 'b'  # Start with 'b'
            else:
                suffix = chr(ord(suffix) + 1)  # Move to next letter
        key += suffix
        ID = entry.get('ID', '')
        processed.append({
            'authors': formatted_authors,
            'title': title,
            'journal': journal,
            'volume': volume,
            'pages': pages,
            'year': year,
            'key': key,
            'ID': ID
        })
        used_name.add(key)

    entry_lines = []
    ref_lines = []
    convert = {}
    for idx, entry in enumerate(processed, start=start_num):
        entry_line = f"\\noindent {idx}. {entry['authors']} et al., {entry['title']}, {entry['journal']}, {entry['volume']}, {entry['pages']} ({entry['year']}). \\\\ \\vskip 0.2cm\n"
        entry_lines.append(entry_line)
        ref_lines.append(f"\\reference{{{entry['key']}}}{{$^{idx}$}}\n")
        ref_lines.append(f"\\reference{{{entry['key']}-r}}{{{idx}}}\n")
        convert[entry['ID']] = entry['key']
    return entry_lines, ref_lines, convert


def main():
    # Read input .bib file
    with open(input_bib, 'r', encoding='utf-8') as f:
        parser = BibTexParser(common_strings=True)
        bib_db = bibtexparser.load(f, parser=parser)

    # Process entries and generate LaTeX
    entries, refs, convert = process_bib_entries(bib_db.entries, start_num)
    print(convert)
    # Write output files
    with open(output_entries, 'w', encoding='utf-8') as f:
        f.writelines(entries)
    with open(output_refs, 'w', encoding='utf-8') as f:
        f.writelines(refs)
    with open("dictionary.json", "w") as file:
        json.dump(convert, file, indent=4)  # Pretty formatting


if __name__ == '__main__':
    main()