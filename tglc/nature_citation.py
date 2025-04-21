import bibtexparser
from bibtexparser.bparser import BibTexParser

# ========== CONFIGURATION ==========
input_bib = "citations_kepler.bib"
output_bib = "apjl_kepler.bib"


# ===================================

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


def generate_unique_keys(entries):
    used_keys = set()
    for entry in entries:
        # Extract author and year
        authors_str = entry.get('author', '')
        authors = [parse_author(a) for a in authors_str.split(' and ')] if authors_str else []
        year = entry.get('year', '')

        # Generate base key
        if authors:
            base_key = f"{authors[0][0]}{year}"
        else:
            base_key = f"Unknown{year}"

        # Ensure uniqueness
        suffix = ''
        new_key = base_key + suffix
        while new_key in used_keys:
            suffix = 'b' if suffix == '' else chr(ord(suffix) + 1)
            new_key = base_key + suffix

        used_keys.add(new_key)
        entry['ID'] = new_key
    return entries


def main():
    # Read input .bib file
    with open(input_bib, 'r', encoding='utf-8') as f:
        parser = BibTexParser(common_strings=True)
        bib_db = bibtexparser.load(f, parser=parser)

    # Generate unique keys for each entry
    generate_unique_keys(bib_db.entries)

    # Write the modified entries to a new .bib file
    with open(output_bib, 'w', encoding='utf-8') as f:
        bibtexparser.dump(bib_db, f)


if __name__ == '__main__':
    main()