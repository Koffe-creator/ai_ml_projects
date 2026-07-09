"""Download missense variants that have BOTH an AlphaMissense score and a
ClinVar clinical label, for a set of disease genes.

We use MyVariant.info, which aggregates dbNSFP (where AlphaMissense lives) and
ClinVar in one free API. For each gene we page through all matching variants
and save the fields we need to data/variants.csv.

Run it with:
    python fetch_variants.py
"""

import csv
import time

import requests

# disease genes with many ClinVar-labelled variants (NPC1 is our spotlight)
genes = ["NPC1", "NPC2", "BRCA1", "BRCA2", "TP53", "MLH1", "MSH2", "PTEN", "LDLR", "CFTR"]

fields = ",".join([
    "clinvar.rcv.clinical_significance",
    "dbnsfp.alphamissense.score",
    "dbnsfp.alphamissense.pred",
    "dbnsfp.genename",
    "dbnsfp.hgvsp",
])


def short_protein_change(hgvsp):
    """dbNSFP gives protein changes like ['p.Ala1018Thr', 'p.A1018T'].
    Return the short single-letter form (e.g. 'A1018T'), or '' if not found."""
    if isinstance(hgvsp, str):
        hgvsp = [hgvsp]
    if not hgvsp:
        return ""
    for item in hgvsp:
        text = item.replace("p.", "")
        # short form is like A1018T: letter, digits, letter
        if len(text) >= 3 and text[0].isalpha() and text[-1].isalpha() and any(c.isdigit() for c in text):
            if text[1].isdigit():   # single-letter code, e.g. A1018T
                return text
    return hgvsp[-1].replace("p.", "")


rows = []
for gene in genes:
    offset = 0
    while True:
        params = {
            "q": "clinvar.gene.symbol:%s AND _exists_:dbnsfp.alphamissense" % gene,
            "fields": fields,
            "size": 1000,
            "from": offset,
        }
        r = requests.get("https://myvariant.info/v1/query", params=params, timeout=60)
        hits = r.json().get("hits", [])
        if not hits:
            break

        for h in hits:
            am = h.get("dbnsfp", {}).get("alphamissense", {})
            clin = h.get("clinvar", {}).get("rcv", {})
            # rcv can be a dict or a list of dicts
            if isinstance(clin, list):
                significance = clin[0].get("clinical_significance", "")
            else:
                significance = clin.get("clinical_significance", "")

            protein_change = short_protein_change(h.get("dbnsfp", {}).get("hgvsp", ""))

            rows.append({
                "gene": gene,
                "variant_id": h.get("_id", ""),
                "protein_change": protein_change,
                "clinvar_significance": significance,
                "am_score": am.get("score", ""),
                "am_pred": am.get("pred", ""),
            })

        offset += 1000
        time.sleep(0.3)          # be gentle to the API
        if offset >= 9000:       # safety cap
            break

    print("%-6s  fetched, running total: %d" % (gene, len(rows)))

with open("data/variants.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("saved %d variants to data/variants.csv" % len(rows))
