"""Download real paper abstracts from PubMed into the corpus.

For each topic we search PubMed, take the most relevant article that has an
abstract, and save its title + abstract + PMID as a .txt file in corpus/.
This replaces the sample notes with real, citable literature.

Uses NCBI's free E-utilities API (no key needed for a few requests).

Run it with:
    python fetch_pubmed.py
"""

import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# filename in corpus/  ->  what to search PubMed for.
# All topics are about atopic dermatitis, so the corpus is focused on one disease.
TOPICS = {
    "ad_overview.txt":    "atopic dermatitis pathophysiology review",
    "ad_dupilumab.txt":   "dupilumab atopic dermatitis IL-4 IL-13 review",
    "ad_jak.txt":         "JAK inhibitor atopic dermatitis review",
    "ad_filaggrin.txt":   "filaggrin gene mutation skin barrier dysfunction atopic dermatitis",
    "ad_microbiome.txt":  "Staphylococcus aureus skin microbiome atopic dermatitis",
    "ad_topical.txt":     "topical corticosteroids calcineurin inhibitors atopic dermatitis treatment",
    "ad_pediatric.txt":   "pediatric childhood atopic dermatitis management",
    "ad_itch.txt":        "itch pruritus mechanism atopic dermatitis",
}


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def search_pmids(query, n=5):
    """Return a list of PubMed IDs for a search query, most relevant first."""
    params = urllib.parse.urlencode({
        "db": "pubmed", "term": query, "retmax": n, "sort": "relevance",
    })
    root = ET.fromstring(get(EUTILS + "esearch.fcgi?" + params))
    return [e.text for e in root.findall(".//IdList/Id")]


def fetch_article(pmid):
    """Return (title, abstract, journal, year) for a PMID, or None if no abstract."""
    params = urllib.parse.urlencode({"db": "pubmed", "id": pmid, "retmode": "xml"})
    root = ET.fromstring(get(EUTILS + "efetch.fcgi?" + params))

    article = root.find(".//Article")
    if article is None:
        return None

    title = article.findtext("ArticleTitle", default="")

    # An abstract can have several labeled sections; join them.
    parts = [a.text for a in article.findall(".//Abstract/AbstractText") if a.text]
    abstract = " ".join(parts)
    if not abstract:
        return None

    journal = article.findtext(".//Journal/Title", default="")
    year = article.findtext(".//JournalIssue/PubDate/Year", default="")
    return title, abstract, journal, year


def main():
    used_pmids = set()   # so two topics don't save the same paper

    for filename, query in TOPICS.items():
        chosen = None
        for pmid in search_pmids(query):
            if pmid in used_pmids:
                continue
            time.sleep(0.4)               # be polite to the NCBI server
            article = fetch_article(pmid)
            if article:
                chosen = (pmid, article)
                used_pmids.add(pmid)
                break

        if not chosen:
            print("  no abstract found for: " + query)
            continue

        pmid, (title, abstract, journal, year) = chosen
        text = (
            "Title: " + title + "\n"
            "Source: " + journal + " (" + year + "), PMID " + pmid + "\n\n"
            + abstract + "\n"
        )
        with open("corpus/" + filename, "w") as f:
            f.write(text)
        print("  " + filename + "  <-  PMID " + pmid + "  (" + str(len(abstract.split())) + " words)")
        time.sleep(0.4)

    print("Done. corpus/ now holds real PubMed abstracts.")


if __name__ == "__main__":
    main()
