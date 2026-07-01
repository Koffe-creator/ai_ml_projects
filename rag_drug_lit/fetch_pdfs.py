"""Download a few open-access atopic-dermatitis papers as PDF from Europe PMC.

The PDFs are saved into raw_pdfs/. Afterwards, run ingest_pdfs.py to turn them
into plain text that the retriever can use. Only open-access articles (which
are free to download) are listed here.

Run it with:
    python fetch_pdfs.py
"""

import os
import urllib.request

# open-access PubMed Central articles on atopic dermatitis
pmc_ids = [
    "PMC12116442",   # skin and systemic infections in children with atopic dermatitis
    "PMC13280930",   # association between atopic dermatitis and rosacea
    "PMC13280072",   # achieving high treatment targets in atopic dermatitis
]

os.makedirs("raw_pdfs", exist_ok=True)

for pmc in pmc_ids:
    url = "https://europepmc.org/articles/" + pmc + "?pdf=render"
    out = os.path.join("raw_pdfs", pmc + ".pdf")

    # some servers reject the default urllib user-agent, so set a normal one
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()

    if not data.startswith(b"%PDF"):
        print("skipped (not a PDF):", pmc)
        continue

    with open(out, "wb") as f:
        f.write(data)
    print("downloaded", out, "(%d bytes)" % len(data))

print("Done. Now run: python ingest_pdfs.py")
