"""Turn PDF papers into the plain-text files the corpus uses.

This is the ingestion step: drop PDFs into raw_pdfs/, run this script once,
and it writes one .txt per PDF into corpus/. The retriever then works on
those .txt files exactly as before - it never has to know about PDFs.

We extract plain text only (no formatting, no images), because TF-IDF
retrieval just needs words.

Run it with:
    python ingest_pdfs.py
"""

import os
import glob

from pypdf import PdfReader


def pdf_to_text(pdf_path):
    """Read every page of a PDF and return its text as one string."""
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text())
    return "\n".join(pages)


def main():
    pdf_files = sorted(glob.glob("raw_pdfs/*.pdf"))
    if not pdf_files:
        print("No PDFs found in raw_pdfs/. Drop some .pdf files there first.")
        return

    os.makedirs("corpus", exist_ok=True)

    for pdf_path in pdf_files:
        text = pdf_to_text(pdf_path)

        # Name the output after the PDF, e.g. raw_pdfs/smith2021.pdf -> corpus/smith2021.txt
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join("corpus", name + ".txt")

        with open(out_path, "w") as f:
            f.write(text)

        print("  " + pdf_path + "  ->  " + out_path + "  (" + str(len(text.split())) + " words)")

    print("Done. Re-run the retriever or eval to use the new documents.")


if __name__ == "__main__":
    main()
