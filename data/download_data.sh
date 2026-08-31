#!/bin/bash
# Downloads a small, prebuilt BM25 (Lucene) index + ANCE (FAISS) index over a
# ~9K-passage MS MARCO subset (all passages judged relevant somewhere in
# TREC DL2019), plus the DL2019 queries/qrels. Good for a quick, realistic
# end-to-end demo without indexing the full 8.8M-passage MS MARCO corpus.
set -e

URL="https://github.com/jaehwan-jung7/bm25-ance-retrieval-kit/releases/download/data-v1/dl2019_mini.zip"
DEST_DIR="$(dirname "$0")"
ZIP_PATH="$DEST_DIR/dl2019_mini.zip"

echo "Downloading $URL"
curl -L "$URL" -o "$ZIP_PATH"

echo "Extracting into $DEST_DIR"
unzip -o "$ZIP_PATH" -d "$DEST_DIR"
rm "$ZIP_PATH"

echo "Done. Contents of $DEST_DIR:"
ls -la "$DEST_DIR"
