# compile pandoc md
DOC=README
RAW=.${DOC}.md
RST=$DOC.rst
PANDOC_FILTERS=$HOME/.cabal/bin/

$PANDOC_FILTERS/pandoc \
  --filter $PANDOC_FILTERS/pandoc-citeproc \
  --filter $PANDOC_FILTERS/pandoc-csv2table \
  --csl=assets/or.csl \
  -N \
  -o $RST -s $RAW
