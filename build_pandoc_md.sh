# compile pandoc md
DOC=$1$2
MD=${DOC}.md
HTML=${DOC}.html
TEX=$DOC.tex
PANDOC_FILTERS=$HOME/.cabal/bin/

cd $DOC/ &&
  $PANDOC_FILTERS/pandoc \
    --filter $PANDOC_FILTERS/pandoc-citeproc \
    --filter $PANDOC_FILTERS/pandoc-csv2table \
    --katex --toc \
    --csl=assets/ieee.csl \
    --css=assets/pandoc.css \
    -o $HTML \
    assets/meta.yaml -s $MD

echo $PWD
# latex
# exit
$PANDOC_FILTERS/pandoc \
  --filter $PANDOC_FILTERS/pandoc-citeproc \
  --filter $PANDOC_FILTERS/pandoc-csv2table \
  --pdf-engine=xelatex \
  --csl=assets/ieee.csl \
  --template=assets/markdown.tex \
  -t latex \
  -N \
  -o $TEX \
  assets/meta.yaml -s $MD

latexmk -C -cd $TEX
latexmk --xelatex -f -cd -quiet $TEX
latexmk -c -cd $TEX
