# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: arto
# @file: notes/build_pandoc_series.sh
# @created: Tuesday, 16th June 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Sunday, 12th July 2020 12:49:34 am
# @description:
#   compile series of pandoc markdown files.
#   by series you have a list of markdown for one topic
#     e.g. convex-1, convex-2, ...
#   that share unique assets like yaml, css, js, bibliography, etc.

TOPIC=$1

MD=${TOPIC}*.md
HTML=${TOPIC}.html
TEX=$TOPIC.tex
PANDOC_FILTERS=$HOME/.cabal/bin/
STANDALONE=all.${TOPIC}.md
MD_FILES=$(ls $TOPIC/$TOPIC*.md | sort)

# delete old files

echo "...clearing old cache..."
rm $TOPIC/$STANDALONE
rm $TOPIC/*.aux $TOPIC/*.fdb* \
  $TOPIC/*.fls $TOPIC/*.xdv \
  $TOPIC/*.toc $TOPIC/*.log

echo "...combining pandoc md files..."
for f in $MD_FILES; do
  echo "...parsing...$f"
  cat $f >>$TOPIC/$STANDALONE
done
echo "...done..."

cd $TOPIC/ &&
  $PANDOC_FILTERS/pandoc \
    --filter $PANDOC_FILTERS/pandoc-citeproc \
    --filter $PANDOC_FILTERS/pandoc-csv2table \
    --filter pandoc-include \
    --katex --toc \
    --csl=assets/institute-for-operations-research-and-the-management-sciences.csl \
    --css=assets/pandoc.css \
    -o $HTML \
    --bibliography $TOPIC.bib \
    $TOPIC.yaml \
    -s $STANDALONE

echo $PWD
# latex
# exit
$PANDOC_FILTERS/pandoc \
  --filter $PANDOC_FILTERS/pandoc-citeproc \
  --filter $PANDOC_FILTERS/pandoc-csv2table \
  --filter pandoc-include \
  --pdf-engine=xelatex \
  --csl=assets/or.csl \
  --template=assets/markdown.tex \
  -t latex \
  -o $TEX -N \
  --toc \
  --bibliography $TOPIC.bib \
  $TOPIC.yaml \
  -s $STANDALONE

latexmk -C -cd $TEX
latexmk --xelatex -f -cd -quiet $TEX
# xetex -interaction=nonstopmode cvx-rv.tex
