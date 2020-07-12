DOC=$1

mkdir $DOC
ln -s $PWD/assets $DOC/assets
cat $PWD/assets/meta.yaml $PWD/assets/series.yaml >$DOC/$DOC.yaml
echo "# References" >$DOC/$DOC.ref.md
echo "" >$DOC/$DOC.bib
