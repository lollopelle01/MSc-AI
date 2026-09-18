$out_dir = 'build';
$aux_dir = 'build';
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 -output-directory=build %O %S';
$biber = 'cd build && biber %O %S';
$clean_ext = 'synctex.gz aux log bbl blg out toc fls fdb_latexmk';
push @default_files, 'images';
