# upLaTeX → dvipdfmx（研究室テンプレート jsarticle + uplatex 用）
$latex     = 'uplatex %O -kanji=utf8 -no-guess-input-enc -synctex=1 -interaction=nonstopmode %S';
$lualatex  = 'lualatex %O -synctex=1 -interaction=nonstopmode %S';
$biber     = 'biber %O --bblencoding=utf8 -u -U --output_safechars %B';
$bibtex    = 'upbibtex %O %B';
$makeindex = 'upmendex %O -o %D %S';
$dvipdf    = 'dvipdfmx %O -o %D %S';
$pdf_mode  = 3;
$pdf_previewer = 'open %S';
