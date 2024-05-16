.PHONY: clean

all:
	mkdir -p build
	# lualatex -shell-escape draft.tex
	# bibtex draft
	# lualatex -shell-escape draft.tex
	# lualatex -shell-escape draft.tex
	# mv draft.aux draft.log build


	lualatex -shell-escape report.tex
	bibtex report
	lualatex -shell-escape report.tex
	lualatex -shell-escape report.tex
	mv report.aux report.log build
