.PHONY: clean

all:
	mkdir -p build
	lualatex -shell-escape draft.tex
	# bibtex draft
	# lualatex -shell-escape draft.tex
	# lualatex -shell-escape draft.tex
	mv draft.aux draft.log build
