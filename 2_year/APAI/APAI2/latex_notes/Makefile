.PHONY: all fast full clean

# Compilazione veloce: ricompila solo le parti modificate
fast:
    latexmk -r $(PWD)/.latexmkrc -pdf -outdir=build -silent main.tex

# Compilazione completa: forza ricompilazione totale
full:
	latexmk -r $(PWD)/.latexmkrc -pdf -outdir=build -g main.tex

# Default
all: fast

# Pulizia
clean:
	latexmk -C -outdir=build
