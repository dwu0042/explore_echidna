DIAG_IN = $(wildcard *.tex)
DIAG_OUT = $(addsuffix .pdf,$(basename $(DIAG_IN)))

%.pdf: %.tex
	tectonic $<

default: $(DIAG_OUT)