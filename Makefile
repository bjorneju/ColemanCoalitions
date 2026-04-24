.PHONY: paper

# Compile the working-paper draft and copy it to assets/draft.pdf.
# Run twice so cross-references and outlines are resolved correctly.
paper:
	cd drafts && pdflatex -interaction=nonstopmode coleman_coalitions_draft.tex
	cd drafts && pdflatex -interaction=nonstopmode coleman_coalitions_draft.tex
	cp drafts/coleman_coalitions_draft.pdf assets/draft.pdf
	@echo "Done — assets/draft.pdf updated."
