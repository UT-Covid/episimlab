# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
GRAPHVIZ_DOT_ARGS ?= -D graphviz_dot_args="-Gratio=compress"
SPHINXOPTS    ?= $(GRAPHVIZ_DOT_ARGS)
SPHINXAUTOBUILD ?= sphinx-autobuild
SPHINXBUILD   ?= sphinx-build
PKGDIR        ?= ../episimlab
SOURCEDIR     = .
BUILDDIR      = _build
GITHUB_DIR ?= ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean api-ref

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: clean api-ref Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	cp -a "$(BUILDDIR)/$@/" "$(GITHUB_DIR)"

api-ref:
	rm -rf $(SOURCEDIR)/$@
	sphinx-apidoc -o $(SOURCEDIR)/$@ --force $(PKGDIR) $(PKGDIR)

clean:
	rm -rf $(GITHUB_DIR)/* $(BUILDDIR)/*

livehtml:
	@$(SPHINXAUTOBUILD) -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"