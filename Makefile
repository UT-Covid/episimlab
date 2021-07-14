PYTHON ?= python3
PREF_SHELL ?= bash
CC ?= gcc-10
GITREF=$(shell git rev-parse --short HEAD)
GITREF_FULL=$(shell git rev-parse HEAD)
VERSION = $(shell python setup.py --version)
PKG = $(shell python setup.py --name)
PKGL = $(shell echo $(PKG) | tr '[:upper:]' '[:lower:]')
PYTEST_OPTS ?= -n 16

.PHONY: sdist pytest pytest-native pytest-tox clean clean-tests cython docs
.SILENT: 

sdist: dist/$(PKG)-$(VERSION).tar.gz

dist/$(PKG)-$(VERSION).tar.gz: 
	$(PYTHON) setup.py sdist -q

pytest: pytest-tox 

pytest-native: clean-tests 
	$(PYTHON) -m pytest $(PYTEST_OPTS)

pytest-tox: clean-tests 
	tox -- $(PYTEST_OPTS)

clean: clean-tests

clean-tests:
	rm -rf .hypothesis .pytest_cache __pycache__ */__pycache__ \
		tmp.* *junit.xml local-mount *message_*_*.json logs/*.log \
		logs/*.o* logs/*.e* logs/*.html

cython: 
	CC=$(CC) $(PYTHON) setup.py build_ext --inplace

docs:
	cp episimlab/**/*.html docsrc/_static || true
	cd docsrc && make html
