PYTHON ?= python3
PREF_SHELL ?= bash
CC=gcc-10
GITREF=$(shell git rev-parse --short HEAD)
GITREF_FULL=$(shell git rev-parse HEAD)

####################################
# Docker image & dist
####################################

VERSION = $(shell python setup.py --version)
PKG = $(shell python setup.py --name)
PKGL = $(shell echo $(PKG) | tr '[:upper:]' '[:lower:]')
IMAGE_ORG ?= enho
IMAGE_NAME ?= $(PKGL)
IMAGE_TAG ?= $(VERSION)
IMAGE_DOCKER ?= $(IMAGE_ORG)/$(IMAGE_NAME):$(IMAGE_TAG)
DOCKER_OPTS ?= --rm -itv ${PWD}/outputs:/outputs -v ${PWD}/inputs:/inputs

####################################
# Sanity checks
####################################

PROGRAMS := git docker python singularity tox $(CC)
.PHONY: $(PROGRAMS)
.SILENT: $(PROGRAMS)

docker:
	docker info 1> /dev/null 2> /dev/null && \
	if [ ! $$? -eq 0 ]; then \
		echo "\n[ERROR] Could not communicate with docker daemon. You may need to run with sudo.\n"; \
		exit 1; \
	fi
python poetry singularity $(CC):
	$@ --help &> /dev/null; \
	if [ ! $$? -eq 0 ]; then \
		echo "[ERROR] $@ does not seem to be on your path. Please install $@"; \
		exit 1; \
	fi
tox:
	$@ -h &> /dev/null; \
	if [ ! $$? -eq 0 ]; then \
		echo "[ERROR] $@ does not seem to be on your path. Please pip install $@"; \
		exit 1; \
	fi
git:
	$@ -h &> /dev/null; \
	if [ ! $$? -eq 129 ]; then \
		echo "[ERROR] $@ does not seem to be on your path. Please install $@"; \
		exit 1; \
	fi

####################################
# Build Docker image
####################################
.PHONY: shell tests tests-pytest clean clean-image clean-tests dist/$(PKG)-$(VERSION).tar.gz

dist/$(PKG)-$(VERSION).tar.gz: tox.ini setup.py MANIFEST.in pyproject.toml | python
	python setup.py sdist -q

image: docker/Dockerfile dist/$(PKG)-$(VERSION).tar.gz requirements.txt | docker
	cp $(word 2, $^) .
	docker build --build-arg SDIST=$(PKG)-$(VERSION) \
		--build-arg REQUIREMENTS=$(word 3, $^) \
		-t $(IMAGE_DOCKER) -f $< .
	rm $(word 2, $^) $(PKG)-$(VERSION).tar.gz

####################################
# Pytest
####################################
PYTEST_OPTS ?= -svvvm 'not slow'
.PHONY: pytest-native pytest-tox

pytest: cython pytest-native

pytest-native: clean-tests | python
	PYTHONPATH=./src $(PYTHON) -m pytest $(PYTEST_OPTS)

pytest-tox: clean-tests | tox $(CC)
	CC=$(CC) $(PYTHON) -m tox -- $(PYTEST_OPTS)

benchmark: image | docker
	docker run --rm -itv ${PWD}:/pwd \
		$(IMAGE_DOCKER) \
		bash /pwd/scripts/20200922_benchmark.sh

####################################
# Docker tests
####################################

shell: image | docker
	docker run --rm -it -v ${PWD}:/pwd $(IMAGE_DOCKER) bash

clean: clean-tests

clean-rmi: image | docker
	docker rmi -f $$(docker images -q --filter=reference="$(IMAGE_ORG)/$(IMAGE_NAME):*" --filter "before=$(IMAGE_DOCKER)")

clean-tests:
	rm -rf .hypothesis .pytest_cache __pycache__ */__pycache__ \
		tmp.* *junit.xml local-mount *message_*_*.json logs/*.log \
		logs/*.o* logs/*.e* logs/*.html

####################################
# Cythonize
####################################
.PHONY: cython

cython: | $(CC)
	CC=$(CC) $(PYTHON) setup.py build_ext --inplace

####################################
# Jupyterhub
####################################
IMAGE_JHUB ?= $(IMAGE_ORG)/$(IMAGE_NAME):jhub

image-jhub: docker/Dockerfile.jhub requirements_files/requirements-jhub.txt | docker
	docker build --build-arg REQUIREMENTS=$(word 2,$^) -t $(IMAGE_JHUB) -f $< .

jhub: image-jhub | docker
	docker run -p 8888:8888 \
		-v ${PWD}:/home/jovyan/work \
		-e JUPYTER_ENABLE_LAB=yes \
		$(IMAGE_JHUB)

####################################
# Sphinx
####################################
.PHONY: docs

docs:
	cp src/SEIRcity/model/cy_model.html docsrc/_static || true
	cd docsrc && make html
