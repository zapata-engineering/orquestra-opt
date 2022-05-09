include subtrees/z_quantum_actions/Makefile

github_actions:
	python3 -m venv ${VENV} && \
		${VENV}/bin/python3 -m pip install --upgrade pip && \
		${VENV}/bin/python3 -m pip install ./orquestra-quantum && \
		${VENV}/bin/python3 -m pip install -e '.[dev]'

install_internal_deps:
	${PYTHON} -m pip install --upgrade pip && \
	${PYTHON}/bin/python3 -m pip install ./orquestra-quantum