include subtrees/z_quantum_actions/Makefile

github_actions:
	python3 -m venv ${VENV} && \
		${PYTHON} -m pip install --upgrade pip && \
		${PYTHON} -m pip install ./orquestra-quantum
		${PYTHON} -m pip install -e '.[dev]'

install_internal_deps:
	${PYTHON} -m pip install --upgrade pip && \
	${PYTHON} -m pip install ./orquestra-quantum