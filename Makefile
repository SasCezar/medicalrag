ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

setup_env:
	uv sync
	download_data

download_synthea:
	wget -O ./tools/synthea-with-dependencies.jar https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar


TOOLS_DIR := $(ROOT)/tools/synthea

generate_data:
	( cd "$(TOOLS_DIR)" && \
	  java -jar synthea-with-dependencies.jar \
	    -c synthea.properties -s 42 -cs 42 -p 500 Massachusetts )



webapp:
	uvicorn medrag.infrastructure.api:app --reload --env-file .env
