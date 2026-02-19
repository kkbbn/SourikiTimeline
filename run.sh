#!/bin/bash -e

PYTHON_VERSION="${PYTHON_VERSION:-3}"

cd `dirname $0`

touch .local_version
local_version=`cat .local_version`
current_version=`cat VERSION`
if [ "$local_version" != "$current_version" ]; then
  ./setup.sh true
fi

source ./venv/bin/activate

"pip${PYTHON_VERSION}" freeze

"python${PYTHON_VERSION}" ./launch.py

deactivate
