#!/bin/bash -ex

PYTHON_VERSION="${PYTHON_VERSION:-3}"

cd `dirname $0`

#skip_key_wait=$1

"python${PYTHON_VERSION}" -m venv venv
source ./venv/bin/activate

"pip${PYTHON_VERSION}" install --upgrade -r requirements-unix.txt

"pip${PYTHON_VERSION}" freeze

cat VERSION > .local_version

#if [ "$skip_key_wait" != "true" ]; then
#  read -p "All complate!!! plass any key..."
#fi

deactivate
