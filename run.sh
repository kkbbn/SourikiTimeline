#!/bin/bash -e

cd `dirname $0`

touch .local_version
local_version=`cat .local_version`
current_version=`cat VERSION`
if [ "$local_version" != "$current_version" ]; then
  ./setup.sh true
fi

source ./venv/bin/activate

pip3 freeze

python3 ./launch.py

deactivate
