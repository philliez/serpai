#!/bin/sh
source ./bin/activate
python -u -m flask --app api run -p $PORT --debug