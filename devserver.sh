#!/bin/sh
source ./bin/activate
python -u -m flask --app main run -p $PORT --debug