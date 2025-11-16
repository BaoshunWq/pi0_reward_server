#!/usr/bin/env bash

for suite in libero_spatial libero_object libero_goal libero_10
do
  echo "Running task suite: $suite"
  python openpi/examples/libero/main.py --args.task-suite-name "$suite" --args.port 4444
done
