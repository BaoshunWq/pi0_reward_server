#!/usr/bin/env bash

for suite in libero_spatial libero_object libero_goal libero_10
do
  echo "Running task suite: $suite"
  python openpi/examples/libero/noaction_test_pi0.py --args.task-suite-name "$suite"
done
