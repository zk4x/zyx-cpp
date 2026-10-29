#!/bin/bash
set -e

cd "$(dirname "$0")/tests"
make clean
make -j "$@"
for test in test_buffer test_tensor test_activations test_losses test_layers test_models; do
  ./$test
done
