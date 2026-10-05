#!/bin/zsh
# Execute the solutions copies in place (student copies are not executed).
# Usage: build/execute.sh 00 01 ...   (no args = all)
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then set -- 00 01 02 03 04 05 06 07 08; fi
for n in "$@"; do
  f=$(ls notebooks/${n}_*_solutions.ipynb)
  echo "== executing $f"
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace "$f" \
      --ExecutePreprocessor.kernel_name=swc-ephys-one --ExecutePreprocessor.timeout=600 2>&1 | grep -v "^\[NbConvertApp\]" || true
done
