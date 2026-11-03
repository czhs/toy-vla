#!/bin/bash
# Clone the one external code dependency of the arm pipeline at the commit it was run with. aviadarn/so101-lerobot carries the scripted DLS-IK
# expert (collect_sim.py) that expert.py imports; it has no licence file, so it is fetched here and not copied into this repository.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${SO101_REPO:-${TOYVLA_DATA:-$HERE/../data}/so101-lerobot}"
COMMIT=7fe686ca3845a1738a14ec756cf26920dac81605
if [ ! -d "$DST/.git" ]; then mkdir -p "$(dirname "$DST")"; git clone https://github.com/aviadarn/so101-lerobot "$DST"; fi
git -C "$DST" checkout --quiet "$COMMIT"
echo "so101-lerobot at $(git -C "$DST" rev-parse HEAD) in $DST"
