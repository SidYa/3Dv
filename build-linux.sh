#!/usr/bin/env bash
# Builds the Linux packages: core/target/release/bundle/{deb,rpm}
#   ./build-linux.sh
# Run from a checkout inside the Linux file system (on WSL /mnt/... is very
# slow, and node_modules installed on Windows holds win32 binaries).
# Needs Rust, Node.js and Tauri's Linux dependencies (libwebkit2gtk-4.1-dev ...).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

# Release binaries carry no paths of this machine (user name, checkout
# location) in their panic messages.
export RUSTFLAGS="--remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=cargo --remap-path-prefix=${RUSTUP_HOME:-$HOME/.rustup}=rustup --remap-path-prefix=$root=3Dv"

cd "$root/app"
[ -d node_modules ] || npm ci --no-audit --no-fund
npm run build

cd "$root/core"
# The frontend is already built; Tauri would run its build command from the
# wrong directory.
../app/node_modules/.bin/tauri build --bundles deb,rpm -c '{"build":{"beforeBuildCommand":""}}'
