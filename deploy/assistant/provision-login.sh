#!/bin/sh
# One-time native login preparation. No service, credentials or inference.
set -eu

if [ "$(id -u)" != 0 ]; then
  echo 'Run this provisioning script as root.' >&2; exit 1
fi
if [ "$#" != 1 ]; then
  echo 'Usage: provision-login.sh /absolute/path/to/verified/claude-2.1.309' >&2; exit 1
fi
assistant_binary_source=$1
case "$assistant_binary_source" in
  /*) ;;
  *) echo 'The verified binary path must be absolute.' >&2; exit 1 ;;
esac
if [ ! -f "$assistant_binary_source" ] || [ ! -x "$assistant_binary_source" ]; then
  echo 'The verified binary must be an executable regular file.' >&2; exit 1
fi
assistant_binary_version=$("$assistant_binary_source" --version)
case "$assistant_binary_version" in
  '2.1.309 (Claude Code)') ;;
  *) echo 'Unexpected Claude version; qualify it before provisioning.' >&2; exit 1 ;;
esac

# Never repair/reuse an existing identity or app-owned tree. Privileged writes
# must not traverse paths that an existing runner process could replace.
if getent passwd reskin-assistant >/dev/null || getent group reskin-assistant >/dev/null; then
  echo 'Dedicated identity already exists; inspect it without rerunning provisioning.' >&2; exit 1
fi
for assistant_dir in /opt/reskin-assistant /var/lib/reskin-assistant; do
  if [ -e "$assistant_dir" ] || [ -L "$assistant_dir" ]; then
    echo 'Dedicated installation already exists; no changes performed.' >&2; exit 1
  fi
done
for assistant_parent in /opt /var /var/lib; do
  if [ -L "$assistant_parent" ] || [ ! -d "$assistant_parent" ] ||
      [ "$(stat -c %u:%g:%a "$assistant_parent")" != 0:0:755 ]; then
    echo 'Installation parent must be a real root-owned mode-0755 directory.' >&2; exit 1
  fi
done

# mkdir without -p refuses a concurrent provisioning attempt. Every directory
# stays root-owned until the complete layout and new identity exist.
mkdir -m 0755 /opt/reskin-assistant
mkdir -m 0755 /opt/reskin-assistant/bin
mkdir -m 0700 /var/lib/reskin-assistant
mkdir -m 0700 /var/lib/reskin-assistant/config
mkdir -m 0700 /var/lib/reskin-assistant/work
install -o root -g root -m 0755 "$assistant_binary_source" /opt/reskin-assistant/bin/claude
useradd --system --user-group --home-dir /var/lib/reskin-assistant --shell /usr/sbin/nologin reskin-assistant
chown reskin-assistant:reskin-assistant /var/lib/reskin-assistant/config /var/lib/reskin-assistant/work
# Transfer the parent last. No privileged writes follow beneath that tree.
chown reskin-assistant:reskin-assistant /var/lib/reskin-assistant
echo 'Native login layout prepared. No credentials copied and no service started.'
sha256sum /opt/reskin-assistant/bin/claude
