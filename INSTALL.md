# Installing wpiug

`wpiug` is a single self-contained binary with no runtime dependencies. Build it
once, copy it onto your GPU nodes, done.

---

## 1. Requirements

**On the machine you build on**

| | |
| --- | --- |
| Rust | 1.70 or newer (`cargo`, `rustc`) |
| Network | not required — `wpiug` has zero crate dependencies and builds offline |

Rust 1.70 is the floor because the colour auto-detection uses
`std::io::IsTerminal`. Check yours with `rustc --version`; if you need a
toolchain, <https://rustup.rs> is the usual route.

**On the machine you run on**

| | |
| --- | --- |
| OS | Linux |
| GPU | AMD, driven by `amdgpu` |
| Kernel interfaces | `/sys/class/drm` and `/sys/class/kfd` |
| ROCm | **not required** — no `rocm-smi`, `amd-smi` or ROCm runtime needed |
| Root | **not required** — see [Permissions](#4-permissions) |

`wpiug` reads kernel sysfs directly rather than shelling out to ROCm tooling, so
it works on a bare driver install. NVIDIA GPUs are not supported.

---

## 2. Install

### With `cargo install` (recommended)

If you have Rust, this is the whole thing — build and install in one step:

```bash
cd wpiug
cargo install --path .
```

The binary lands in `~/.cargo/bin/wpiug`, which rustup already put on your
`PATH`, so `wpiug` works immediately in a new shell. `cargo install` always
builds in release mode and honours the `[profile.release]` settings in
`Cargo.toml`, so you get the same optimised ~490 KB binary as a manual build.

Re-running `cargo install --path .` after a `git pull` replaces the installed
copy — no `--force` needed for path installs. (If cargo ever refuses because the
same version is already installed, add `--force`.)

To install somewhere else — for example system-wide, so every user on the node
gets it — point `--root` at a prefix; the binary goes in `<prefix>/bin`:

```bash
sudo cargo install --path . --root /usr/local   # -> /usr/local/bin/wpiug
```

> `sudo cargo` only works if root has a Rust toolchain. If it doesn't, build as
> your normal user and copy the binary with the manual method below.

### With `cargo build` (manual)

Useful when you want the binary as an artifact — to copy onto other nodes, say —
rather than installed on this one:

```bash
cargo build --release
```

The binary lands at `target/release/wpiug` (~490 KB, builds in a few seconds).
Put it on your `PATH` for just your user:

```bash
mkdir -p ~/.local/bin
install -m755 target/release/wpiug ~/.local/bin/wpiug
```

If `~/.local/bin` isn't already on your `PATH`, add it:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && exec bash
```

Or system-wide, which needs no Rust toolchain for root:

```bash
sudo install -m755 target/release/wpiug /usr/local/bin/wpiug
```

---

## 3. Verify

```bash
wpiug --version      # -> wpiug 0.2.0
wpiug --help
wpiug                # queries every GPU
```

A healthy run prints the concise table with one row per GPU:

```
GPU   VRAM used / total      VRAM%   Util%   Procs
--------------------------------------------------
0      183.1GiB / 255.7GiB     71%     11%       3
1      178.7GiB / 255.7GiB     69%    100%       3
```

Cross-check the numbering against ROCm if you have it — GPU ids are assigned in
PCI-address order, which is the same ordering `rocm-smi` uses, so `wpiug 3` and
`rocm-smi`'s `GPU[3]` are the same card:

```bash
rocm-smi --showbus
```

---

## 4. Permissions

**No root, and no group membership, for GPU and process data.** Everything
`wpiug` reads is world-readable:

- `/sys/class/drm/card*/device/*` — mode `-r--r--r--`
- `/sys/class/kfd/kfd/proc/<pid>/*` — mode `-r--r--r--`, directories `drwxr-xr-x`
- `/proc/<pid>/status` and `/proc/<pid>/cgroup` — readable across users

You do **not** need to be in the `render` or `video` group: `wpiug` reads sysfs
attributes, it never opens `/dev/kfd` or `/dev/dri/*`. Processes owned by other
users show up normally.

**Exact VRAM for other users' processes needs root.** KFD's per-process VRAM
counter misses memory allocated through the render node — PyTorch's
`expandable_segments` puts nearly all of a job's memory there — so `wpiug` also
reads `/proc/<pid>/fdinfo`, which the kernel only shows for your own processes
(or anyone's, to root). Without root, other users' processes fall back to KFD's
figure; when that leaves more than 1 GiB on a GPU unaccounted for, `wpiug`
prints a note on stderr. Your own processes are always counted in full.

```bash
sudo "$(command -v wpiug)"   # exact VRAM for every process on the node
```

(`sudo wpiug` alone works only for a system-wide install — sudo's `PATH`
usually leaves out `~/.cargo/bin`.)

**Container *names* are the one exception.** Turning a container id into a
readable name requires talking to the container runtime, so it needs whatever
access that runtime demands — usually membership of the `docker` group:

```bash
sudo usermod -aG docker "$USER"   # log out and back in to take effect
```

Without that access nothing breaks — `wpiug` falls back to the short container
id, and still exits 0:

```
├── 1261635 sglang::schedul (32efcf744e1b)   170.7GiB     # no docker access
├── 1261635 sglang::schedul (llm-worker-1)   170.7GiB     # with docker access
```

`docker`, `podman` and `nerdctl` are all tried, only when a containerised
process is actually present, and each call is capped at 5 seconds so a wedged
daemon can never hang the tool.

> **Note on `hidepid`.** If `/proc` is mounted with `hidepid=2`, a non-root user
> cannot read other users' `/proc/<pid>` entries. GPU attribution still works,
> but other users' processes will render as `<pid> ?` with no container. Run as
> root on such hosts.

---

## 5. Deploying to other GPU nodes

The binary is dynamically linked against glibc, so a build from one node runs on
another **only if the target's glibc is the same or newer**. Check both ends:

```bash
ldd --version | head -1
```

Same distro image across the fleet? Just copy it:

```bash
scp target/release/wpiug node02:/tmp/
ssh node02 'sudo install -m755 /tmp/wpiug /usr/local/bin/wpiug && wpiug --version'
```

Mixed or older distros? Build a fully static binary that has no glibc dependency
at all — a one-time toolchain addition (this step needs network):

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
# -> target/x86_64-unknown-linux-musl/release/wpiug
```

That binary runs on any x86-64 Linux regardless of distro or glibc version.
The musl target is not installed on this build host yet, so the `rustup target
add` above is a prerequisite rather than something already verified here.

---

## 6. Uninstall

```bash
cargo uninstall wpiug                        # installed via `cargo install --path .`
sudo cargo uninstall wpiug --root /usr/local # installed via `cargo install --root`
rm -f ~/.local/bin/wpiug                     # copied manually
sudo rm -f /usr/local/bin/wpiug              # copied manually, system-wide
```

`wpiug` writes no config, no cache and no state — removing the binary removes
it completely.

---

## 7. Troubleshooting

**`wpiug: no AMD GPUs found under /sys/class/drm (vendor 0x1002 with mem_info_vram_total)`**

No card matched. Either the host has no AMD GPU, or the `amdgpu` driver isn't
loaded:

```bash
lsmod | grep amdgpu
ls -d /sys/class/drm/card*/device
```

**`wpiug: cannot read /sys/class/drm: ...`**

`sysfs` isn't mounted, or you're in a container without it. Bind-mount `/sys`
into the container, or run on the host.

**`wpiug: no GPU with id 8; this machine has 8 (0-7)`**

Ids are zero-based and ordered by PCI address. `wpiug` with no argument lists
every valid id.

**Container shows a hex id like `(32efcf744e1b)` instead of a name**

Expected when the container runtime isn't reachable — see
[Permissions](#4-permissions).

**Process Detail is empty but the GPU reports VRAM in use**

A few hundred MiB per idle card is firmware/driver reserve, not a process.
Beyond that, `wpiug` lists processes holding KFD (ROCm compute) resources; a
pure graphics client that never touches KFD won't appear.

**Tree characters render as garbage boxes**

Your terminal or log pipeline can't handle box-drawing characters:

```bash
wpiug --ascii
```

**Escape codes like `^[[36m` end up in a file**

Colour turns itself off when stdout isn't a terminal, so this only happens if
something forces it on. Use `wpiug --no-color`, or set `NO_COLOR=1`. To *keep*
colour through a pager, use `wpiug --color | less -R`.

**Output is very tall on training nodes**

Dataloader workers expand the tree. Reduce the depth:

```bash
wpiug --depth 1     # GPU processes only; other children hidden
```

**Several processes share one name, so I can't tell which job is which**

Show the command line each one was started with:

```bash
wpiug -w      # cut to the terminal width
wpiug -ww     # in full, however long
```
