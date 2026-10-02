# wpiug — What Process Is Using GPU

`wpiug` shows what is running on each AMD GPU in a machine: VRAM and utilisation
per GPU, the process tree behind every GPU user, how much VRAM each process
holds, and which container it lives in.

It is a single binary that reads the kernel's sysfs directly — no ROCm, no root,
no dependencies — and finishes in well under 100 ms.

```
==================================== What process is using GPU =====================================
=========================================== Concise Info ===========================================
GPU   VRAM used / total      VRAM%   Util%   Procs
--------------------------------------------------
0      183.1GiB / 255.7GiB     71%     11%       3
1      178.7GiB / 255.7GiB     69%    100%       3
2        292MiB / 255.7GiB      0%      0%       0
3        292MiB / 255.7GiB      0%      0%       0
====================================================================================================
========================================== Process Detail ==========================================
[GPU 0]
├── 1261635 sglang::schedul (llm-worker-1)         170.7GiB
│   └── 1286577 python (llm-worker-1)
├── 1294342 sglang::tokeniz (llm-worker-1)           6.5GiB
└── 1294343 sglang::tokeniz (llm-worker-1)           5.6GiB
[GPU 1]
├── 2680433 python                                 128.3GiB
└── 2127172 python3 (qwen35-step600)                 4.5GiB
    ├── 2128967 python3 (qwen35-step600)
    ├── 2128968 sglang::schedul (qwen35-step600)    45.5GiB
    └── 2128969 sglang::detoken (qwen35-step600)
[GPU 2-3] no processes
====================================================================================================
========================================= End of Wpiug Log =========================================
```

## Features

- **Per-GPU summary** — VRAM used / total, VRAM %, utilisation and the number of
  processes on each GPU.
- **Process tree per GPU** — every GPU user with its children, and the VRAM each
  one holds on that GPU.
- **VRAM that adds up** — per-process figures come from both KFD and the
  driver's DRM fdinfo, so memory that KFD alone misses (PyTorch's
  `expandable_segments`, for one) is counted too.
- **Container names** — docker, podman and nerdctl; cgroup v1 and v2 layouts from
  docker, containerd/CRI, CRI-O and Kubernetes.
- **Container view** — `-sc` swaps the tree for a matrix of VRAM per container
  and GPU, with the user behind each container.
- **Command lines on demand** — `-w` fits them to the terminal, `-ww` prints them
  in full.
- **Same GPU numbering as `rocm-smi`** — `wpiug 3` and `rocm-smi`'s `GPU[3]` are
  the same card.
- **Safe to run anywhere** — no ROCm userspace, no root, no crates; calls to the
  container runtime are capped at 5 s, so a wedged daemon can't hang it.

## Requirements

| To run | |
| --- | --- |
| OS | Linux |
| GPU | AMD, driven by `amdgpu` (reads `/sys/class/drm` and `/sys/class/kfd`) |
| ROCm, root | Not required |

| To build | |
| --- | --- |
| Rust | 1.70 or newer |
| Network | Not required — there are no crate dependencies |

## Installation

```bash
cd wpiug
cargo install --path .      # builds in release mode -> ~/.cargo/bin/wpiug
wpiug --version             # -> wpiug 0.2.0
```

Run the same command again to upgrade. For a system-wide install, copying the
binary to other GPU nodes, a fully static build, or uninstalling, see
**[INSTALL.md](INSTALL.md)**.

## Usage

```
wpiug [OPTIONS] [GPU_ID...]
```

```bash
wpiug                 # every GPU
wpiug 0               # only GPU 0
wpiug 0,3             # GPUs 0 and 3 (same as `wpiug 0 3`)
wpiug --depth 5       # expand the process tree deeper
wpiug -w              # add each process's command line
wpiug -sc             # VRAM per container and GPU instead of the tree
wpiug -sp -sc         # both views
watch -c -n1 wpiug --color      # live view, refreshed every second
```

GPU ids are zero-based and follow PCI-address order, the same order `rocm-smi`
uses. Running `wpiug` with no id lists every valid one.

### Modes

The detail part of the report comes in two views. Pick one, or give both flags
to get both sections, tree first.

| Option | Description |
| --- | --- |
| `-sp`, `--show-processes` | A process tree per GPU — see [Process Detail](#process-detail). The default. |
| `-sc`, `--show-containers` | A matrix of VRAM per container and GPU — see [Container Detail](#container-detail). |

### Options

| Option | Description |
| --- | --- |
| `-a`, `--all` | Query every GPU. This is already the default; the flag exists for scripts that want to say so. Can't be combined with GPU ids. |
| `--depth <N>` | How many tree levels to draw from each GPU process down, counting the process itself: `1` shows GPU processes only, the default `3` adds children and grandchildren. Processes using the GPU are always shown, however deep they sit. Minimum `1`; `--depth=N` also works. |
| `-w` | Tree: show each process's command line, cut to fit the terminal. Matrix: widen container names to fill the terminal instead of cutting them at 16 characters. |
| `-ww` | Show command lines and container names in full, never cut, like `ps -ww`. `-w -w` means the same. |
| `--ascii` | Draw with plain ASCII instead of box-drawing characters and `…`, for terminals and logs that can't show them. |
| `--color`, `--no-color` | Force colour on or off. By default colour is on only when stdout is a terminal. |
| `-h`, `--help` | Print usage and exit. |
| `-V`, `--version` | Print the version and exit. |

### Environment

| Variable | Effect |
| --- | --- |
| `NO_COLOR` | Turns colour off when it is decided automatically. `--color` still wins. |
| `COLUMNS` | Terminal width to use when neither stdout nor stderr is a terminal. |

### Exit status

| Code | Meaning |
| --- | --- |
| `0` | Report printed. Also returned when output is cut short by a closed pipe (`wpiug \| head`). |
| `1` | Error — no AMD GPU found, unknown GPU id, or an invalid option. The message goes to stderr. |

## Reading the output

### Concise Info

One row per GPU.

| Column | Meaning |
| --- | --- |
| `GPU` | GPU id, as used on the command line and by `rocm-smi`. |
| `VRAM used / total` | Device-wide VRAM in use and installed. Values under 1 GiB are shown in MiB. |
| `VRAM%` | Used as a percentage of total, rounded down. |
| `Util%` | GPU busy percentage reported by the driver; `N/A` if the driver doesn't expose it. |
| `Procs` | Processes using this GPU — holding VRAM on it or owning a compute queue there. Child processes shown only for context are not counted. |

A few hundred MiB in use on an idle GPU is driver and firmware reserve, not a
process.

### Process Detail

One tree per GPU, rooted at a `[GPU N]` header.

```
[GPU 1]
├── 2680433 python                                 128.3GiB    <- host process
└── 2127172 python3 (qwen35-step600)                 4.5GiB    <- in container "qwen35-step600"
    ├── 2128967 python3 (qwen35-step600)                       <- child, holds no VRAM here
    ├── 2128968 sglang::schedul (qwen35-step600)    45.5GiB    <- child that also uses this GPU
    └── 2128969 sglang::detoken (qwen35-step600)
```

- **Each line** is `<pid> <name> (<container>)`. A process on the host has no
  parentheses, so host vs. container is visible at a glance.
- **The VRAM column** is how much memory that process holds *on this GPU*. It is
  blank for processes that hold none and appear only as part of the tree. See
  [Where per-process VRAM comes from](#where-per-process-vram-comes-from).
- **Order** — top-level processes are sorted by VRAM, largest first. A GPU
  process whose parent (or any ancestor) uses the same GPU is drawn under it
  rather than as a separate tree.
- **`--depth` never hides a GPU process** — it only limits the other processes
  around them. A GPU process nested deeper than the limit is still drawn, along
  with the processes that lead down to it, so every process counted in `Procs`
  appears in the tree.
- **Idle GPUs** get one line instead of a tree, and neighbouring ones share it:
  `[GPU 2-3] no processes`.
- **Names are at most 15 characters** — the kernel cuts them there, which is why
  you see `sglang::schedul` for `sglang::scheduler`. Use `-w` for the full command.
- **A hex id instead of a container name**, such as `(32efcf744e1b)`, means the
  container runtime couldn't be asked — usually no access to the docker socket.
  See [Permissions](#permissions).
- **`?` instead of a name** means the process couldn't be read: it exited during
  the scan, or `/proc` is mounted with `hidepid`.

### Command lines (`-w`, `-ww`)

`-w` adds the command line as the last column, cut to fit the terminal. A cut
command ends in `…` (`...` with `--ascii`):

```
[GPU 3]
└── 3028786 python3 (docker-train-1)        95.1GiB  /opt/venv/bin/python3 -u -m qwen_sft.cli.tr…
    ├── 3030667 python3 (docker-train-1)             /opt/venv/bin/python3 -c from multiprocessi…
    └── 3030668 python3 (docker-train-1)             /opt/venv/bin/python3 -c from multiprocessi…
```

`-ww` prints every command in full. The VRAM column stays in place either way —
only the command runs on — so the numbers can never be pushed off screen.

If the terminal leaves fewer than 20 columns for commands (long container names
on an 80-column terminal, for example), `-w` leaves the column out instead of
showing a useless stub, and prints a note on stderr saying how wide the terminal
needs to be. Widen it, or use `-ww`.

Kernel threads have no command line and show as `[name]`, as in `ps`.

### Container Detail

With `-sc`, the detail part is a matrix: a column per container, a row per GPU,
and the VRAM each container holds on each GPU in the cells.

```
GPU     llm-worker-1   qwen35-step600       host
                root            theho    giangnt
------------------------------------------------
0           182.8GiB                -          -
1                  -          50.0GiB   128.3GiB
2                  -                -          -
------------------------------------------------
Total       182.8GiB          50.0GiB   128.3GiB
```

- **Each column** is one container, headed by its name, with the user owning its
  processes underneath — the host account, as `ps` would show it. A container
  running as root shows `root`; one with processes owned by several users lists
  them all, comma-separated.
- **Processes on the host** are grouped by user, one `host` column per user.
- **Cells** are the sum of that container's per-process VRAM on that GPU, `-`
  where it has nothing there. **Total** adds up each column across the GPUs.
- **Order** — containers first, then host columns, each sorted by total VRAM,
  largest first. Every selected GPU gets a row, idle ones included.
- **Long names** are cut at 16 characters, ending in `…`. `-w` widens them as far
  as the terminal allows; `-ww` never cuts them.
- **Look-alike names stay apart.** Where cutting the end would make two names
  identical — `bench-oneocr0930_10427-g4` and `…-g5` both becoming
  `bench-oneocr093…` — those are cut in the middle instead, keeping the end where
  such names usually differ: `bench-o…10427-g4`, `bench-o…10427-g5`.

Only processes that actually use a GPU are counted, so the matrix adds up to the
same processes the tree shows with a VRAM value. `--depth` has no effect here.

### Width and colour

The report fits the terminal: banners follow its width up to 124 columns, and
`-w` cuts commands to it. The width is read from stdout, then stderr — so
`wpiug -w | less -R` still fits the real terminal — then `COLUMNS`, and falls
back to 124.

On a terminal, fields are colour-coded: pid cyan, process name bold, container
yellow, VRAM green, tree lines and commands dim. In the matrix, users are bold
and empty cells dim. Output to a file or pipe is
plain text. To keep colour through a pager, use `wpiug --color | less -R`.

## Permissions

GPU and process data need **no root and no group membership** — everything
`wpiug` reads is world-readable, and it never opens `/dev/kfd` or `/dev/dri`.
Other users' processes show up normally.

Two things are better with more access:

- **Container names** come from the container runtime and need whatever access
  it requires — usually membership of the `docker` group. Without it, `wpiug`
  shows the short container id instead and still exits 0.
- **Exact VRAM for other users' processes** needs root. The full per-process
  figure comes from DRM fdinfo, which Linux only lets you read for your own
  processes; for anyone else's, `wpiug` falls back to KFD's figure, which can be
  too low. When that leaves VRAM unaccounted for on a GPU, a note on stderr says
  so — see [Where per-process VRAM comes from](#where-per-process-vram-comes-from).

On hosts where `/proc` is mounted with `hidepid=2`, run as root to see other
users' processes. Details are in [INSTALL.md § Permissions](INSTALL.md#4-permissions).

## How it works

Everything comes from sysfs and procfs, read in one pass:

| Data | Source |
| --- | --- |
| GPUs | `/sys/class/drm/card*/device`, vendor `0x1002`: `mem_info_vram_used`, `mem_info_vram_total`, `gpu_busy_percent`. Sorted by PCI address to match `rocm-smi`. |
| GPU processes | `/sys/class/kfd/kfd/proc/<pid>/`: `vram_<gpu_id>` for VRAM, `queues/*/gpuid` for queues. `gpu_id` is mapped to a card through the KFD topology's `drm_render_minor`. |
| Per-process VRAM | The larger of KFD's `vram_<gpu_id>` and the `drm-resident-vram` of the process's render-node clients in `/proc/<pid>/fdinfo` for that card — see below. |
| Process tree | `Name` and `PPid` from `/proc/<pid>/status`. |
| Users | Effective `Uid` from `/proc/<pid>/status`, turned into a login name through NSS (`getpwuid_r`, as `ps` does), so LDAP and SSSD accounts resolve too. An unknown uid shows as the number. |
| Command lines | `/proc/<pid>/cmdline`, read only with `-w`/`-ww` and only for processes that are drawn. |
| Containers | Id parsed from `/proc/<pid>/cgroup`, name looked up with `docker`, `podman` or `nerdctl ps` — once per run, only if a containerised process is present, 5 s timeout each. |

A process counts as using a GPU when it holds VRAM there **or** has a queue on
it. Processes that merely have `/dev/kfd` open — such as dataloader workers that
inherited the file descriptor — are skipped, because they would bury the real
users.

Reading `cmdline` briefly takes the target process's `mmap_lock`, so on a node
full of busy trainers `-w` can add a few hundred milliseconds — the same cost
`ps` pays.

### Where per-process VRAM comes from

KFD keeps a VRAM counter per process, but it only counts memory allocated
through KFD itself. Memory created through the GPU's render node and shared as a
dma-buf never reaches it — and that is exactly what HIP's virtual memory API
does, which PyTorch uses when `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.
A training job can hold 195 GiB of a card while KFD reports under 5.

The amdgpu driver also reports, per open render node, every buffer the client
holds (DRM fdinfo). `wpiug` reads both and shows the larger, with two rules so
nothing is counted twice:

- **Only the GPUs KFD already counts a process on.** A process's render-node
  client on another GPU usually holds a peer's buffers mapped in over IPC — a
  training rank sees slices of every other rank's memory — so those stay with
  their owner.
- **Each client is counted once.** A forked child inherits its parent's open
  render nodes; the memory is credited to the parent.

fdinfo can only be read for your own processes, or anyone's as root. For other
users' processes `wpiug` has KFD's figure alone; if a GPU then shows more than
1 GiB in use that no process accounts for, it prints a note on stderr naming the
GPU and the gap. Run as root to count it.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `no AMD GPUs found under /sys/class/drm` | No AMD GPU, or `amdgpu` isn't loaded — check `lsmod \| grep amdgpu`. |
| Tree characters show as boxes or garbage | Use `--ascii`. |
| Escape codes like `^[[36m` in a file | Something forced colour on; use `--no-color` or `NO_COLOR=1`. |
| Output is very tall | Dataloader workers expand the tree; use `--depth 1` or `--depth 2`. |
| Can't tell same-named processes apart | Use `-w` to show their command lines. |
| GPU has VRAM in use but no processes listed | Driver reserve (a few hundred MiB), or a graphics-only client — see [Limitations](#limitations). |
| `VRAM in use that no process accounts for` on stderr | Other users' processes could only be read through KFD; run as root. See [Where per-process VRAM comes from](#where-per-process-vram-comes-from). |

More cases are covered in [INSTALL.md § Troubleshooting](INSTALL.md#7-troubleshooting).

## Limitations

- **AMD only.** NVIDIA GPUs are not supported.
- **Compute users only.** Processes are found through KFD (ROCm compute); a pure
  graphics client that never touches KFD won't be listed, though its VRAM still
  counts in the Concise Info totals.
- **One snapshot per run.** There is no built-in live mode; use
  `watch -c -n1 wpiug --color`.
- **Buffers shared on the same GPU count for each holder.** Two processes on one
  card mapping the same buffer both show it, so their VRAM can add up to a little
  more than the card's total in use.

## Development

```bash
cargo build --release       # -> target/release/wpiug, about 490 KB
cargo test --release        # tree building, container matrix, fdinfo VRAM
cargo clippy --release
```

| File | Responsibility |
| --- | --- |
| `src/main.rs` | Entry point: gathers data, builds the trees, assembles the report |
| `src/args.rs` | Command-line parsing and `--help` text |
| `src/amd.rs` | GPU and GPU-process discovery from sysfs |
| `src/drm.rs` | Per-process VRAM from DRM fdinfo, credited to the right process |
| `src/procs.rs` | `/proc` snapshot: names, parents, owners, children, command lines |
| `src/container.rs` | Container id from cgroups, name from the container runtime |
| `src/render.rs` | Banners, concise table, process tree, container matrix and column layout |
| `src/util.rs` | Terminal width, user names, padding and truncation helpers, subprocess timeout |
