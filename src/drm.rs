//! Per-process VRAM from the fdinfo amdgpu keeps for every open render node.
//!
//! KFD's own counter (`vram_<gpu_id>`) only sees memory allocated through KFD.
//! Memory created through the render node and shared as a dma-buf - what HIP's
//! virtual memory API does, and with it PyTorch's `expandable_segments` - never
//! shows up there, so a job can hold 190 GiB while KFD reports 5. fdinfo counts
//! every buffer the client holds, however it was made.
//!
//! Reading another process's fdinfo takes the same access as `ptrace` would:
//! your own processes, or anyone's as root. Where it can't be read, KFD's
//! figure is all there is.

use std::collections::{HashMap, HashSet};
use std::fs;

use crate::amd::{sort_procs, Gpu, GpuProc};
use crate::procs::ProcTable;

/// One open DRM client: the device, its client id, and the VRAM it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Client {
    pub pdev: String,
    pub id: u64,
    pub vram: u64,
}

/// Raise each process's VRAM to what its DRM clients hold on that GPU, and
/// re-sort, since the largest holders may have changed. Returns the pids whose
/// fdinfo could not be read; theirs stays KFD's.
pub fn credit(procs: &mut [GpuProc], gpus: &[Gpu], table: &ProcTable) -> HashSet<u32> {
    let mut held: HashMap<u32, Vec<Client>> = HashMap::new();
    let mut unreadable = HashSet::new();
    for pid in procs.iter().map(|p| p.pid).collect::<HashSet<_>>() {
        match clients(pid) {
            Some(c) => {
                held.insert(pid, c);
            }
            None => {
                unreadable.insert(pid);
            }
        }
    }
    apply(procs, gpus, table, &held);
    sort_procs(procs);
    unreadable
}

/// `credit` minus the reading, so it can be tested.
///
/// Only GPUs KFD already counts a process on are touched: a client on any other
/// GPU usually holds a peer's buffers, mapped in over IPC, not memory of its
/// own. KFD's figure is kept when it is the larger one.
fn apply(procs: &mut [GpuProc], gpus: &[Gpu], table: &ProcTable, held: &HashMap<u32, Vec<Client>>) {
    // A forked child inherits its parent's open files, DRM clients included.
    // Credit each client to its topmost holder so it is counted once.
    let mut holders: HashMap<(&str, u64), Vec<u32>> = HashMap::new();
    for (&pid, cs) in held {
        for c in cs {
            holders.entry((c.pdev.as_str(), c.id)).or_default().push(pid);
        }
    }
    let owner: HashMap<(&str, u64), u32> = holders
        .iter()
        .filter_map(|(&key, hs)| {
            let top = hs
                .iter()
                .copied()
                .filter(|&pid| !table.ancestors(pid).any(|up| hs.contains(&up)))
                .min()
                .or_else(|| hs.iter().copied().min())?;
            Some((key, top))
        })
        .collect();

    let pci: HashMap<u32, &str> = gpus.iter().map(|g| (g.index, g.pci.as_str())).collect();
    for p in procs.iter_mut() {
        let (Some(cs), Some(&pdev)) = (held.get(&p.pid), pci.get(&p.gpu_index)) else {
            continue;
        };
        let drm: u64 = cs
            .iter()
            .filter(|c| c.pdev == pdev && owner.get(&(c.pdev.as_str(), c.id)) == Some(&p.pid))
            .map(|c| c.vram)
            .sum();
        p.vram = p.vram.max(drm);
    }
}

/// The amdgpu clients `pid` has open, or `None` when its file descriptors
/// can't be read.
fn clients(pid: u32) -> Option<Vec<Client>> {
    let fds = fs::read_dir(format!("/proc/{pid}/fd")).ok()?;
    let mut out: Vec<Client> = Vec::new();
    for fd in fds.flatten() {
        let is_drm = fs::read_link(fd.path()).is_ok_and(|t| t.starts_with("/dev/dri/"));
        if !is_drm {
            continue;
        }
        let path = format!("/proc/{pid}/fdinfo/{}", fd.file_name().to_string_lossy());
        let Some(c) = fs::read_to_string(path).ok().as_deref().and_then(parse) else {
            continue;
        };
        // A dup'd descriptor is the same client again.
        if !out.iter().any(|o| o.pdev == c.pdev && o.id == c.id) {
            out.push(c);
        }
    }
    Some(out)
}

/// One fdinfo file, if it belongs to an amdgpu client.
fn parse(info: &str) -> Option<Client> {
    let (mut driver, mut pdev, mut id) = (None, None, None);
    let (mut resident, mut legacy) = (None, None);
    for line in info.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key {
            "drm-driver" => driver = Some(value),
            "drm-pdev" => pdev = Some(value),
            "drm-client-id" => id = value.parse().ok(),
            // The standard key; `drm-memory-vram` is the older name for it.
            "drm-resident-vram" => resident = bytes(value),
            "drm-memory-vram" => legacy = bytes(value),
            _ => {}
        }
    }
    if driver != Some("amdgpu") {
        return None;
    }
    Some(Client {
        pdev: pdev?.to_string(),
        id: id?,
        vram: resident.or(legacy).unwrap_or(0),
    })
}

/// `2019988 KiB`, `1344 MiB` or a bare `0`, in bytes.
fn bytes(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let n: u64 = parts.next()?.parse().ok()?;
    let unit: u64 = match parts.next() {
        None | Some("B") => 1,
        Some("KiB") => 1 << 10,
        Some("MiB") => 1 << 20,
        Some("GiB") => 1 << 30,
        Some(_) => return None,
    };
    n.checked_mul(unit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procs::ProcInfo;

    const GIB: u64 = 1 << 30;

    /// Straight from a ROCm 6.x rank on kernel 6.8.
    const FDINFO: &str = "\
pos:\t0
flags:\t02100002
mnt_id:\t26
ino:\t1102
drm-driver:\tamdgpu
drm-client-id:\t71882
drm-pdev:\t0000:9b:00.0
pasid:\t32803
drm-total-vram:\t2019988 KiB
drm-shared-vram:\t1344 MiB
drm-active-vram:\t0
drm-resident-vram:\t2019988 KiB
drm-purgeable-vram:\t0
drm-memory-vram:\t2019988 KiB
drm-memory-gtt: \t13209624 KiB
amd-evicted-vram:\t0 KiB
";

    #[test]
    fn parses_amdgpu_fdinfo() {
        let want = Client {
            pdev: "0000:9b:00.0".to_string(),
            id: 71882,
            vram: 2019988 * 1024,
        };
        assert_eq!(parse(FDINFO), Some(want));
        // Older kernels only print the legacy key.
        let legacy = "drm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-pdev:\tp\ndrm-memory-vram:\t3 MiB\n";
        assert_eq!(parse(legacy).map(|c| c.vram), Some(3 << 20));
        // Not a GPU client, or not ours.
        assert_eq!(parse("pos:\t0\nflags:\t02\n"), None);
        assert_eq!(parse(&FDINFO.replace("amdgpu", "i915")), None);
    }

    #[test]
    fn parses_sizes() {
        assert_eq!(bytes("0"), Some(0));
        assert_eq!(bytes("2019988 KiB"), Some(2019988 << 10));
        assert_eq!(bytes("1344 MiB"), Some(1344 << 20));
        assert_eq!(bytes("lots"), None);
    }

    fn gpus() -> Vec<Gpu> {
        (0..2)
            .map(|i| Gpu {
                index: i,
                pci: format!("pci{i}"),
                gpu_id: Some(100 + i),
                vram_used: 0,
                vram_total: 0,
                busy: None,
            })
            .collect()
    }

    fn table(procs: &[(u32, u32)]) -> ProcTable {
        let mut t = ProcTable {
            by_pid: HashMap::new(),
            children: HashMap::new(),
        };
        for &(pid, ppid) in procs {
            let name = format!("p{pid}");
            t.by_pid.insert(pid, ProcInfo { ppid, name, uid: None });
            t.children.entry(ppid).or_default().push(pid);
        }
        t
    }

    fn client(gpu: u32, id: u64, gib: u64) -> Client {
        Client {
            pdev: format!("pci{gpu}"),
            id,
            vram: gib * GIB,
        }
    }

    fn proc(gpu_index: u32, pid: u32, gib: u64) -> GpuProc {
        GpuProc {
            gpu_index,
            pid,
            vram: gib * GIB,
        }
    }

    fn vram_after(procs: &mut [GpuProc], t: &ProcTable, held: &[(u32, Vec<Client>)]) -> Vec<u64> {
        apply(procs, &gpus(), t, &held.iter().cloned().collect());
        procs.iter().map(|p| p.vram / GIB).collect()
    }

    /// The case that started it: KFD says 4, the render node holds 195.
    #[test]
    fn fdinfo_raises_kfd_figure() {
        let mut procs = [proc(0, 10, 4)];
        let held = [(10, vec![client(0, 1, 195)])];
        assert_eq!(vram_after(&mut procs, &table(&[(10, 1)]), &held), [195]);
    }

    #[test]
    fn kfd_kept_when_larger() {
        let mut procs = [proc(0, 10, 88)];
        let held = [(10, vec![client(0, 1, 2)])];
        assert_eq!(vram_after(&mut procs, &table(&[(10, 1)]), &held), [88]);
    }

    /// A rank holds a peer's buffers on the peer's GPU; they stay the peer's.
    #[test]
    fn other_gpus_clients_ignored() {
        let mut procs = [proc(0, 10, 4)];
        let held = [(10, vec![client(0, 1, 50), client(1, 2, 2)])];
        assert_eq!(vram_after(&mut procs, &table(&[(10, 1)]), &held), [50]);
    }

    /// A child that inherited its parent's client doesn't count it again.
    #[test]
    fn inherited_client_counted_once() {
        let mut procs = [proc(0, 10, 1), proc(0, 11, 1)];
        let held = [
            (10, vec![client(0, 1, 100)]),
            (11, vec![client(0, 1, 100), client(0, 2, 7)]),
        ];
        let t = table(&[(10, 1), (11, 10)]);
        assert_eq!(vram_after(&mut procs, &t, &held), [100, 7]);
    }

    /// No fdinfo (another user's process): KFD's figure stands.
    #[test]
    fn unreadable_process_keeps_kfd() {
        let mut procs = [proc(0, 10, 4)];
        assert_eq!(vram_after(&mut procs, &table(&[(10, 1)]), &[]), [4]);
    }
}
