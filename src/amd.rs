//! AMD GPU discovery straight from sysfs.
//!
//! Two sources, no ROCm userspace required:
//!   * `/sys/class/drm/card*/device` - amdgpu device totals (VRAM, busy %).
//!   * `/sys/class/kfd/kfd/proc/<pid>` - per-process VRAM and queues, keyed by
//!     the KFD `gpu_id`, which we tie back to a card through the topology
//!     node's `drm_render_minor`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

const DRM: &str = "/sys/class/drm";
const KFD: &str = "/sys/class/kfd/kfd";
const AMD_VENDOR: &str = "0x1002";

#[derive(Debug, Clone)]
pub struct Gpu {
    /// Logical index, ordered by PCI address - the same numbering rocm-smi uses.
    pub index: u32,
    /// KFD gpu_id, absent if the card has no KFD topology node.
    pub gpu_id: Option<u32>,
    pub vram_used: u64,
    pub vram_total: u64,
    pub busy: Option<u32>,
}

impl Gpu {
    /// Truncated, not rounded, so the number matches `rocm-smi --showmemuse`.
    pub fn vram_percent(&self) -> u64 {
        if self.vram_total == 0 {
            0
        } else {
            self.vram_used.saturating_mul(100) / self.vram_total
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuProc {
    pub gpu_index: u32,
    pub pid: u32,
    pub vram: u64,
}

fn read_trim(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn read_u64(path: impl AsRef<Path>) -> Option<u64> {
    read_trim(path).and_then(|s| s.parse().ok())
}

/// Enumerate AMD cards, ordered by PCI address so `index` matches rocm-smi.
pub fn discover_gpus() -> Result<Vec<Gpu>, String> {
    let entries = fs::read_dir(DRM).map_err(|e| format!("cannot read {DRM}: {e}"))?;

    // (pci, render_minor, vram_used, vram_total, busy)
    let mut found: Vec<(String, Option<u32>, u64, u64, Option<u32>)> = Vec::new();

    for entry in entries.flatten() {
        let card = entry.file_name().to_string_lossy().into_owned();
        // Only `cardN`; `cardN-DP-1` and friends are connectors, not devices.
        let Some(num) = card.strip_prefix("card") else {
            continue;
        };
        if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }

        let dev = entry.path().join("device");
        if read_trim(dev.join("vendor")).as_deref() != Some(AMD_VENDOR) {
            continue;
        }
        // Display-only or virtual devices expose no VRAM totals.
        let Some(vram_total) = read_u64(dev.join("mem_info_vram_total")) else {
            continue;
        };

        let pci = fs::read_link(&dev)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| card.clone());

        found.push((
            pci,
            render_minor(&dev),
            read_u64(dev.join("mem_info_vram_used")).unwrap_or(0),
            vram_total,
            read_u64(dev.join("gpu_busy_percent")).map(|v| v as u32),
        ));
    }

    if found.is_empty() {
        return Err(format!(
            "no AMD GPUs found under {DRM} (vendor {AMD_VENDOR} with mem_info_vram_total)"
        ));
    }

    found.sort_by(|a, b| a.0.cmp(&b.0));

    let minor_to_gpu_id = topology_render_minor_map();
    let gpus = found
        .into_iter()
        .enumerate()
        .map(|(i, (_pci, minor, vram_used, vram_total, busy))| Gpu {
            index: i as u32,
            gpu_id: minor.and_then(|m| minor_to_gpu_id.get(&m).copied()),
            vram_used,
            vram_total,
            busy,
        })
        .collect();

    Ok(gpus)
}

/// `/sys/class/drm/cardN/device/drm/renderD129` -> 129
fn render_minor(dev: &Path) -> Option<u32> {
    for e in fs::read_dir(dev.join("drm")).ok()?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if let Some(n) = name.strip_prefix("renderD") {
            if let Ok(v) = n.parse() {
                return Some(v);
            }
        }
    }
    None
}

/// KFD topology: `drm_render_minor` (in `properties`) -> `gpu_id`.
fn topology_render_minor_map() -> HashMap<u32, u32> {
    let mut map = HashMap::new();
    let Ok(nodes) = fs::read_dir(format!("{KFD}/topology/nodes")) else {
        return map;
    };
    for node in nodes.flatten() {
        let gpu_id = match read_u64(node.path().join("gpu_id")) {
            // gpu_id 0 marks a CPU node.
            Some(0) | None => continue,
            Some(v) => v as u32,
        };
        let Ok(props) = fs::read_to_string(node.path().join("properties")) else {
            continue;
        };
        for line in props.lines() {
            if let Some(v) = line.strip_prefix("drm_render_minor ") {
                if let Ok(minor) = v.trim().parse::<u32>() {
                    map.insert(minor, gpu_id);
                }
                break;
            }
        }
    }
    map
}

/// Processes holding KFD resources, attributed per GPU.
///
/// A process counts as using a GPU when it holds VRAM there or has a queue on
/// it. Plain attachment with neither (idle dataloader workers that inherited
/// the KFD fd) is ignored - it would bury the real users in noise.
pub fn discover_procs(gpus: &[Gpu]) -> Vec<GpuProc> {
    let by_gpu_id: HashMap<u32, u32> = gpus
        .iter()
        .filter_map(|g| g.gpu_id.map(|id| (id, g.index)))
        .collect();

    let mut out = Vec::new();
    let Ok(procs) = fs::read_dir(format!("{KFD}/proc")) else {
        return out;
    };

    for entry in procs.flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        let dir = entry.path();

        let mut queues: HashMap<u32, usize> = HashMap::new();
        if let Ok(qs) = fs::read_dir(dir.join("queues")) {
            for q in qs.flatten() {
                if let Some(gid) = read_u64(q.path().join("gpuid")) {
                    *queues.entry(gid as u32).or_default() += 1;
                }
            }
        }

        for (&gpu_id, &gpu_index) in &by_gpu_id {
            let vram = read_u64(dir.join(format!("vram_{gpu_id}"))).unwrap_or(0);
            let queues = queues.get(&gpu_id).copied().unwrap_or(0);
            if vram == 0 && queues == 0 {
                continue;
            }
            out.push(GpuProc {
                gpu_index,
                pid,
                vram,
            });
        }
    }

    out.sort_by(|a, b| {
        a.gpu_index
            .cmp(&b.gpu_index)
            .then(b.vram.cmp(&a.vram))
            .then(a.pid.cmp(&b.pid))
    });
    out
}
