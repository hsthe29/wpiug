use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone)]
pub struct ProcInfo {
    pub ppid: u32,
    pub name: String,
}

pub struct ProcTable {
    pub by_pid: HashMap<u32, ProcInfo>,
    pub children: HashMap<u32, Vec<u32>>,
}

impl ProcTable {
    pub fn name(&self, pid: u32) -> &str {
        self.by_pid.get(&pid).map_or("?", |p| p.name.as_str())
    }

    pub fn children_of(&self, pid: u32) -> &[u32] {
        self.children.get(&pid).map_or(&[], |v| v.as_slice())
    }

    /// The command line of `pid` the way `ps` prints it: argv joined by spaces,
    /// or the bracketed name for kernel threads, which carry no argv at all.
    /// Read on demand - only the pids we actually draw need it.
    pub fn cmdline(&self, pid: u32) -> String {
        let raw = fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
        // argv arrives NUL-separated; those and any stray control byte would
        // wreck the layout, so they all collapse to spaces.
        let joined: String = String::from_utf8_lossy(&raw)
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        let trimmed = joined.trim();
        if trimmed.is_empty() {
            format!("[{}]", self.name(pid))
        } else {
            trimmed.to_string()
        }
    }

    /// `pid`'s parent, grandparent and so on, stopping short of init, at a
    /// process that exited mid-scan, or after 64 hops - pid reuse during the
    /// scan could in principle stitch the snapshot into a loop.
    pub fn ancestors(&self, pid: u32) -> impl Iterator<Item = u32> + '_ {
        let mut cur = self.by_pid.get(&pid).map(|p| p.ppid);
        std::iter::from_fn(move || {
            let up = cur.filter(|&p| p > 1)?;
            cur = self.by_pid.get(&up).map(|p| p.ppid);
            Some(up)
        })
        .take(64)
    }

    /// Walk up from `pid` looking for the closest ancestor in `set`.
    /// Used to fold a GPU process under its GPU-using parent.
    pub fn nearest_ancestor_in(&self, pid: u32, set: &std::collections::HashSet<u32>) -> Option<u32> {
        self.ancestors(pid).find(|up| set.contains(up))
    }
}

/// Snapshot every process from /proc in one pass.
pub fn snapshot() -> ProcTable {
    let mut by_pid = HashMap::new();
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();

    if let Ok(entries) = fs::read_dir("/proc") {
        for entry in entries.flatten() {
            let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            // Processes come and go while we scan; a missing status is normal.
            let Ok(status) = fs::read_to_string(entry.path().join("status")) else {
                continue;
            };

            let mut name = String::new();
            let mut ppid = 0u32;
            for line in status.lines() {
                if let Some(v) = line.strip_prefix("Name:") {
                    name = v.trim().to_string();
                } else if let Some(v) = line.strip_prefix("PPid:") {
                    ppid = v.trim().parse().unwrap_or(0);
                    break; // PPid always follows Name in /proc/<pid>/status
                }
            }
            if name.is_empty() {
                name = "?".to_string();
            }

            children.entry(ppid).or_default().push(pid);
            by_pid.insert(pid, ProcInfo { ppid, name });
        }
    }

    for kids in children.values_mut() {
        kids.sort_unstable();
    }

    ProcTable { by_pid, children }
}
