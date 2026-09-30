use std::collections::HashMap;
use std::fs;
use std::time::Duration;

use crate::util::run;

const TIMEOUT: Duration = Duration::from_secs(5);

/// Resolves a pid to the name of the container it runs in, if any.
pub struct Resolver {
    /// Full container id -> human name. Built lazily on first containerized pid.
    names: Option<HashMap<String, String>>,
}

impl Resolver {
    pub fn new() -> Self {
        Resolver { names: None }
    }

    pub fn lookup(&mut self, pid: u32) -> Option<String> {
        let id = container_id(pid)?;
        if self.names.is_none() {
            self.names = Some(build_name_map());
        }
        let names = self.names.as_ref().unwrap();
        // cgroup ids are full 64-hex; runtimes may report either length.
        if let Some(name) = names.get(&id) {
            return Some(name.clone());
        }
        let short: String = id.chars().take(12).collect();
        Some(names.get(&short).cloned().unwrap_or(short))
    }
}

/// Pull a container id out of /proc/<pid>/cgroup.
///
/// Handles cgroup v2 (`0::/system.slice/docker-<id>.scope`) and v1, across
/// docker, containerd/CRI, CRI-O, podman and kubepods layouts.
fn container_id(pid: u32) -> Option<String> {
    let raw = fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?;

    for line in raw.lines() {
        // v1: "12:memory:/docker/<id>"; v2: "0::/system.slice/docker-<id>.scope"
        let path = line.rsplit(':').next().unwrap_or("");
        for seg in path.rsplit('/') {
            let seg = seg.strip_suffix(".scope").unwrap_or(seg);
            let seg = seg
                .strip_prefix("docker-")
                .or_else(|| seg.strip_prefix("cri-containerd-"))
                .or_else(|| seg.strip_prefix("crio-"))
                .or_else(|| seg.strip_prefix("libpod-"))
                .or_else(|| seg.strip_prefix("containerd-"))
                .unwrap_or(seg);
            if is_container_id(seg) {
                return Some(seg.to_string());
            }
        }
    }
    None
}

fn is_container_id(s: &str) -> bool {
    s.len() >= 12 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Ask whatever runtimes are installed for id -> name. Failures are silent:
/// a missing or wedged daemon just means we fall back to the short id.
fn build_name_map() -> HashMap<String, String> {
    let mut map = HashMap::new();

    let sources: [(&str, &[&str]); 3] = [
        (
            "docker",
            &["ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}"],
        ),
        (
            "podman",
            &["ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}"],
        ),
        ("nerdctl", &["ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}"]),
    ];

    for (bin, args) in sources {
        let Ok(out) = run(bin, args, TIMEOUT) else {
            continue;
        };
        for line in out.lines() {
            let mut it = line.split_whitespace();
            if let (Some(id), Some(name)) = (it.next(), it.next()) {
                map.insert(id.to_string(), name.to_string());
            }
        }
    }

    map
}
