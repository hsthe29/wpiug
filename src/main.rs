mod amd;
mod args;
mod container;
mod drm;
mod procs;
mod render;
mod util;

use std::collections::{HashMap, HashSet};
use std::process::ExitCode;

use render::Node;

fn main() -> ExitCode {
    match args::parse(std::env::args().skip(1)) {
        Ok(args::Parsed::Help) => {
            print!("{}", args::HELP);
            ExitCode::SUCCESS
        }
        Ok(args::Parsed::Version) => {
            println!("wpiug {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(args::Parsed::Run(a)) => match run(&a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("wpiug: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("wpiug: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(a: &args::Args) -> Result<(), String> {
    let all_gpus = amd::discover_gpus()?;

    let selected: Vec<&amd::Gpu> = match &a.only {
        None => all_gpus.iter().collect(),
        Some(ids) => {
            let mut out = Vec::new();
            for id in ids {
                match all_gpus.iter().find(|g| g.index == *id) {
                    Some(g) => out.push(g),
                    None => {
                        return Err(format!(
                            "no GPU with id {id}; this machine has {} (0-{})",
                            all_gpus.len(),
                            all_gpus.len() - 1
                        ))
                    }
                }
            }
            out
        }
    };

    // Attribution needs every gpu_id, so scan against the full device list and
    // narrow afterwards.
    let mut gpu_procs = amd::discover_procs(&all_gpus);
    let table = procs::snapshot();
    // KFD misses some of what a process holds; DRM fdinfo, where we may read
    // it, has the full figure.
    let unreadable = drm::credit(&mut gpu_procs, &all_gpus, &table);
    let mut resolver = container::Resolver::new();

    let mut by_gpu: HashMap<u32, Vec<&amd::GpuProc>> = HashMap::new();
    for p in &gpu_procs {
        by_gpu.entry(p.gpu_index).or_default().push(p);
    }

    let glyphs = if a.ascii {
        &render::ASCII
    } else {
        &render::UNICODE
    };
    let color = a.color.unwrap_or_else(|| {
        use std::io::IsTerminal;
        std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
    });
    let term = util::term_width().unwrap_or(render::WIDTH);

    // Each section gathers only what it draws: command lines for the tree
    // under `-w`, process owners for the matrix.
    let mut sections: Vec<(&str, Vec<String>)> = Vec::new();
    let mut notes = Vec::new();
    if a.show_processes {
        let (lines, squeezed) = process_detail(
            a, &selected, &by_gpu, &table, &mut resolver, glyphs, color, term,
        );
        sections.push(("Process Detail", lines));
        notes.extend(squeezed);
    }
    if a.show_containers {
        let lines = container_detail(
            a, &selected, &by_gpu, &table, &mut resolver, glyphs, color, term,
        );
        sections.push(("Container Detail", lines));
    }

    notes.extend(unaccounted_note(&selected, &by_gpu, &unreadable));

    let width = term.min(render::WIDTH);
    emit(&render_report(&selected, &by_gpu, &sections, color, width))?;
    // After the report rather than before it, so they are the last thing on
    // screen instead of scrolling away above the output.
    for note in notes {
        eprintln!("wpiug: {note}");
    }
    Ok(())
}

/// A gap this small between a GPU's VRAM in use and what its processes hold
/// is driver reserve and rounding, not anything worth a note.
const UNACCOUNTED: u64 = 1 << 30;

/// A note for stderr when a GPU has VRAM in use that its processes don't
/// account for, and some of them could only be read through KFD - the likely
/// reason, and one running as root fixes.
fn unaccounted_note(
    selected: &[&amd::Gpu],
    by_gpu: &HashMap<u32, Vec<&amd::GpuProc>>,
    unreadable: &HashSet<u32>,
) -> Option<String> {
    let gaps: Vec<String> = selected
        .iter()
        .filter_map(|g| {
            let procs = by_gpu.get(&g.index)?;
            if !procs.iter().any(|p| unreadable.contains(&p.pid)) {
                return None;
            }
            let held: u64 = procs.iter().map(|p| p.vram).sum();
            let gap = g.vram_used.saturating_sub(held);
            (gap > UNACCOUNTED).then(|| format!("GPU {} {}", g.index, render::human_bytes(gap)))
        })
        .collect();
    if gaps.is_empty() {
        return None;
    }
    Some(format!(
        "VRAM in use that no process accounts for: {}. Other users' processes can \
         only be read through KFD, which misses some allocations (PyTorch's \
         expandable_segments, for one) - run as root to count them",
        gaps.join(", ")
    ))
}

/// Write in one shot. A closed pipe (`wpiug --all | head`) is a normal exit,
/// not a panic - Rust ignores SIGPIPE, so we have to handle it ourselves.
fn emit(text: &str) -> Result<(), String> {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    match out.write_all(text.as_bytes()).and_then(|_| out.flush()) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => Err(format!("failed writing to stdout: {e}")),
    }
}

/// The process trees for one GPU, given the pids using it. Every pid drawn is
/// added to `labelled`.
///
/// Every GPU process is drawn, however deep it sits; `depth` only limits the
/// other processes around them.
fn forest(
    pids: &[u32],
    depth: usize,
    table: &procs::ProcTable,
    labelled: &mut HashSet<u32>,
) -> Vec<Node> {
    let gpu: HashSet<u32> = pids.iter().copied().collect();

    // A GPU process whose ancestor also uses this GPU hangs off that ancestor
    // instead of starting its own tree. The processes between the two are
    // drawn even past `depth` - cut them and the lower one would vanish from
    // the report while still counting in the Procs column.
    let mut roots = Vec::new();
    let mut links = HashSet::new();
    for &pid in pids {
        if table.nearest_ancestor_in(pid, &gpu).is_some() {
            links.extend(table.ancestors(pid).take_while(|up| !gpu.contains(up)));
        } else {
            roots.push(pid);
        }
    }

    let expand = Expand {
        table,
        depth,
        gpu: &gpu,
        links: &links,
    };
    let mut drawn = HashSet::new();
    let nodes = roots
        .iter()
        .map(|&pid| expand.build(pid, depth, &mut drawn))
        .collect();
    labelled.extend(drawn);
    nodes
}

/// Decides which processes make it into one GPU's trees.
struct Expand<'a> {
    table: &'a procs::ProcTable,
    /// Levels drawn from each GPU process down, counting the process itself.
    depth: usize,
    /// Processes using this GPU.
    gpu: &'a HashSet<u32>,
    /// Processes leading from a GPU process down to another one.
    links: &'a HashSet<u32>,
}

impl Expand<'_> {
    /// Expand `pid` with `left` levels to go (1 = `pid` alone). A GPU process
    /// starts over with the full depth, so each one gets the same context
    /// wherever it sits. `drawn` keeps a looped snapshot from recursing forever.
    fn build(&self, pid: u32, left: usize, drawn: &mut HashSet<u32>) -> Node {
        drawn.insert(pid);
        let mut children = Vec::new();
        for &child in self.table.children_of(pid) {
            if drawn.contains(&child) {
                continue;
            }
            let left = if self.gpu.contains(&child) {
                self.depth
            } else if left > 1 {
                left - 1
            } else if self.links.contains(&child) {
                // Past the depth, but a GPU process lies below: draw just the
                // path down to it.
                1
            } else {
                continue;
            };
            children.push(self.build(child, left, drawn));
        }
        Node { pid, children }
    }
}

/// The Process Detail section: a tree per GPU. Also returns a note for stderr
/// when `-w` had no room to show commands.
#[allow(clippy::too_many_arguments)]
fn process_detail(
    a: &args::Args,
    selected: &[&amd::Gpu],
    by_gpu: &HashMap<u32, Vec<&amd::GpuProc>>,
    table: &procs::ProcTable,
    resolver: &mut container::Resolver,
    glyphs: &render::Glyphs,
    color: bool,
    term: usize,
) -> (Vec<String>, Option<String>) {
    // Build the trees first: every pid they touch needs a label.
    let mut trees: Vec<(u32, Vec<Node>)> = Vec::new();
    let mut labelled: HashSet<u32> = HashSet::new();
    for gpu in selected {
        let pids: Vec<u32> = by_gpu
            .get(&gpu.index)
            .map(|v| v.iter().map(|p| p.pid).collect())
            .unwrap_or_default();
        let nodes = forest(&pids, a.depth, table, &mut labelled);
        trees.push((gpu.index, nodes));
    }

    let mut labels: HashMap<u32, render::Label> = HashMap::new();
    for pid in labelled {
        labels.insert(
            pid,
            render::Label {
                pid,
                name: table.name(pid).to_string(),
                container: resolver.lookup(pid),
                cmd: match a.wide {
                    args::Wide::Off => None,
                    _ => Some(table.cmdline(pid)),
                },
            },
        );
    }

    // Build every GPU's rows before emitting any, so the VRAM and command
    // columns can be sized once across the whole section rather than per block.
    let mut blocks: Vec<render::Block> = Vec::new();
    for (index, nodes) in &trees {
        if nodes.is_empty() {
            // Neighbouring GPUs with nothing on them share one line.
            match blocks.last_mut() {
                Some(render::Block::Idle(ids)) => ids.push(*index),
                _ => blocks.push(render::Block::Idle(vec![*index])),
            }
            continue;
        }
        // Per-process VRAM on *this* GPU; a pid can hold memory on several.
        let vram: HashMap<u32, u64> = by_gpu
            .get(index)
            .map(|ps| ps.iter().map(|p| (p.pid, p.vram)).collect())
            .unwrap_or_default();
        let header = format!("[GPU {index}]");
        blocks.push(render::Block::Tree(render::forest_rows(
            &header, nodes, &labels, &vram, glyphs, color,
        )));
    }

    // `-w` cuts commands to the terminal; `-ww` lets them run to full length.
    let cmd_limit = match a.wide {
        args::Wide::Full => None,
        _ => Some(term),
    };
    let col = render::label_column(&blocks);
    let cmd_col = render::cmd_column(&blocks, col, cmd_limit);
    let mut lines = Vec::new();
    for block in &blocks {
        match block {
            render::Block::Tree(rows) => {
                lines.extend(render::align(rows, col, &cmd_col, glyphs, color));
            }
            render::Block::Idle(ids) => lines.push(render::idle_line(ids, color)),
        }
    }

    let note = match cmd_col {
        render::CmdCol::Squeezed { need } => Some(format!(
            "commands hidden: -w needs about {need} columns here, the terminal has {term} \
             (widen it, or use -ww to print them in full)"
        )),
        _ => None,
    };
    (lines, note)
}

/// The Container Detail section: VRAM per container (columns) and GPU (rows).
#[allow(clippy::too_many_arguments)]
fn container_detail(
    a: &args::Args,
    selected: &[&amd::Gpu],
    by_gpu: &HashMap<u32, Vec<&amd::GpuProc>>,
    table: &procs::ProcTable,
    resolver: &mut container::Resolver,
    glyphs: &render::Glyphs,
    color: bool,
    term: usize,
) -> Vec<String> {
    let gpus: Vec<u32> = selected.iter().map(|g| g.index).collect();

    // Only GPU processes count here - their children hold no VRAM.
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut holders = Vec::new();
    for p in gpus.iter().filter_map(|i| by_gpu.get(i)).flatten() {
        let user = match table.uid(p.pid) {
            Some(uid) => names
                .entry(uid)
                .or_insert_with(|| util::user_name(uid).unwrap_or_else(|| uid.to_string()))
                .clone(),
            // Exited mid-scan, or hidden by `hidepid`.
            None => "?".to_string(),
        };
        holders.push(Holder {
            gpu: p.gpu_index,
            vram: p.vram,
            container: resolver.lookup(p.pid),
            user,
        });
    }

    let cols = columns(&gpus, &holders);
    if cols.is_empty() {
        return vec![render::idle_line(&gpus, color)];
    }
    let names = match a.wide {
        args::Wide::Off => render::NameWidth::Cap,
        args::Wide::Fit => render::NameWidth::Fit(term),
        args::Wide::Full => render::NameWidth::Full,
    };
    render::container_matrix(&gpus, &cols, &names, glyphs, color)
}

/// One GPU process, as the container matrix sees it.
struct Holder {
    gpu: u32,
    vram: u64,
    /// `None` for a process running on the host.
    container: Option<String>,
    user: String,
}

/// Group `holders` into matrix columns, one per container and one per user for
/// processes on the host, with a row per GPU in `gpus`. Containers come first,
/// then the host users, each ordered by total VRAM, largest first.
fn columns(gpus: &[u32], holders: &[Holder]) -> Vec<render::Column> {
    let mut cols: Vec<render::Column> = Vec::new();
    let mut index: HashMap<(Option<&str>, Option<&str>), usize> = HashMap::new();
    for h in holders {
        let Some(row) = gpus.iter().position(|g| *g == h.gpu) else {
            continue;
        };
        let key = match &h.container {
            Some(c) => (Some(c.as_str()), None),
            None => (None, Some(h.user.as_str())),
        };
        let i = *index.entry(key).or_insert_with(|| {
            cols.push(render::Column {
                container: h.container.clone(),
                users: Vec::new(),
                vram: vec![None; gpus.len()],
            });
            cols.len() - 1
        });
        let col = &mut cols[i];
        if !col.users.contains(&h.user) {
            col.users.push(h.user.clone());
        }
        *col.vram[row].get_or_insert(0) += h.vram;
    }

    cols.sort_by(|a, b| {
        a.container
            .is_none()
            .cmp(&b.container.is_none())
            .then(b.total().cmp(&a.total()))
            .then(a.container.cmp(&b.container))
            .then(a.users.cmp(&b.users))
    });
    cols
}

/// The whole report: the Concise Info table, then each detail section under
/// its own banner.
///
/// `width` is how wide the banners are drawn: the terminal's, capped at
/// `render::WIDTH`, so they never wrap on a narrow one.
fn render_report(
    selected: &[&amd::Gpu],
    by_gpu: &HashMap<u32, Vec<&amd::GpuProc>>,
    sections: &[(&str, Vec<String>)],
    color: bool,
    width: usize,
) -> String {
    use std::fmt::Write;
    let mut s = String::new();

    let _ = writeln!(s, "{}", render::banner("What process is using GPU", width));
    let _ = writeln!(s, "{}", render::banner("Concise Info", width));

    let entries: Vec<(&amd::Gpu, usize)> = selected
        .iter()
        .map(|g| (*g, by_gpu.get(&g.index).map_or(0, |v| v.len())))
        .collect();
    for line in render::concise_table(&entries, color) {
        let _ = writeln!(s, "{line}");
    }
    let _ = writeln!(s, "{}", render::rule(width));

    for (title, lines) in sections {
        let _ = writeln!(s, "{}", render::banner(title, width));
        for line in lines {
            let _ = writeln!(s, "{line}");
        }
        let _ = writeln!(s, "{}", render::rule(width));
    }
    let _ = writeln!(s, "{}", render::banner("End of Wpiug Log", width));

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A process table from `(pid, ppid)` pairs, children in the order given.
    fn table(procs: &[(u32, u32)]) -> procs::ProcTable {
        let mut by_pid = HashMap::new();
        let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(pid, ppid) in procs {
            let name = format!("p{pid}");
            by_pid.insert(pid, procs::ProcInfo { ppid, name, uid: None });
            children.entry(ppid).or_default().push(pid);
        }
        procs::ProcTable { by_pid, children }
    }

    /// `10(11,12(13))`: each pid followed by its children in brackets.
    fn shape(nodes: &[Node]) -> String {
        nodes
            .iter()
            .map(|n| match n.children.as_slice() {
                [] => n.pid.to_string(),
                kids => format!("{}({})", n.pid, shape(kids)),
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn draw(procs: &[(u32, u32)], gpu: &[u32], depth: usize) -> String {
        shape(&forest(gpu, depth, &table(procs), &mut HashSet::new()))
    }

    #[test]
    fn depth_limits_plain_children() {
        let procs = [(10, 1), (11, 10), (12, 11), (13, 12)];
        assert_eq!(draw(&procs, &[10], 1), "10");
        assert_eq!(draw(&procs, &[10], 3), "10(11(12))");
    }

    /// The README's GPU 1: 2128968 holds VRAM under 2127172, which does too.
    /// `--depth 1` used to drop it from the tree while Procs still said 3.
    #[test]
    fn nested_gpu_child_survives_depth_1() {
        let procs = [
            (2680433, 1),
            (2127172, 1),
            (2128967, 2127172),
            (2128968, 2127172),
            (2128969, 2127172),
        ];
        let gpu = [2680433, 2127172, 2128968];
        assert_eq!(draw(&procs, &gpu, 1), "2680433,2127172(2128968)");
        assert_eq!(
            draw(&procs, &gpu, 2),
            "2680433,2127172(2128967,2128968,2128969)"
        );
    }

    /// A GPU process below the depth limit is reached through the processes
    /// in between - and only through those, not their other children.
    #[test]
    fn deep_gpu_process_keeps_only_its_path() {
        // 10 -> 11 -> 12 -> 13(gpu); 11 and 12 also have plain children.
        let procs = [(10, 1), (11, 10), (14, 10), (12, 11), (15, 11), (13, 12)];
        assert_eq!(draw(&procs, &[10, 13], 1), "10(11(12(13)))");
        assert_eq!(draw(&procs, &[10, 13], 2), "10(11(12(13)),14)");
    }

    /// Each GPU process gets the full depth below it, wherever it sits.
    #[test]
    fn gpu_process_resets_depth() {
        let procs = [(10, 1), (20, 10), (21, 20), (22, 21)];
        assert_eq!(draw(&procs, &[10, 20], 2), "10(20(21))");
    }

    #[test]
    fn unrelated_gpu_processes_are_separate_trees() {
        let procs = [(10, 1), (20, 1), (21, 20)];
        assert_eq!(draw(&procs, &[10, 20], 2), "10,20(21)");
    }

    /// Every drawn pid, links included, needs a label.
    #[test]
    fn links_are_labelled() {
        let procs = [(10, 1), (11, 10), (12, 11), (13, 12)];
        let mut labelled = HashSet::new();
        forest(&[10, 13], 1, &table(&procs), &mut labelled);
        let mut got: Vec<u32> = labelled.into_iter().collect();
        got.sort_unstable();
        assert_eq!(got, [10, 11, 12, 13]);
    }

    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    fn holder(gpu: u32, gib: f64, container: Option<&str>, user: &str) -> Holder {
        Holder {
            gpu,
            vram: (gib * GIB) as u64,
            container: container.map(str::to_string),
            user: user.to_string(),
        }
    }

    /// The README's sample node, as the matrix sees it.
    fn sample() -> Vec<Holder> {
        vec![
            holder(0, 170.7, Some("llm-worker-1"), "root"),
            holder(0, 6.5, Some("llm-worker-1"), "root"),
            holder(0, 5.6, Some("llm-worker-1"), "root"),
            holder(1, 128.3, None, "giangnt"),
            holder(1, 4.5, Some("qwen35-step600"), "theho"),
            holder(1, 45.5, Some("qwen35-step600"), "theho"),
        ]
    }

    fn matrix(holders: &[Holder], names: render::NameWidth, g: &render::Glyphs) -> String {
        let gpus = [0, 1, 2];
        render::container_matrix(&gpus, &columns(&gpus, holders), &names, g, false).join("\n")
    }

    #[test]
    fn container_matrix_layout() {
        let want = "\
GPU     llm-worker-1   qwen35-step600       host
                root            theho    giangnt
------------------------------------------------
0           182.8GiB                -          -
1                  -          50.0GiB   128.3GiB
2                  -                -          -
------------------------------------------------
Total       182.8GiB          50.0GiB   128.3GiB";
        assert_eq!(matrix(&sample(), render::NameWidth::Cap, &render::UNICODE), want);
    }

    /// Containers first, then host users; each by VRAM, largest first. Host
    /// processes split by user, a container's users share its column.
    #[test]
    fn columns_group_and_order() {
        let holders = [
            holder(0, 1.0, None, "alice"),
            holder(0, 9.0, None, "bob"),
            holder(0, 2.0, Some("small"), "root"),
            holder(1, 3.0, Some("big"), "root"),
            holder(1, 4.0, Some("big"), "carol"),
            holder(7, 99.0, Some("not-selected"), "root"),
        ];
        let cols = columns(&[0, 1], &holders);
        let got: Vec<(Option<&str>, String)> = cols
            .iter()
            .map(|c| (c.container.as_deref(), c.users.join(",")))
            .collect();
        assert_eq!(
            got,
            [
                (Some("big"), "root,carol".to_string()),
                (Some("small"), "root".to_string()),
                (None, "bob".to_string()),
                (None, "alice".to_string()),
            ]
        );
        // `big` has nothing on GPU 0 and 7 GiB on GPU 1.
        assert_eq!(cols[0].vram[0], None);
        assert_eq!(cols[0].total(), (3.0 * GIB) as u64 + (4.0 * GIB) as u64);
    }

    #[test]
    fn container_names_cut_by_width_flags() {
        let long = "docker-test-run-357ca3406d66";
        let holders = [holder(0, 1.2, Some(long), "theho")];
        let header = |names, g| matrix(&holders, names, g).lines().next().unwrap().to_string();

        // Default: 16 columns, ending in the glyph set's marker.
        assert!(header(render::NameWidth::Cap, &render::UNICODE).ends_with(" docker-test-run…"));
        assert!(header(render::NameWidth::Cap, &render::ASCII).ends_with(" docker-test-r..."));
        // -ww: whole.
        assert!(header(render::NameWidth::Full, &render::UNICODE).ends_with(&format!(" {long}")));
        // -w: as much as the terminal allows, but never below the default.
        assert!(header(render::NameWidth::Fit(200), &render::UNICODE).ends_with(&format!(" {long}")));
        assert!(header(render::NameWidth::Fit(30), &render::UNICODE).ends_with(" docker-test-run-357ca…"));
        assert!(header(render::NameWidth::Fit(10), &render::UNICODE).ends_with(" docker-test-run…"));
    }

    #[test]
    fn user_names_resolve() {
        assert_eq!(util::user_name(0).as_deref(), Some("root"));
        // No account should ever own this one.
        assert_eq!(util::user_name(4_000_000_000), None);
    }

    fn gpu(index: u32, used_gib: f64) -> amd::Gpu {
        amd::Gpu {
            index,
            pci: format!("pci{index}"),
            gpu_id: None,
            vram_used: (used_gib * GIB) as u64,
            vram_total: 0,
            busy: None,
        }
    }

    /// Speak up only where VRAM is missing *and* a process was out of reach.
    #[test]
    fn unaccounted_note_needs_a_gap_and_an_unreadable_process() {
        let (g0, g1) = (gpu(0, 100.0), gpu(1, 50.0));
        let p0 = amd::GpuProc { gpu_index: 0, pid: 10, vram: (5.0 * GIB) as u64 };
        let p1 = amd::GpuProc { gpu_index: 1, pid: 20, vram: (49.6 * GIB) as u64 };
        let selected = [&g0, &g1];
        let by_gpu: HashMap<u32, Vec<&amd::GpuProc>> = [(0, vec![&p0]), (1, vec![&p1])].into();

        let note = unaccounted_note(&selected, &by_gpu, &[10, 20].into()).unwrap();
        // GPU 1's 0.4 GiB gap is driver reserve, not worth naming.
        assert!(note.contains("GPU 0 95.0GiB") && !note.contains("GPU 1"), "{note}");
        // Every process read in full: the gap has some other cause.
        assert_eq!(unaccounted_note(&selected, &by_gpu, &HashSet::new()), None);
    }

    fn header_names(names: &[&str]) -> Vec<String> {
        let holders: Vec<Holder> = names.iter().map(|n| holder(0, 1.0, Some(n), "u")).collect();
        let gpus = [0];
        let cols = columns(&gpus, &holders);
        let lines = render::container_matrix(&gpus, &cols, &render::NameWidth::Cap, &render::UNICODE, false);
        lines[0].split_whitespace().skip(1).map(str::to_string).collect()
    }

    /// Cut from the end they'd look the same; cut from the middle they don't.
    #[test]
    fn look_alike_names_cut_in_the_middle() {
        let got = header_names(&["bench-oneocr0930_10427-g4", "bench-oneocr0930_10427-g5"]);
        assert_eq!(got, ["bench-o…10427-g4", "bench-o…10427-g5"]);
    }

    /// Only the look-alikes change; other names keep the usual cut.
    #[test]
    fn other_names_keep_the_end_cut() {
        let got = header_names(&[
            "bench-oneocr0930_10427-g4",
            "docker-train-fulllr",
            "bench-oneocr0930_10427-g5",
        ]);
        // Equal VRAM, so the columns fall back to name order.
        assert_eq!(got, ["bench-o…10427-g4", "bench-o…10427-g5", "docker-train-fu…"]);
    }

    /// Same head and same tail: widen until the difference shows.
    #[test]
    fn look_alikes_widen_when_middle_cut_is_not_enough() {
        let got = header_names(&["aaaaaaaaaaX-bbbbbbbbbbbbbb", "aaaaaaaaaaY-bbbbbbbbbbbbbb"]);
        assert_ne!(got[0], got[1]);
        assert!(got.iter().all(|n| n.contains('X') || n.contains('Y')), "{got:?}");
    }
}
