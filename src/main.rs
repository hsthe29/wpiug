mod amd;
mod args;
mod container;
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
    let gpu_procs = amd::discover_procs(&all_gpus);
    let table = procs::snapshot();
    let mut resolver = container::Resolver::new();

    let mut by_gpu: HashMap<u32, Vec<&amd::GpuProc>> = HashMap::new();
    for p in &gpu_procs {
        by_gpu.entry(p.gpu_index).or_default().push(p);
    }

    // Build the trees first: every pid they touch needs a label.
    let mut trees: Vec<(u32, Vec<Node>)> = Vec::new();
    let mut labelled: HashSet<u32> = HashSet::new();

    for gpu in &selected {
        let pids: Vec<u32> = by_gpu
            .get(&gpu.index)
            .map(|v| v.iter().map(|p| p.pid).collect())
            .unwrap_or_default();
        let nodes = forest(&pids, a.depth, &table, &mut labelled);
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
    // `-w` cuts commands to the terminal; `-ww` lets them run to full length.
    let cmd_limit = match a.wide {
        args::Wide::Full => None,
        _ => Some(term),
    };

    let (report, note) = render_report(
        &selected,
        &by_gpu,
        &trees,
        &labels,
        glyphs,
        color,
        term.min(render::WIDTH),
        cmd_limit,
    );
    emit(&report)?;
    // After the report rather than before it, so it is the last thing on
    // screen instead of scrolling away above the output.
    if let Some(note) = note {
        eprintln!("wpiug: {note}");
    }
    Ok(())
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

/// The report, plus a note for stderr when `-w` had no room to show commands.
///
/// `width` is how wide the banners are drawn: the terminal's, capped at
/// `render::WIDTH`, so they never wrap on a narrow one.
#[allow(clippy::too_many_arguments)]
fn render_report(
    selected: &[&amd::Gpu],
    by_gpu: &HashMap<u32, Vec<&amd::GpuProc>>,
    trees: &[(u32, Vec<Node>)],
    labels: &HashMap<u32, render::Label>,
    glyphs: &render::Glyphs,
    color: bool,
    width: usize,
    cmd_limit: Option<usize>,
) -> (String, Option<String>) {
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

    let _ = writeln!(s, "{}", render::banner("Process Detail", width));
    // Build every GPU's rows before emitting any, so the VRAM and command
    // columns can be sized once across the whole section rather than per block.
    let mut blocks: Vec<render::Block> = Vec::new();
    for (index, nodes) in trees {
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
            &header, nodes, labels, &vram, glyphs, color,
        )));
    }

    let col = render::label_column(&blocks);
    let cmd_col = render::cmd_column(&blocks, col, cmd_limit);
    for block in &blocks {
        match block {
            render::Block::Tree(rows) => {
                for line in render::align(rows, col, &cmd_col, glyphs, color) {
                    let _ = writeln!(s, "{line}");
                }
            }
            render::Block::Idle(ids) => {
                let _ = writeln!(s, "{}", render::idle_line(ids, color));
            }
        }
    }
    let _ = writeln!(s, "{}", render::rule(width));
    let _ = writeln!(s, "{}", render::banner("End of Wpiug Log", width));

    let note = match (cmd_col, cmd_limit) {
        (render::CmdCol::Squeezed { need }, Some(term)) => Some(format!(
            "commands hidden: -w needs about {need} columns here, the terminal has {term} \
             (widen it, or use -ww to print them in full)"
        )),
        _ => None,
    };
    (s, note)
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
            by_pid.insert(pid, procs::ProcInfo { ppid, name });
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
}
