use std::borrow::Cow;
use std::collections::HashMap;

use crate::amd::Gpu;
use crate::util::{ellipsize, pad_left, pad_right, width};

/// Widest the banners get, and the width assumed when no terminal answers.
pub const WIDTH: usize = 124;

/// Blank columns in front of the VRAM and command columns.
const GUTTER: usize = 2;
/// Width the VRAM values are right-aligned in.
const VRAM_W: usize = 9;
/// Narrowest the `-w` command column is worth drawing. Cut shorter than this,
/// a command is mostly its interpreter path (`/opt/venv/bin/`) and tells you
/// nothing.
const MIN_CMD: usize = 20;

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const PID: &str = "\x1b[36m"; // cyan
const NAME: &str = "\x1b[1m"; // bold
const CONTAINER: &str = "\x1b[33m"; // yellow
const VRAM: &str = "\x1b[32m"; // green

/// A process label split into its parts so each can be styled independently.
pub struct Label {
    pub pid: u32,
    pub name: String,
    pub container: Option<String>,
    /// Full command line, `None` unless `-w` asked for it. Kept raw: how much
    /// of it fits is decided once, at align time.
    pub cmd: Option<String>,
}

impl Label {
    /// Visible width, ignoring any escape codes we add.
    pub fn width(&self) -> usize {
        let mut w = self.pid.to_string().len() + 1 + width(&self.name);
        if let Some(c) = &self.container {
            w += 3 + width(c); // " (" + name + ")"
        }
        w
    }

    pub fn paint(&self, color: bool) -> String {
        let Label { pid, name, .. } = self;
        let container = match &self.container {
            Some(c) if color => format!(" {CONTAINER}({c}){RESET}"),
            Some(c) => format!(" ({c})"),
            None => String::new(),
        };
        if color {
            format!("{PID}{pid}{RESET} {NAME}{name}{RESET}{container}")
        } else {
            format!("{pid} {name}{container}")
        }
    }
}

/// `title` centred in a `=` bar `w` columns wide. Callers pass the terminal
/// width (capped at `WIDTH`) so the bar never wraps onto a second line.
pub fn banner(title: &str, w: usize) -> String {
    let t = format!(" {title} ");
    let tw = width(&t);
    if tw >= w {
        return t.trim().to_string();
    }
    let left = (w - tw) / 2;
    let right = w - tw - left;
    format!("{}{}{}", "=".repeat(left), t, "=".repeat(right))
}

pub fn rule(w: usize) -> String {
    "=".repeat(w)
}

pub fn human_bytes(b: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let f = b as f64;
    if f >= GIB {
        format!("{:.1}GiB", f / GIB)
    } else {
        format!("{:.0}MiB", f / MIB)
    }
}

/// The VRAM header is split at its `/` like the cells below it, so the two
/// halves can be padded to line up with the numbers.
const VRAM_HEAD: (&str, &str) = ("VRAM used", "total");
const GAP: &str = "   ";

/// Header, a rule under it, then one row per GPU, every column width derived
/// from the data.
///
/// `used` and `total` are padded independently so the `/` sits in the same
/// place on every row - right-aligning the joined string would not do that,
/// since `292MiB` is shorter than `183.1GiB`. The header's `/` is padded the
/// same way, so it sits right above them.
pub fn concise_table(entries: &[(&Gpu, usize)], color: bool) -> Vec<String> {
    let used: Vec<String> = entries
        .iter()
        .map(|(g, _)| human_bytes(g.vram_used))
        .collect();
    let total: Vec<String> = entries
        .iter()
        .map(|(g, _)| human_bytes(g.vram_total))
        .collect();
    let w_used = widest(&used).max(width(VRAM_HEAD.0));
    let w_total = widest(&total).max(width(VRAM_HEAD.1));

    let vram_head = format!(
        "{} / {}",
        pad_left(VRAM_HEAD.0, w_used),
        pad_right(VRAM_HEAD.1, w_total)
    );
    let headers = ["GPU", vram_head.as_str(), "VRAM%", "Util%", "Procs"];

    let cols: [Vec<String>; 5] = [
        entries.iter().map(|(g, _)| g.index.to_string()).collect(),
        (0..entries.len())
            .map(|i| {
                format!(
                    "{} / {}",
                    pad_left(&used[i], w_used),
                    pad_left(&total[i], w_total)
                )
            })
            .collect(),
        entries
            .iter()
            .map(|(g, _)| format!("{}%", g.vram_percent()))
            .collect(),
        entries
            .iter()
            .map(|(g, _)| match g.busy {
                Some(v) => format!("{v}%"),
                None => "N/A".to_string(),
            })
            .collect(),
        entries.iter().map(|(_, n)| n.to_string()).collect(),
    ];

    let widths: Vec<usize> = cols
        .iter()
        .zip(headers)
        .map(|(col, header)| widest(col).max(width(header)))
        .collect();

    // The GPU id is a label, so it reads better flush left; the rest are
    // quantities and line up on their last digit.
    let row = |cells: &[&str]| {
        cells
            .iter()
            .zip(&widths)
            .enumerate()
            .map(|(i, (c, w))| {
                if i == 0 {
                    pad_right(c, *w)
                } else {
                    pad_left(c, *w)
                }
            })
            .collect::<Vec<_>>()
            .join(GAP)
    };

    // The rule spans the table, not the banner: a full-width one left the
    // table looking stranded at the left edge.
    let header = row(&headers);
    let mut out = vec![bold(&header, color), "-".repeat(width(&header))];
    for r in 0..entries.len() {
        let cells: Vec<&str> = cols.iter().map(|c| c[r].as_str()).collect();
        out.push(row(&cells));
    }
    out
}

fn widest<S: AsRef<str>>(items: &[S]) -> usize {
    items
        .iter()
        .map(|s| width(s.as_ref()))
        .max()
        .unwrap_or_default()
}

/// A process-tree node, already trimmed to the requested depth.
pub struct Node {
    pub pid: u32,
    pub children: Vec<Node>,
}

pub fn bold(s: &str, color: bool) -> String {
    if color {
        format!("{NAME}{s}{RESET}")
    } else {
        s.to_string()
    }
}

pub struct Glyphs {
    tee: &'static str,
    ell: &'static str,
    bar: &'static str,
    gap: &'static str,
    /// Marks a command cut short, so it can't pass for the whole thing.
    more: &'static str,
}

pub const UNICODE: Glyphs = Glyphs {
    tee: "├── ",
    ell: "└── ",
    bar: "│   ",
    gap: "    ",
    more: "…",
};

pub const ASCII: Glyphs = Glyphs {
    tee: "|-- ",
    ell: "`-- ",
    bar: "|   ",
    gap: "    ",
    more: "...",
};

/// One rendered tree row. `vis` is the width with escape codes discounted -
/// `line.len()` would be wrong once colour is on.
pub struct Row {
    pub line: String,
    pub vis: usize,
    pub cmd: Option<String>,
    pub vram: Option<u64>,
}

/// Classic top-down tree, one process per line:
///
///     2127172 python3 qwen35-step600
///     ├── 2128967 python3 qwen35-step600
///     ├── 2128968 sglang::schedul qwen35-step600
///     │   └── 2132556 python3 qwen35-step600
///     └── 2128969 sglang::detoken qwen35-step600
/// The `[GPU N]` header doubles as the tree root, so top-level processes get
/// connectors too and can't be mistaken for children of the one above.
pub fn forest_rows(
    root_label: &str,
    nodes: &[Node],
    labels: &HashMap<u32, Label>,
    vram: &HashMap<u32, u64>,
    g: &Glyphs,
    color: bool,
) -> Vec<Row> {
    let mut out = vec![Row {
        line: bold(root_label, color),
        vis: width(root_label),
        cmd: None,
        vram: None,
    }];

    let last = nodes.len().saturating_sub(1);
    for (i, node) in nodes.iter().enumerate() {
        let connector = if i == last { g.ell } else { g.tee };
        push(
            node,
            labels,
            vram,
            g,
            String::new(),
            Some(connector),
            color,
            &mut out,
        );
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn push(
    node: &Node,
    labels: &HashMap<u32, Label>,
    vram: &HashMap<u32, u64>,
    g: &Glyphs,
    prefix: String,
    connector: Option<&str>,
    color: bool,
    out: &mut Vec<Row>,
) {
    let fallback = Label {
        pid: node.pid,
        name: "?".to_string(),
        container: None,
        cmd: None,
    };
    let label = labels.get(&node.pid).unwrap_or(&fallback);

    let drawn = format!("{prefix}{}", connector.unwrap_or(""));
    out.push(Row {
        vis: width(&drawn) + label.width(),
        line: if color {
            format!("{DIM}{drawn}{RESET}{}", label.paint(true))
        } else {
            format!("{drawn}{}", label.paint(false))
        },
        cmd: label.cmd.clone(),
        vram: vram.get(&node.pid).copied(),
    });

    // Children hang below a vertical bar, unless we were the last sibling -
    // then the bar stops and the space below is blank.
    let child_prefix = match connector {
        None => prefix,
        Some(c) if c == g.ell => format!("{prefix}{}", g.gap),
        Some(_) => format!("{prefix}{}", g.bar),
    };

    let last = node.children.len().saturating_sub(1);
    for (i, child) in node.children.iter().enumerate() {
        let connector = if i == last { g.ell } else { g.tee };
        push(
            child,
            labels,
            vram,
            g,
            child_prefix.clone(),
            Some(connector),
            color,
            out,
        );
    }
}

/// One line standing in for a run of GPUs nothing is running on, e.g.
/// `[GPU 1-4] no processes`. The concise table already lists them, so a tree
/// header apiece would only be noise.
pub fn idle_line(ids: &[u32], color: bool) -> String {
    let header = format!("[GPU {}]", id_ranges(ids));
    let note = "no processes";
    if color {
        format!("{} {DIM}{note}{RESET}", bold(&header, true))
    } else {
        format!("{header} {note}")
    }
}

/// `[1, 2, 3, 4, 6]` -> `1-4,6`.
fn id_ranges(ids: &[u32]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        let start = ids[i];
        while i + 1 < ids.len() && ids[i + 1] == ids[i] + 1 {
            i += 1;
        }
        parts.push(if ids[i] == start {
            start.to_string()
        } else {
            format!("{start}-{}", ids[i])
        });
        i += 1;
    }
    parts.join(",")
}

/// One piece of the Process Detail section, in GPU order.
pub enum Block {
    /// A GPU header and the process tree under it.
    Tree(Vec<Row>),
    /// A run of GPUs nothing is running on, drawn as a single `idle_line`.
    Idle(Vec<u32>),
}

impl Block {
    /// Rows that take part in column alignment - none for an idle line, which
    /// has nothing after it to line up.
    fn rows(&self) -> &[Row] {
        match self {
            Block::Tree(rows) => rows,
            Block::Idle(_) => &[],
        }
    }
}

/// Width the label column must clear. Computed over every row of every GPU so
/// what follows forms one straight column down the whole section.
pub fn label_column(blocks: &[Block]) -> usize {
    blocks
        .iter()
        .flat_map(Block::rows)
        .map(|r| r.vis)
        .max()
        .unwrap_or_default()
}

/// How the command column is laid out.
pub enum CmdCol {
    /// Nothing to show - no `-w`, or no command on any row.
    Off,
    /// `-w`, but the terminal leaves less than `MIN_CMD` columns for it; `need`
    /// is how wide it would have to be.
    Squeezed { need: usize },
    /// Commands cut to this many columns so no line runs off the terminal.
    Cut(usize),
    /// Commands printed whole (`-ww`), however long the line gets.
    Full,
}

/// Decide the command column from the commands the rows actually carry.
///
/// `limit` is the terminal width under `-w`, or `None` under `-ww`. The command
/// is the last column, so under `-w` it gets whatever the label and VRAM leave.
pub fn cmd_column(blocks: &[Block], label_col: usize, limit: Option<usize>) -> CmdCol {
    let widest = blocks
        .iter()
        .flat_map(Block::rows)
        .filter_map(|r| r.cmd.as_deref())
        .map(width)
        .max()
        .unwrap_or_default();
    if widest == 0 {
        return CmdCol::Off;
    }

    let Some(term) = limit else {
        return CmdCol::Full;
    };
    let before = label_col + GUTTER + VRAM_W + GUTTER;
    let room = term.saturating_sub(before);
    // Too narrow to say anything useful - unless every command fits anyway.
    if room < MIN_CMD.min(widest) {
        CmdCol::Squeezed {
            need: before + MIN_CMD,
        }
    } else {
        CmdCol::Cut(room)
    }
}

/// Lay each row out as `label   VRAM   command`, padding from `Row::vis` so
/// escape codes never skew the columns. The command goes last, the way `ps`
/// and `top` do it: however long it runs, it can't push the VRAM off screen,
/// so VRAM is one straight column under `-ww` too. Trailing columns a row does
/// not have are simply left off, so no line ends in whitespace.
pub fn align(
    rows: &[Row],
    label_col: usize,
    cmd_col: &CmdCol,
    g: &Glyphs,
    color: bool,
) -> Vec<String> {
    rows.iter()
        .map(|r| {
            let mut line = r.line.clone();

            let cmd = match (&r.cmd, cmd_col) {
                (Some(cmd), CmdCol::Cut(w)) => Some(ellipsize(cmd, *w, g.more)),
                (Some(cmd), CmdCol::Full) => Some(Cow::Borrowed(cmd.as_str())),
                _ => None,
            };
            if r.vram.is_none() && cmd.is_none() {
                return line;
            }

            line.push_str(&" ".repeat(label_col.saturating_sub(r.vis) + GUTTER));
            // A row without VRAM still holds its place, so its command lines
            // up with the rest.
            let bytes = pad_left(&r.vram.map(human_bytes).unwrap_or_default(), VRAM_W);
            if color && r.vram.is_some() {
                line.push_str(&format!("{VRAM}{bytes}{RESET}"));
            } else {
                line.push_str(&bytes);
            }

            if let Some(cmd) = cmd {
                line.push_str(&" ".repeat(GUTTER));
                if color {
                    line.push_str(&format!("{DIM}{cmd}{RESET}"));
                } else {
                    line.push_str(&cmd);
                }
            }

            line
        })
        .collect()
}
