pub const HELP: &str = "\
wpiug - What Process Is Using GPU

USAGE:
    wpiug                    Query every GPU on the machine (default)
    wpiug <ID>...            Query only the given GPU index(es), e.g. `wpiug 0` or `wpiug 0,3`

FLAGS:
    --depth <N>              Tree levels to draw from each GPU process down, counting itself
                             (default: 3, minimum: 1). GPU processes are always shown
    -a, --all                Query all GPUs (the default; accepted for explicitness)
    --ascii                  Draw the tree with ASCII instead of box-drawing characters
    -w                       Show each process's command, truncated to the terminal width
    -ww                      Show each process's command in full, never truncated (like `ps`)
    --color / --no-color     Force colour on or off (default: on when stdout is a terminal)
    -h, --help               Print this help
    -V, --version            Print version

Processes inside a container show as `<pid> <process_name> (<container_name>)`;
host processes show as `<pid> <process_name>`. NO_COLOR is honoured.
";

/// How much of each process's command line to show, counted in `-w`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wide {
    /// No command column at all.
    Off,
    /// Show it, cut to whatever the terminal has room for.
    Fit,
    /// Show all of it, however long the line gets.
    Full,
}

#[derive(Debug)]
pub struct Args {
    /// `None` means "all GPUs", `Some(list)` means only those indexes.
    pub only: Option<Vec<u32>>,
    pub depth: usize,
    pub ascii: bool,
    pub wide: Wide,
    /// `None` = decide from the terminal at render time.
    pub color: Option<bool>,
}

pub enum Parsed {
    Run(Args),
    Help,
    Version,
}

pub fn parse<I: Iterator<Item = String>>(argv: I) -> Result<Parsed, String> {
    let mut all = false;
    let mut ascii = false;
    let mut wide = 0usize;
    let mut color: Option<bool> = None;
    let mut ids: Vec<u32> = Vec::new();
    let mut depth: usize = 3;
    let mut it = argv.peekable();

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Parsed::Help),
            "-V" | "--version" => return Ok(Parsed::Version),
            "-a" | "--all" => all = true,
            "--ascii" => ascii = true,
            // Repeats add up, so `-w -w` means the same as `-ww`.
            "-w" => wide += 1,
            "-ww" => wide += 2,
            "--color" => color = Some(true),
            "--no-color" => color = Some(false),
            "--depth" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--depth requires a value".to_string())?;
                depth = parse_depth(&v)?;
            }
            other => {
                if let Some(v) = other.strip_prefix("--depth=") {
                    depth = parse_depth(v)?;
                } else if other.starts_with('-') {
                    return Err(format!("unknown flag `{other}` (try --help)"));
                } else {
                    // Accept `0`, `0,1,2` and `0 1 2`.
                    for part in other.split(',') {
                        let part = part.trim();
                        if part.is_empty() {
                            continue;
                        }
                        let id: u32 = part
                            .parse()
                            .map_err(|_| format!("invalid GPU id `{part}`"))?;
                        ids.push(id);
                    }
                }
            }
        }
    }

    if all && !ids.is_empty() {
        return Err("--all cannot be combined with explicit GPU ids".to_string());
    }

    ids.sort_unstable();
    ids.dedup();

    Ok(Parsed::Run(Args {
        // No ids given means every GPU; `--all` just says so explicitly.
        only: if ids.is_empty() { None } else { Some(ids) },
        depth,
        ascii,
        wide: match wide {
            0 => Wide::Off,
            1 => Wide::Fit,
            _ => Wide::Full,
        },
        color,
    }))
}

fn parse_depth(v: &str) -> Result<usize, String> {
    let n: usize = v
        .parse()
        .map_err(|_| format!("invalid --depth value `{v}`"))?;
    if n == 0 {
        return Err("--depth must be at least 1".to_string());
    }
    Ok(n)
}
