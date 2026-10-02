use std::borrow::Cow;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Run a command, capture stdout, and give up after `timeout` so a wedged
/// daemon (docker/containerd) can never hang the whole tool.
pub fn run(bin: &str, args: &[&str], timeout: Duration) -> Result<String, String> {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to run `{bin}`: {e}"))?;

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                if let Some(mut s) = child.stdout.take() {
                    let _ = s.read_to_string(&mut out);
                }
                if !status.success() {
                    let mut err = String::new();
                    if let Some(mut s) = child.stderr.take() {
                        let _ = s.read_to_string(&mut err);
                    }
                    let err = err.trim();
                    return Err(if err.is_empty() {
                        format!("`{bin}` exited with {status}")
                    } else {
                        format!("`{bin}`: {err}")
                    });
                }
                return Ok(out);
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("`{bin}` timed out after {:?}", timeout));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => return Err(format!("failed to wait for `{bin}`: {e}")),
        }
    }
}

/// Display width, good enough for the ASCII that /proc and docker hand us.
pub fn width(s: &str) -> usize {
    s.chars().count()
}

/// First `w` characters of `s`, cut on a char boundary.
pub fn truncate(s: &str, w: usize) -> &str {
    match s.char_indices().nth(w) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `s` in at most `w` columns. Anything cut ends in `more`, so a shortened
/// command can't be mistaken for the whole one.
pub fn ellipsize<'a>(s: &'a str, w: usize, more: &str) -> Cow<'a, str> {
    if width(s) <= w {
        return Cow::Borrowed(s);
    }
    match w.checked_sub(width(more)) {
        // Trim so the marker sits against the last word, not after a gap.
        Some(keep) if keep > 0 => Cow::Owned(format!("{}{more}", truncate(s, keep).trim_end())),
        _ => Cow::Borrowed(truncate(s, w)),
    }
}

/// Columns of the terminal we are attached to, if any.
///
/// std already links libc, so `TIOCGWINSZ` costs us nothing extra - no crate
/// needed. stdout is asked first, then stderr, so `wpiug -w | less` still sizes
/// to the real terminal; `COLUMNS` is the last resort.
pub fn term_width() -> Option<usize> {
    use std::ffi::{c_int, c_ulong};

    #[repr(C)]
    struct Winsize {
        row: u16,
        col: u16,
        xpixel: u16,
        ypixel: u16,
    }

    #[cfg(any(
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "sparc",
        target_arch = "sparc64",
    ))]
    const TIOCGWINSZ: c_ulong = 0x4008_7468;
    #[cfg(not(any(
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "sparc",
        target_arch = "sparc64",
    )))]
    const TIOCGWINSZ: c_ulong = 0x5413;

    extern "C" {
        fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    }

    for fd in [1, 2] {
        let mut ws = Winsize {
            row: 0,
            col: 0,
            xpixel: 0,
            ypixel: 0,
        };
        // Safe: the kernel writes at most one Winsize through the pointer, and
        // a non-tty fd just returns an error we ignore.
        if unsafe { ioctl(fd, TIOCGWINSZ, &mut ws as *mut Winsize) } == 0 && ws.col > 0 {
            return Some(ws.col as usize);
        }
    }

    std::env::var("COLUMNS").ok()?.parse().ok()
}

/// Login name for `uid`, looked up through NSS the way `ps` does it, so LDAP
/// and SSSD accounts resolve too. `None` if no account has that uid.
pub fn user_name(uid: u32) -> Option<String> {
    use std::ffi::{c_char, c_int, CStr};

    // `struct passwd`, identical in glibc and musl on Linux.
    #[repr(C)]
    struct Passwd {
        name: *mut c_char,
        passwd: *mut c_char,
        uid: u32,
        gid: u32,
        gecos: *mut c_char,
        dir: *mut c_char,
        shell: *mut c_char,
    }

    extern "C" {
        fn getpwuid_r(
            uid: u32,
            pwd: *mut Passwd,
            buf: *mut c_char,
            len: usize,
            result: *mut *mut Passwd,
        ) -> c_int;
    }

    let mut pwd = std::mem::MaybeUninit::<Passwd>::uninit();
    // Far more than any real entry needs; a bigger one just reads as unknown.
    let mut buf = vec![0 as c_char; 16 * 1024];
    let mut found: *mut Passwd = std::ptr::null_mut();
    // Safe: getpwuid_r writes only into `pwd` and `buf`, both as large as we
    // say, and on success points `found` at `pwd`, whose strings live in `buf`.
    unsafe {
        let rc = getpwuid_r(uid, pwd.as_mut_ptr(), buf.as_mut_ptr(), buf.len(), &mut found);
        if rc != 0 || found.is_null() {
            return None;
        }
        Some(CStr::from_ptr((*found).name).to_string_lossy().into_owned())
    }
}

pub fn pad_right(s: &str, w: usize) -> String {
    let len = width(s);
    if len >= w {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(w - len))
    }
}

pub fn pad_left(s: &str, w: usize) -> String {
    let len = width(s);
    if len >= w {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(w - len))
    }
}
