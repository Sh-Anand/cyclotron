use crate::muon::config::CacheConfig;

const INVALID: u64 = u64::MAX;

#[derive(Clone, Copy)]
struct Line {
    tag: u64,   // line number
    ready: u64, // cycle its data arrives
    dirty: bool,
}

/// One cache level.  All levels live in one array and link to each other by index.
pub struct Cache {
    conf: CacheConfig,
    next: Option<usize>, // next level; None = DRAM
    prev: Vec<usize>,    // caches whose next level is this one
    lines: Vec<Line>,    // sets × ways, each set newest first
    mshrs: Vec<u64>,     // cycle each frees
    bus_free: u64,       // cycle the link to the next level frees
    pub accesses: u64,
    pub misses: u64,
}

impl Cache {
    fn new(conf: CacheConfig, next: Option<usize>) -> Self {
        Cache {
            conf,
            next,
            prev: vec![],
            lines: vec![Line { tag: INVALID, ready: 0, dirty: false }; conf.sets * conf.ways],
            mshrs: vec![0; conf.mshrs],
            bus_free: 0,
            accesses: 0,
            misses: 0,
        }
    }

    fn set(&mut self, line: u64) -> &mut [Line] {
        let s = line as usize % self.conf.sets * self.conf.ways;
        &mut self.lines[s..s + self.conf.ways]
    }
}

/// append a cache whose misses go to `next`; returns its index
pub fn add(caches: &mut Vec<Cache>, conf: CacheConfig, next: Option<usize>) -> usize {
    let i = caches.len();
    if let Some(n) = next {
        caches[n].prev.push(i)
    }
    caches.push(Cache::new(conf, next));
    i
}

/// drop `line` from every cache above `i`; whether a dropped copy was dirty
fn evict_above(caches: &mut [Cache], i: usize, line: u64) -> bool {
    let mut dirty = false;
    for k in 0..caches[i].prev.len() {
        let p = caches[i].prev[k];
        if let Some(l) = caches[p].set(line).iter_mut().find(|l| l.tag == line) {
            dirty |= l.dirty;
            l.tag = INVALID;
            dirty |= evict_above(caches, p, line);
        }
    }
    dirty
}

/// access `line` in cache `i` at cycle `t`: (cycle accepted, cycle its data is available)
pub fn access(caches: &mut [Cache], i: usize, line: u64, t: u64, write: bool) -> (u64, u64) {
    let c = &mut caches[i];
    let (conf, next) = (c.conf, c.next);
    c.accesses += 1;
    if let Some(l) = c.set(line).iter_mut().find(|l| l.tag == line) {
        l.dirty |= write;
        return (t, t.max(l.ready) + conf.hit_latency);
    }
    c.misses += 1;
    let m = (0..conf.mshrs).min_by_key(|&m| c.mshrs[m]).unwrap();
    let accepted = t.max(c.mshrs[m]);
    let start = accepted.max(c.bus_free);
    c.bus_free = start + conf.bus_cycles;
    let old = c.set(line)[conf.ways - 1]; // the oldest line in the set
    if old.tag != INVALID && (old.dirty | evict_above(caches, i, old.tag)) {
        caches[i].bus_free += conf.bus_cycles; // write it back
        if let Some(n) = next {
            access(caches, n, old.tag, start, true);
        }
    }
    let fill = next.map_or(start, |n| access(caches, n, line, start, false).1);
    let done = fill + conf.miss_latency;
    caches[i].mshrs[m] = done;
    let set = caches[i].set(line);
    set.rotate_right(1);
    set[0] = Line { tag: line, ready: done, dirty: write };
    (accepted, done)
}
