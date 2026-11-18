use crate::muon::config::CacheConfig;

const INVALID: u64 = u64::MAX;

#[derive(Clone, Copy)]
struct Line {
    tag: u64,   // line number
    ready: u64, // cycle its data arrives
    dirty: bool, // only in the level backed by DRAM: the levels above are write-through
    atomic: u64, // cycle its last atomic finishes
}

/// One cache level.  All levels live in one array and link to each other by index.
pub struct Cache {
    conf: CacheConfig,
    next: Option<usize>, // next level; None = DRAM
    prev: Vec<usize>,    // caches whose next level is this one
    lines: Vec<Line>,    // sets × ways, each set newest first
    mshrs: Vec<u64>,     // cycle each frees
    atomics: Vec<u64>,   // cycle each atomic unit frees; as many as MSHRs
    bus_free: u64,       // cycle the link to the next level frees
    pub free: u64,       // first cycle it can take the next memory op from a core
    pub depth: usize,    // levels below this one before DRAM
    pub accesses: u64,
    pub misses: u64,
}

impl Cache {
    fn new(conf: CacheConfig, next: Option<usize>, depth: usize) -> Self {
        Cache {
            conf,
            next,
            prev: vec![],
            lines: vec![Line { tag: INVALID, ready: 0, dirty: false, atomic: 0 }; conf.sets * conf.ways],
            mshrs: vec![0; conf.mshrs],
            atomics: vec![0; conf.mshrs],
            bus_free: 0,
            free: 0,
            depth,
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
    let depth = next.map_or(0, |n| caches[n].depth + 1);
    caches.push(Cache::new(conf, next, depth));
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

/// access `line` in cache `i` at cycle `t`: (cycle accepted, cycle its data is available).  A level
/// above another cache is write-through: a store completes on its own timing and is passed on to the
/// next level, so everything below the first level always sees it.
pub fn access(caches: &mut [Cache], i: usize, line: u64, t: u64, write: bool) -> (u64, u64) {
    let c = &mut caches[i];
    let (conf, next) = (c.conf, c.next);
    c.accesses += 1;
    let hit = c.set(line).iter_mut().find(|l| l.tag == line).map(|l| {
        l.dirty |= write && next.is_none();
        t.max(l.ready) + conf.hit_latency
    });
    if let Some(done) = hit {
        if let (true, Some(n)) = (write, next) {
            // the store holds the link to the next level as a miss does
            let start = t.max(caches[i].bus_free);
            caches[i].bus_free = start + conf.bus_cycles;
            access(caches, n, line, start, true);
        }
        return (t, done);
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
    let fill = next.map_or(start, |n| access(caches, n, line, start, write).1); // a store's fill passes it on
    let done = fill + conf.miss_latency;
    caches[i].mshrs[m] = done;
    let set = caches[i].set(line);
    set.rotate_right(1);
    set[0] = Line { tag: line, ready: done, dirty: write && next.is_none(), atomic: 0 };
    (accepted, done)
}

/// an atomic on `line` sent at cycle `t` from cache `i`: the cycle its result is back.  The last level
/// serves it: atomics on one line run one after another, each holding the line and an atomic unit for
/// the hit latency.  The levels above are skipped, and the trip back costs what their fills cost.
pub fn atomic(caches: &mut [Cache], mut i: usize, line: u64, t: u64) -> u64 {
    let mut back = 0;
    while let Some(n) = caches[i].next {
        back += caches[i].conf.miss_latency;
        i = n;
    }
    if caches[i].set(line).iter().all(|l| l.tag != line) {
        access(caches, i, line, t, true); // fetch the line first
    } else {
        caches[i].accesses += 1;
    }
    let c = &mut caches[i];
    let l = c.set(line).iter_mut().find(|l| l.tag == line).unwrap();
    l.dirty = true;
    let free = t.max(l.ready).max(l.atomic);
    // the unit that frees last before then, so one waiting for the line takes over the unit of the one ahead
    let u = (0..c.atomics.len()).filter(|&u| c.atomics[u] <= free).max_by_key(|&u| c.atomics[u]);
    let u = u.unwrap_or_else(|| (0..c.atomics.len()).min_by_key(|&u| c.atomics[u]).unwrap());
    let done = free.max(c.atomics[u]) + c.conf.hit_latency;
    c.atomics[u] = done;
    c.set(line).iter_mut().find(|l| l.tag == line).unwrap().atomic = done;
    done + back
}
