use crate::muon::config::Unit;
use std::fmt::{Display, Formatter};

/// Per-core performance counters; every busy core cycle is either an issue or one kind of stall.
#[derive(Debug, Default, Clone, Copy)]
pub struct CoreStats {
    pub unit_issued: [u64; Unit::ALL.len()],
    pub stall_deps: u64,  // entries in some window, none ready (window hazards or scoreboard)
    pub stall_fetch: u64, // windows empty, waiting on fetch
    pub stall_barrier: u64, // windows empty, some warp waiting at a barrier
    pub gmem_instrs: u64,
    pub gmem_reqs: u64, // lines presented to the L0; more than one per instruction is uncoalesced
    pub smem_instrs: u64,
    pub smem_reqs: u64, // serial bank requests; more than one per instruction is a bank conflict
}

impl CoreStats {
    pub fn add(&mut self, o: &CoreStats) {
        self.unit_issued.iter_mut().zip(o.unit_issued).for_each(|(a, b)| *a += b);
        self.stall_deps += o.stall_deps;
        self.stall_fetch += o.stall_fetch;
        self.stall_barrier += o.stall_barrier;
        self.gmem_instrs += o.gmem_instrs;
        self.gmem_reqs += o.gmem_reqs;
        self.smem_instrs += o.smem_instrs;
        self.smem_reqs += o.smem_reqs;
    }
}

impl Display for CoreStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let issued: u64 = self.unit_issued.iter().sum();
        let cycles = issued + self.stall_deps + self.stall_fetch + self.stall_barrier;
        let pct = |n: u64| 100.0 * n as f64 / cycles.max(1) as f64;
        write!(
            f,
            "core stats: {} cycles | issued {} ({:.1}%) | deps {} ({:.1}%) | fetch {} ({:.1}%) | barrier {} ({:.1}%)\n ",
            cycles,
            issued, pct(issued),
            self.stall_deps, pct(self.stall_deps),
            self.stall_fetch, pct(self.stall_fetch),
            self.stall_barrier, pct(self.stall_barrier),
        )?;
        Unit::ALL.iter().zip(self.unit_issued).try_for_each(|(u, n)| write!(f, " {u:?} {n}"))?;
        write!(
            f,
            "\n  gmem {} instrs, {} requests | smem {} instrs, {} requests",
            self.gmem_instrs, self.gmem_reqs, self.smem_instrs, self.smem_reqs
        )
    }
}
