use crate::base::{behavior::*, module::*};
use crate::muon::cache::{self, Cache};
use crate::info;
use crate::muon::config::{LaneConfig, MuonConfig, Unit};
use crate::muon::decode::{DecodedInst, MicroOp};
use crate::muon::scheduler::Scheduler;
use crate::muon::stats::CoreStats;
use crate::muon::warp::{ExecErr, Warp};
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use crate::sim::trace::{MemTracer, Tracer};
use crate::utils::BitSlice;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct MuonState {
    last_issued: usize, // round-robin pointer for single issue
}

/// a CTA: (launching command id, block idx), as its warps' CSRs hold
pub type CtaId = (u64, (u32, u32, u32));

pub struct MuonCore {
    base: ModuleBase<MuonState, MuonConfig>,
    pub scheduler: Scheduler,
    pub warps: Vec<Warp>,

    logger: Arc<Logger>,
    tracer: Arc<Tracer>,
    mem_tracer: Arc<MemTracer>,
    pub trace: bool,
    pub stats: CoreStats, // accumulates across CTAs; not cleared by reset
    pub arrived: Option<CtaId>, // CTA of a warp that just issued a barrier
    pub l0: usize, // this core's instruction cache
    pub l1: usize, // its SM's cache, the first level for data
}

impl MuonCore {
    pub fn new(
        config: Arc<MuonConfig>,
        cluster_id: usize,
        core_id: usize,
        logger: &Arc<Logger>,
        l0: usize,
        l1: usize,
    ) -> Self {
        let num_warps = config.num_warps;
        let mut core = MuonCore {
            base: Default::default(),
            scheduler: Scheduler::new(Arc::clone(&config)),
            warps: (0..num_warps)
                .map(|warp_id| {
                    Warp::new(
                        Arc::new(MuonConfig {
                            lane_config: LaneConfig {
                                warp_id,
                                core_id,
                                cluster_id,
                                ..config.lane_config
                            },
                            ..*config
                        }),
                        logger,
                    )
                })
                .collect(),
            logger: logger.clone(),
            tracer: Arc::new(Tracer::new(&config)),
            mem_tracer: Arc::new(MemTracer::new()),
            trace: false,
            stats: CoreStats::default(),
            arrived: None,
            l0,
            l1,
        };

        info!(
            core.logger,
            "muon core {} in cluster {} instantiated!", core_id, cluster_id
        );

        core.init_conf(Arc::clone(&config));
        core
    }

    /// Spawn a single warp to this core.
    /// spawn one CTA's warps on this core; each warp names its slot
    pub fn spawn_n_warps(
        &mut self,
        pc: u32,
        block_idx: (u32, u32, u32),
        warps: &[(usize, Vec<(u32, u32, u32)>)], // (wid, thread idxs)
        bp: u32,
        smem_base: u32,
        cmd_id: u64,
        regs: usize,
    ) {
        for (wid, thread_idxs) in warps {
            self.scheduler.spawn_warp(*wid, pc, thread_idxs.len());
            self.warps[*wid].set_block_threads_bp(block_idx, thread_idxs, bp, smem_base, cmd_id);
            self.warps[*wid].regs = regs;
            self.warps[*wid].fetch_ready = 0;
        }
    }

    /// unallocated warp slots, as many as the core's free regs hold at `regs` per thread
    pub fn free_warps(&self, regs: usize) -> Vec<usize> {
        let allocated = self.scheduler.state().allocated;
        let used: usize = self.warps.iter().filter(|w| allocated.bit(w.wid)).map(|w| w.regs).sum();
        (0..self.conf().num_warps).filter(|&w| !allocated.bit(w)).take((self.conf().regs_per_core - used) / regs).collect()
    }

    /// warps in `mask` that belong to CTA `id`
    pub(crate) fn cta_warps(&self, id: CtaId, mask: u32) -> impl Iterator<Item = usize> + '_ {
        (0..self.conf().num_warps).filter(move |&w| mask.bit(w) && self.warps[w].cta() == id)
    }

    pub fn cta_live(&self, id: CtaId) -> bool {
        self.cta_warps(id, self.scheduler.active_warp_mask()).next().is_some()
    }

    /// free CTA `id`'s warp slots, stopping any still running; whether it had any here
    pub fn free_cta(&mut self, id: CtaId) -> bool {
        let mask = self.cta_warps(id, self.scheduler.state().allocated).fold(0, |m, w| m | 1 << w);
        let s = self.scheduler.state_mut();
        (s.active_warps, s.allocated) = (s.active_warps & !mask, s.allocated & !mask);
        mask != 0
    }

    /// RTL-style single issue: the next warp, round-robin, with a ready ibuf entry
    fn pick_warp(&mut self, now: u64, smem_free: u64, gmem_free: u64) -> Option<(usize, usize)> {
        let n = self.conf().num_warps;
        let (wid, i) = (1..=n)
            .map(|k| (self.base.state.last_issued + k) % n)
            .find_map(|w| self.ready_entry(w, now, smem_free, gmem_free).map(|i| (w, i)))?;
        self.base.state.last_issued = wid;
        Some((wid, i))
    }

    /// oldest ibuf entry of warp `w` with no hazard on older entries and its registers ready at `now`
    fn ready_entry(&self, w: usize, now: u64, smem_free: u64, gmem_free: u64) -> Option<usize> {
        let warp = &self.warps[w];
        (0..warp.ibuf.len()).find(|&i| {
            let u = &warp.ibuf[i].inst;
            let (rd, mem) = (u.rd(), u.unit() == Unit::Lsu);
            let hazard = |o: &DecodedInst| {
                o.rd().is_some_and(|r| rd == Some(r) || u.rs().any(|s| s == r)) // WAW, RAW
                    || rd.is_some_and(|r| o.rs().any(|s| s == r)) // WAR
                    || (mem && o.unit() == Unit::Lsu) // memory ops stay in order
            };
            (i == 0 || !u.is_control()) // control issues last
                && !warp.ibuf.range(..i).any(|o| hazard(&o.inst))
                && u.rs().chain(rd).all(|r| warp.reg_ready[r] <= now) // scoreboard: RAW, WAW
                // one memory op at a time per SM, for shared memory and for global memory.  A global
                // op also waits for the warp's older stores, and a store for its older loads
                && (!mem || if warp.ibuf[i].smem > 0 {
                    smem_free
                } else {
                    gmem_free.max(warp.store_done).max(if u.is_store() { warp.load_done } else { 0 })
                } <= now)
        })
    }

    /// top up warp `wid`'s ibuf; non-control instructions execute as they are decoded
    fn refill(
        &mut self,
        wid: usize,
        now: u64,
        smem: &mut FlatMemory,
        gmem: &mut FlatMemory,
        caches: &mut [Cache],
    ) -> Result<(), ExecErr> {
        while self.warps[wid].fetch_ready <= now
            && self.warps[wid].ibuf.len() < self.conf().ibuf_depth
            && !self.warps[wid].ibuf.back().is_some_and(|u| u.inst.is_control())
        {
            // entering a new line fetches it through the instruction cache; a miss stops the warp until
            // the line arrives, and the warp then keeps the line even if the cache has replaced it
            let line = self.scheduler.pc(wid) as u64 / self.conf().gmem.line as u64;
            if line != self.warps[wid].fetched {
                self.warps[wid].fetched = line;
                let ready = cache::access(caches, self.l0, line, now, false).1;
                if ready > now {
                    self.warps[wid].fetch_ready = ready;
                    break;
                }
            }
            let mut uop = self.warps[wid].frontend(self.scheduler.schedule(wid).unwrap(), gmem);
            uop.smem = if uop.inst.is_control() { 0 } else { self.exec(wid, uop, smem, gmem)? };
            self.warps[wid].ibuf.push_back(uop);
        }
        Ok(())
    }

    /// Advance the core by one cycle, cycle `now` of the one global clock: issue one instruction from one warp.
    pub fn process(
        &mut self,
        now: u64,
        smem: &mut FlatMemory,
        smem_free: &mut u64,
        gmem: &mut FlatMemory,
        caches: &mut [Cache],
    ) -> Result<(), ExecErr> {
        let active = self.scheduler.active_warp_mask();
        for w in (0..self.conf().num_warps).filter(|&w| active.bit(w)) {
            self.refill(w, now, smem, gmem, caches)?;
        }
        if let Some((wid, i)) = self.pick_warp(now, *smem_free, caches[self.l1].free) {
            // control executes at issue; everything else already executed at decode
            let uop = self.warps[wid].ibuf.remove(i).unwrap();
            self.stats.unit_issued[uop.inst.unit() as usize] += 1;
            if uop.inst.is_control() {
                self.exec(wid, uop, smem, gmem)?;
                if uop.inst.is_barrier() {
                    self.warps[wid].fetch_ready = u64::MAX;
                    self.arrived = Some(self.warps[wid].cta());
                }
            }
            let unit = uop.inst.unit();
            let latency = self.conf().passes(unit) - 1 + self.conf().units.get(unit).latency;
            let done = if uop.smem > 0 {
                self.stats.smem_instrs += 1;
                self.stats.smem_reqs += uop.smem as u64;
                *smem_free = now + uop.smem as u64 * self.conf().smem.per_bank_latency;
                *smem_free + self.conf().smem.startup_latency
            } else if unit == Unit::Lsu {
                // global memory: present each line to the L1 a cycle after it accepted the previous one
                let (mut t, mut ready) = (now, now);
                let lines = self.warps[wid].gmem_lines.pop_front().unwrap();
                self.stats.gmem_instrs += 1;
                self.stats.gmem_reqs += lines.len() as u64;
                for line in lines {
                    let (accepted, data) = cache::access(caches, self.l1, line, t, uop.inst.is_store());
                    (t, ready) = (accepted + 1, ready.max(data));
                }
                caches[self.l1].free = t;
                let (warp, done) = (&mut self.warps[wid], ready + latency);
                if uop.inst.is_store() { warp.store_done = done } else { warp.load_done = warp.load_done.max(done) }
                done
            } else {
                now + latency
            };
            if let Some(rd) = uop.inst.rd() {
                self.warps[wid].reg_ready[rd] = done;
            }
        } else if self.warps.iter().any(|w| !w.ibuf.is_empty()) {
            self.stats.stall_deps += 1;
        } else if self.warps.iter().any(|w| w.fetch_ready == u64::MAX) {
            self.stats.stall_barrier += 1;
        } else {
            self.stats.stall_fetch += 1;
        }
        Ok(())
    }

    /// Execute one instruction functionally and trace it; returns its `MicroOp::smem` count.
    fn exec(&mut self, wid: usize, uop: MicroOp, smem: &mut FlatMemory, gmem: &mut FlatMemory) -> Result<u8, ExecErr> {
        let (wb, mem_lines) = self.warps[wid].backend(uop, &mut self.scheduler, smem, gmem)?;
        if self.trace {
            Arc::get_mut(&mut self.tracer).expect("failed to get tracer").record(wid, &wb);
            Arc::get_mut(&mut self.mem_tracer).expect("failed to get mem tracer").record(&mem_lines);
        }
        let banks = self.conf().smem.banks;
        let mut words: Vec<_> = // (bank, word) of each shared memory access
            mem_lines.iter().filter(|l| l.is_smem).map(|l| (l.address as usize / 4 % banks, l.address / 4)).collect();
        words.sort_unstable();
        words.dedup(); // lanes on the same word count once
        let smem = words.chunk_by(|a, b| a.0 == b.0).map(<[_]>::len).max().unwrap_or(0) as u8;
        if uop.inst.unit() == Unit::Lsu && smem == 0 {
            let line = self.conf().gmem.line as u32;
            let mut lines: Vec<_> = // the lines each lane's first and last byte fall in
                mem_lines.iter().flat_map(|l| [l.address / line, (l.address + l.size - 1) / line]).map(u64::from).collect();
            lines.sort_unstable();
            lines.dedup();
            self.warps[wid].gmem_lines.push_back(lines);
        }
        Ok(smem)
    }

    pub fn get_tracer_mut(&mut self) -> &mut Tracer {
        Arc::get_mut(&mut self.tracer).expect("failed to get tracer")
    }

    pub fn get_tracer(&self) -> &Tracer {
        &self.tracer
    }

    pub fn get_mem_tracer_mut(&mut self) -> &mut MemTracer {
        Arc::get_mut(&mut self.mem_tracer).expect("failed to get mem tracer")
    }

    pub fn get_mem_tracer(&self) -> &MemTracer {
        &self.mem_tracer
    }

}

module!(MuonCore, MuonState, MuonConfig,);

impl ModuleBehaviors for MuonCore {
    fn reset(&mut self) {
        self.scheduler.reset();
        self.warps.iter_mut().for_each(Warp::reset);
    }
}

