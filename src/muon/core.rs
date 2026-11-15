use crate::muon::cache::{self, Cache};
use crate::info;
use crate::muon::config::{LaneConfig, MuonConfig, Unit};
use crate::muon::decode::MicroOp;
use crate::muon::scheduler::Scheduler;
use crate::muon::stats::CoreStats;
use crate::muon::warp::{ExecErr, Warp};
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use crate::utils::BitSlice;
use std::sync::Arc;

/// a CTA: (launching command id, block idx), as its warps' CSRs hold
pub type CtaId = (u64, (u32, u32, u32));

pub struct MuonCore {
    conf: MuonConfig,
    last_issued: usize, // round-robin pointer for single issue
    pub scheduler: Scheduler,
    pub warps: Vec<Warp>,

    logger: Arc<Logger>,
    pub stats: CoreStats, // accumulates across CTAs; not cleared by reset
    pub arrived: Option<CtaId>, // CTA of a warp that just issued a barrier
    pub l0: usize, // this core's instruction cache
    pub l1: usize, // its SM's cache, the first level for data
    pub wake: u64, // first cycle it could issue or decode anything; a spawn or a barrier release resets it
}

impl MuonCore {
    pub fn new(
        config: MuonConfig,
        cluster_id: usize,
        core_id: usize,
        logger: &Arc<Logger>,
        l0: usize,
        l1: usize,
    ) -> Self {
        let num_warps = config.num_warps;
        let core = MuonCore {
            conf: config,
            last_issued: 0,
            scheduler: Scheduler::new(config),
            warps: (0..num_warps)
                .map(|warp_id| {
                    let lane_config = LaneConfig { warp_id, core_id, cluster_id, ..config.lane_config };
                    Warp::new(MuonConfig { lane_config, ..config }, logger)
                })
                .collect(),
            logger: logger.clone(),
            stats: CoreStats::default(),
            arrived: None,
            l0,
            l1,
            wake: 0,
        };

        info!(
            core.logger,
            "muon core {} in cluster {} instantiated!", core_id, cluster_id
        );

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
        self.wake = 0;
    }

    /// unallocated warp slots, as many as the core's free regs hold at `regs` per thread
    pub fn free_warps(&self, regs: usize) -> Vec<usize> {
        let allocated = self.scheduler.state.allocated;
        let used: usize = self.warps.iter().filter(|w| allocated.bit(w.wid)).map(|w| w.regs).sum();
        (0..self.conf.num_warps).filter(|&w| !allocated.bit(w)).take((self.conf.regs_per_core - used) / regs).collect()
    }

    /// warps in `mask` that belong to CTA `id`
    pub(crate) fn cta_warps(&self, id: CtaId, mask: u32) -> impl Iterator<Item = usize> + '_ {
        (0..self.conf.num_warps).filter(move |&w| mask.bit(w) && self.warps[w].cta() == id)
    }

    pub fn cta_live(&self, id: CtaId) -> bool {
        self.cta_warps(id, self.scheduler.active_warp_mask()).next().is_some()
    }

    /// free CTA `id`'s warp slots, stopping any still running; whether it had any here
    pub fn free_cta(&mut self, id: CtaId) -> bool {
        let mask = self.cta_warps(id, self.scheduler.state.allocated).fold(0, |m, w| m | 1 << w);
        let s = &mut self.scheduler.state;
        (s.active_warps, s.allocated) = (s.active_warps & !mask, s.allocated & !mask);
        mask != 0
    }

    /// RTL-style single issue: the next warp, round-robin, with a ready ibuf entry
    fn pick_warp(&mut self, now: u64, smem_free: u64, gmem_free: u64) -> Option<(usize, usize)> {
        let n = self.conf.num_warps;
        let (wid, i) = (1..=n)
            .map(|k| (self.last_issued + k) % n)
            .find_map(|w| self.ready_entry(w, now, smem_free, gmem_free).map(|i| (w, i)))?;
        self.last_issued = wid;
        Some((wid, i))
    }

    /// set `ready` of `warp`'s ibuf entries from index `from` on: a decode changes only the new entry's,
    /// and an issue only those of the entries younger than the one issued
    fn set_ready(warp: &mut Warp, from: usize) {
        for i in from..warp.ibuf.len() {
            let u = &warp.ibuf[i];
            let hazard = |o: &MicroOp| {
                o.rd.is_some_and(|r| u.rd == Some(r) || u.rs.contains(&r)) // WAW, RAW
                    || u.rd.is_some_and(|r| o.rs.contains(&r)) // WAR
                    || (u.unit == Unit::Lsu && o.unit == Unit::Lsu) // memory ops stay in order
            };
            let ready = if (i > 0 && u.inst.is_control()) || warp.ibuf.range(..i).any(hazard) {
                u64::MAX // control issues last
            } else {
                let regs = u.rs.iter().chain(&u.rd).map(|&r| warp.reg_ready[r]).max().unwrap_or(0); // scoreboard: RAW, WAW
                // a global memory op waits for the warp's older stores, and a store for its older loads
                let global = u.unit == Unit::Lsu && u.smem == 0;
                regs.max(if global { warp.store_done.max(if u.inst.is_store() { warp.load_done } else { 0 }) } else { 0 })
            };
            warp.ibuf[i].ready = ready;
        }
    }

    /// the cycle entry `e` can issue: one memory op at a time per SM, for shared memory and for global memory
    fn ready_at(e: &MicroOp, smem_free: u64, gmem_free: u64) -> u64 {
        e.ready.max(if e.unit != Unit::Lsu { 0 } else if e.smem > 0 { smem_free } else { gmem_free })
    }

    /// oldest ibuf entry of warp `w` that can issue at `now`
    fn ready_entry(&self, w: usize, now: u64, smem_free: u64, gmem_free: u64) -> Option<usize> {
        self.warps[w].ibuf.iter().position(|e| Self::ready_at(e, smem_free, gmem_free) <= now)
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
            && self.warps[wid].ibuf.len() < self.conf.ibuf_depth
            && !self.warps[wid].ibuf.back().is_some_and(|u| u.inst.is_control())
        {
            // entering a new line fetches it through the instruction cache; a miss stops the warp until
            // the line arrives, and the warp then keeps the line even if the cache has replaced it
            let line = self.scheduler.pc(wid) as u64 / self.conf.gmem.line as u64;
            if line != self.warps[wid].fetched {
                self.warps[wid].fetched = line;
                let ready = cache::access(caches, self.l0, line, now, false).1;
                if ready > now {
                    self.warps[wid].fetch_ready = ready;
                    break;
                }
            }
            let mut uop = self.warps[wid].frontend(self.scheduler.schedule(wid).unwrap(), gmem);
            if !uop.inst.is_control() {
                (uop.smem, uop.gmem) = self.warps[wid].backend(uop, &mut self.scheduler, smem, gmem)?;
            }
            let warp = &mut self.warps[wid];
            warp.ibuf.push_back(uop);
            Self::set_ready(warp, warp.ibuf.len() - 1);
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
        // asleep until `wake`: nothing a warp here waits on can complete before then
        let awake = now >= self.wake;
        let active = self.scheduler.active_warp_mask();
        for w in (0..self.conf.num_warps).filter(|&w| awake && active.bit(w)) {
            self.refill(w, now, smem, gmem, caches)?;
        }
        let (smem_now, gmem_now) = (*smem_free, caches[self.l1].free);
        let picked = if awake { self.pick_warp(now, smem_now, gmem_now) } else { None };
        if awake && picked.is_none() {
            self.wake = (self.warps.iter())
                .flat_map(|w| {
                    let fetch = (w.fetch_ready != u64::MAX).then_some(w.fetch_ready);
                    w.ibuf.iter().map(move |e| Self::ready_at(e, smem_now, gmem_now)).chain(fetch)
                })
                .filter(|&t| t > now)
                .min()
                .unwrap_or(u64::MAX);
        }
        if let Some((wid, i)) = picked {
            // control executes at issue; everything else already executed at decode
            let uop = self.warps[wid].ibuf.remove(i).unwrap();
            self.stats.unit_issued[uop.unit as usize] += 1;
            if uop.inst.is_control() {
                self.warps[wid].backend(uop, &mut self.scheduler, smem, gmem)?;
                if uop.inst.is_barrier() {
                    self.warps[wid].fetch_ready = u64::MAX;
                    self.arrived = Some(self.warps[wid].cta());
                }
            }
            let unit = uop.unit;
            let latency = self.conf.passes(unit) - 1 + self.conf.units.get(unit).latency;
            let done = if uop.smem > 0 {
                self.stats.smem_instrs += 1;
                self.stats.smem_reqs += uop.smem as u64;
                *smem_free = now + uop.smem as u64 * self.conf.smem.per_bank_latency;
                *smem_free + self.conf.smem.startup_latency
            } else if unit == Unit::Lsu {
                // global memory: present each line to the L1 a cycle after it accepted the previous one
                let (mut t, mut ready) = (now, now);
                self.stats.gmem_instrs += 1;
                self.stats.gmem_reqs += uop.gmem as u64;
                for _ in 0..uop.gmem {
                    let line = self.warps[wid].gmem_lines.pop_front().unwrap();
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
            if let Some(rd) = uop.rd {
                self.warps[wid].reg_ready[rd] = done;
            }
            Self::set_ready(&mut self.warps[wid], i);
        } else if self.warps.iter().any(|w| !w.ibuf.is_empty()) {
            self.stats.stall_deps += 1;
        } else if self.warps.iter().any(|w| w.fetch_ready == u64::MAX) {
            self.stats.stall_barrier += 1;
        } else {
            self.stats.stall_fetch += 1;
        }
        Ok(())
    }

}

impl MuonCore {
    pub fn reset(&mut self) {
        self.scheduler.reset();
        self.warps.iter_mut().for_each(Warp::reset);
    }
}

