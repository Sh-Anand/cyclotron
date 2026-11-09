use crate::muon::config::MuonConfig;
use crate::muon::core::{CtaId, MuonCore};
use crate::muon::warp::ExecErr;
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use std::ops::Range;
use std::sync::Arc;

/// Cores sharing one smem; a command processor launches CTAs onto and retires them from it.
pub struct SM {
    pub config: MuonConfig,
    pub cores: Vec<MuonCore>,
    pub errs: Vec<Option<ExecErr>>, // per core: error that stopped it
    smem: FlatMemory,
    smem_used: Vec<Range<usize>>,   // allocated smem ranges, sorted by start
    pub frees: usize,               // CTAs freed so far; launchers retry placement only after one
}

impl SM {
    pub fn new(id: usize, config: MuonConfig, logger: &Arc<Logger>) -> Self {
        SM {
            config,
            cores: (0..config.num_cores).map(|i| MuonCore::new(Arc::new(config), id, i, logger)).collect(),
            smem: FlatMemory::new_with_size(config.smem_size, None),
            errs: vec![None; config.num_cores],
            smem_used: vec![],
            frees: 0,
        }
    }

    // first fit: the lowest gap between allocations that holds `len` bytes
    pub fn alloc_smem(&mut self, len: usize) -> Option<Range<usize>> {
        if len == 0 {
            return Some(0..0);
        }
        let mut base = 0;
        let idx = match self.smem_used.iter().position(|used| {
            let fits = used.start - base >= len;
            if !fits {
                base = used.end;
            }
            fits
        }) {
            Some(idx) => idx,
            None if self.config.smem_size - base >= len => self.smem_used.len(),
            None => return None,
        };
        self.smem_used.insert(idx, base..base + len);
        Some(base..base + len)
    }

    /// free warp slots across cores, roomiest core first; per core as many as its free regs hold at
    /// `regs` per thread
    pub fn free_slots(&self, regs: usize) -> Vec<(usize, usize)> {
        let mut free: Vec<Vec<(usize, usize)>> = self
            .cores
            .iter()
            .enumerate()
            .map(|(c, core)| core.free_warps(regs).into_iter().map(|w| (c, w)).collect())
            .collect();
        free.sort_by_key(|slots| std::cmp::Reverse(slots.len()));
        free.concat()
    }

    /// whether any warp of CTA `id` still runs
    pub fn cta_live(&self, id: CtaId) -> bool {
        self.cores.iter().any(|core| core.cta_live(id))
    }

    /// the error that stopped a core running CTA `id`
    pub fn cta_err(&self, id: CtaId) -> Option<ExecErr> {
        self.cores.iter().zip(&self.errs).find_map(|(core, err)| err.clone().filter(|_| core.cta_live(id)))
    }

    /// free CTA `id`'s warp slots and its `smem`; cores it errored on run again
    pub fn free_cta(&mut self, id: CtaId, smem: &Range<usize>) {
        for (core, err) in self.cores.iter_mut().zip(self.errs.iter_mut()) {
            if core.free_cta(id) {
                *err = None;
            }
        }
        self.smem_used.retain(|used| used != smem);
        self.frees += 1;
    }

    pub fn busy(&self) -> bool {
        self.cores.iter().zip(&self.errs).any(|(core, err)| err.is_none() && core.scheduler.active_warp_mask() != 0)
    }

    pub fn tick(&mut self, now: u64, gmem: &mut FlatMemory) {
        // cores with running warps; a core that errored stops until its CTAs are freed
        for (core, err) in self.cores.iter_mut().zip(self.errs.iter_mut()) {
            if err.is_none() && core.scheduler.active_warp_mask() != 0 {
                if let Err(e) = core.process(now, &mut self.smem, gmem) {
                    *err = Some(e);
                }
            }
        }
    }
}
