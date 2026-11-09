use crate::muon::config::MuonConfig;
use crate::muon::core::MuonCore;
use crate::muon::warp::ExecErr;
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use std::ops::Range;
use std::sync::{Arc, RwLock};

/// Cores sharing one smem; a command processor launches CTAs onto and retires them from it.
pub struct SM {
    pub config: MuonConfig,
    pub cores: Vec<MuonCore>,
    pub errs: Vec<Option<ExecErr>>, // per core: error that stopped it
    smem_used: Vec<Range<usize>>,   // allocated smem ranges, sorted by start
    pub frees: usize,               // CTAs freed so far; launchers retry placement only after one
}

impl SM {
    pub fn new(id: usize, config: MuonConfig, logger: &Arc<Logger>, gmem: Arc<RwLock<FlatMemory>>) -> Self {
        let smem = Arc::new(RwLock::new(FlatMemory::new_with_size(config.smem_size, None)));
        SM {
            config,
            cores: (0..config.num_cores)
                .map(|i| MuonCore::new(Arc::new(config), id, i, logger, gmem.clone(), smem.clone()))
                .collect(),
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

    pub fn free_smem(&mut self, range: &Range<usize>) {
        self.smem_used.retain(|used| used != range);
    }

    pub fn busy(&self) -> bool {
        self.cores.iter().zip(&self.errs).any(|(core, err)| err.is_none() && core.scheduler.active_warp_mask() != 0)
    }

    pub fn tick(&mut self, now: u64) {
        // cores with running warps; a core that errored stops until its CTAs are freed
        for (core, err) in self.cores.iter_mut().zip(self.errs.iter_mut()) {
            if err.is_none() && core.scheduler.active_warp_mask() != 0 {
                if let Err(e) = core.process(now) {
                    *err = Some(e);
                }
            }
        }
    }
}
