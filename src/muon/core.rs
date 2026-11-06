use crate::base::{behavior::*, module::*};
use crate::info;
use crate::muon::config::{LaneConfig, MuonConfig};
use crate::muon::decode::MicroOp;
use crate::muon::scheduler::Scheduler;
use crate::muon::warp::{ExecErr, Warp};
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use crate::sim::trace::{MemTracer, Tracer};
use crate::utils::BitSlice;
use std::sync::{Arc, RwLock};

#[derive(Debug, Default)]
pub struct MuonState {
    last_issued: usize, // round-robin pointer for single issue
}

pub struct MuonCore {
    base: ModuleBase<MuonState, MuonConfig>,
    pub scheduler: Scheduler,
    pub warps: Vec<Warp>,
    pub(crate) shared_mem: Arc<RwLock<FlatMemory>>,

    logger: Arc<Logger>,
    tracer: Arc<Tracer>,
    mem_tracer: Arc<MemTracer>,
    pub trace: bool,
}

impl MuonCore {
    pub fn new(
        config: Arc<MuonConfig>,
        cluster_id: usize,
        core_id: usize,
        logger: &Arc<Logger>,
        gmem: Arc<RwLock<FlatMemory>>,
        shared_mem: Arc<RwLock<FlatMemory>>,
    ) -> Self {
        let num_warps = config.num_warps;
        let mut core = MuonCore {
            base: Default::default(),
            scheduler: Scheduler::new(Arc::clone(&config), core_id),
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
                        gmem.clone(),
                    )
                })
                .collect(),
            shared_mem,
            logger: logger.clone(),
            tracer: Arc::new(Tracer::new(&config)),
            mem_tracer: Arc::new(MemTracer::new()),
            trace: false,
        };

        info!(
            core.logger,
            "muon core {} in cluster {} instantiated!", core_id, cluster_id
        );

        core.init_conf(Arc::clone(&config));
        core
    }

    /// Spawn a single warp to this core.
    pub fn spawn_single_warp(&mut self) {
        self.scheduler.spawn_single_warp()
    }

    /// spawn one CTA's warps on this core; each warp names its slot
    pub fn spawn_n_warps(
        &mut self,
        pc: u32,
        block_idx: (u32, u32, u32),
        warps: &[(usize, Vec<(u32, u32, u32)>)], // (wid, thread idxs)
        bp: u32,
        smem_base: u32,
    ) {
        for (wid, thread_idxs) in warps {
            self.scheduler.spawn_warp(*wid, pc, thread_idxs.len());
            self.warps[*wid].set_block_threads_bp(block_idx, thread_idxs, bp, smem_base);
        }
    }

    // TODO: This should differentiate between different threadblocks.
    pub fn all_warps_retired(&self) -> bool {
        self.scheduler.all_warps_retired()
    }

    /// RTL-style single issue: at most one active warp per cycle, round-robin
    fn pick_warp(&mut self) -> Option<usize> {
        let n = self.conf().num_warps;
        let ready = self.scheduler.active_warp_mask();
        let wid = (1..=n).map(|i| (self.base.state.last_issued + i) % n).find(|&w| ready.bit(w))?;
        self.base.state.last_issued = wid;
        Some(wid)
    }

    /// Advance the core by one cycle: issue one instruction from one warp.
    pub fn process(&mut self) -> Result<(), ExecErr> {
        if let Some(wid) = self.pick_warp() {
            let smem = self.shared_mem.clone();
            let mut smem = smem.write().expect("shared memory lock poisoned");
            // top up the ibuf; non-control instructions execute as they are decoded
            while self.warps[wid].ibuf.len() < self.conf().ibuf_depth
                && !self.warps[wid].ibuf.back().is_some_and(|u| u.inst.is_control())
            {
                let uop = self.warps[wid].frontend(self.scheduler.schedule(wid).unwrap());
                (!uop.inst.is_control()).then(|| self.exec(wid, uop, &mut smem)).transpose()?;
                self.warps[wid].ibuf.push_back(uop);
            }
            // issue the oldest ready entry (all are ready for now); control executes here
            self.warps[wid]
                .ibuf
                .pop_front()
                .filter(|u| u.inst.is_control())
                .map(|u| self.exec(wid, u, &mut smem))
                .transpose()?;
        }
        self.scheduler.tick_one();
        self.warps.iter_mut().for_each(Warp::tick_one);
        Ok(())
    }

    /// Execute one instruction functionally and trace it.
    fn exec(&mut self, wid: usize, uop: MicroOp, smem: &mut FlatMemory) -> Result<(), ExecErr> {
        let (wb, mem_lines) = self.warps[wid].backend(uop, &mut self.scheduler, smem)?;
        if self.trace {
            Arc::get_mut(&mut self.tracer).expect("failed to get tracer").record(wid, &wb);
            Arc::get_mut(&mut self.mem_tracer).expect("failed to get mem tracer").record(&mem_lines);
        }
        Ok(())
    }

    /// Exposes a non-mutating fetch interface for the core.
    pub fn fetch(&self, warp: u32, pc: u32) -> u64 {
        // warp is not really necessary here
        self.warps[warp as usize].fetch(pc)
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

module!(
    MuonCore,
    MuonState,
    MuonConfig,
    fn get_children(&mut self) -> Vec<&mut dyn ModuleBehaviors> {
        todo!()
    }
);

impl ModuleBehaviors for MuonCore {
    fn tick_one(&mut self) {
        self.base.cycle += 1;
    }

    fn reset(&mut self) {
        self.scheduler.reset();
        self.warps.iter_mut().for_each(Warp::reset);
    }
}

impl MuonCore {
    pub fn time(&self) -> u64 {
        self.base.cycle
    }
}
