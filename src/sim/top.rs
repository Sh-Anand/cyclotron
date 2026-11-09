use crate::base::behavior::*;
use crate::base::mem::HasMemory;
use crate::muon::config::MuonConfig;
use crate::muon::core::MuonCore;
use crate::muon::sm::SM;
use crate::sim::config::{MemConfig, SimConfig};
use crate::sim::elf::ElfBackedMem;
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use crate::sim::trace_db::{default_trace_db_path, TraceDb};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

pub struct Sim {
    pub config: SimConfig,
    pub top: CyclotronTop,
    pub logger: Arc<Logger>,
    trace_db: Option<Mutex<TraceDb>>,
}

impl Sim {
    fn drain_traces(&mut self) {
        let Some(trace_db) = self.trace_db.as_ref() else {
            return;
        };
        let trace_db = trace_db.lock().expect("trace db lock poisoned");

        for (core_id, core) in self.top.sm.cores.iter_mut().enumerate() {
            let num_warps = core.conf().num_warps;
            for warp_id in 0..num_warps {
                while let Some(line) = core.get_tracer_mut().consume(warp_id) {
                    trace_db.record_inst_line(0, core_id as u32, &line);
                }
            }
            while let Some(line) = core.get_mem_tracer_mut().consume() {
                trace_db.record_mem_line(0, core_id as u32, &line);
            }
        }
    }

    pub fn new(
        sim_config: SimConfig,
        muon_config: MuonConfig,
        mem_config: MemConfig,
    ) -> Sim {
        let trace_db = if sim_config.trace {
            let trace_db_path = default_trace_db_path(None, Some(sim_config.elf.as_path()));
            println!(
                "Cyclotron: writing ISA instruction trace to {}",
                trace_db_path.display()
            );
            Some(Mutex::new(TraceDb::new(&trace_db_path)))
        } else {
            None
        };
        let logger = Arc::new(Logger::new(sim_config.log_level));
        let top = CyclotronTop::new(
            Arc::new(CyclotronConfig {
                timeout: sim_config.timeout,
                elf: sim_config.elf.clone(),
                muon_config,
                mem_config,
            }),
            &logger,
        );

        let mut sim = Sim {
            config: sim_config,
            top,
            logger,
            trace_db,
        };
        sim.top.reset();
        sim
    }

    pub fn simulate(&mut self) -> Result<(), u32> {
        self.top.reset();
        for cycle in 0..self.top.timeout {
            if self.top.finished() {
                println!("simulation finished after {} cycles", cycle + 1);
                return self.check_tohost();
            }
            self.tick();
        }

        Err(0)
    }

    pub fn check_tohost(&self) -> Result<(), u32> {
        if let Some(tohost) = self.top.sm.cores[0].scheduler.tohost() {
            if tohost != 0 {
                let case = tohost >> 1;
                println!(
                    "Cyclotron: isa-test failed with tohost={}, case={}",
                    tohost, case
                );
                return Err(tohost);
            } else {
                println!("Cyclotron: isa-test passed with tohost={}", tohost);
            }
        }
        return Ok(());
    }

    /// Advances all cores by one instruction.
    pub fn tick(&mut self) {
        if self.top.finished() {
            return;
        }
        self.top.tick_one();
        self.drain_traces();
    }

    pub fn finished(&self) -> bool {
        self.top.finished()
    }
}

pub struct CyclotronConfig {
    pub timeout: u64, // TODO: use sim
    pub elf: PathBuf, // TODO: use sim
    pub muon_config: MuonConfig,
    pub mem_config: MemConfig,
}

pub struct CyclotronTop {
    scheduled: bool, // the standalone sim starts warp 0 on every core of its one SM
    cycle: u64,      // the global clock; gluon keeps its own
    pub sm: SM,
    pub timeout: u64,
    pub gmem: Arc<RwLock<FlatMemory>>,
}

impl CyclotronTop {
    pub fn new(
        config: Arc<CyclotronConfig>,
        logger: &Arc<Logger>,
    ) -> CyclotronTop {
        let elf_path = Path::new(&config.elf);
        let imem = ElfBackedMem::new(&elf_path);

        // TODO: current implementation means imem is writable, but this is true
        // in hardware too?
        let mut gmem = FlatMemory::new(Some(config.mem_config));
        gmem.copy_elf(&imem);

        let gmem = Arc::new(RwLock::new(gmem));

        CyclotronTop {
            scheduled: false,
            cycle: 0,
            sm: SM::new(0, config.muon_config, logger, gmem.clone()),
            timeout: config.timeout,
            gmem,
        }
    }

    pub fn schedule(&mut self) {
        if !self.scheduled {
            self.sm.cores.iter_mut().for_each(MuonCore::spawn_single_warp);
            self.scheduled = true;
        }
    }

    /// Load a word from gmem.
    pub fn gmem_load(&self, addr: u32) -> [u8; 4] {
        self.gmem
            .read()
            .expect("lock poisoned")
            .read_n::<4>(addr as usize)
            .expect("load failed")
    }

    /// Store a data with given size to gmem.  Doesn't support partial writes.
    pub fn gmem_store(&self, addr: u32, data: u32, size: u32) {
        let mut gmem = self.gmem.write().expect("lock poisoned");
        let data_bytes = data.to_le_bytes();
        match size {
            0 => gmem.write(addr as usize, &data_bytes[0..1]), // store byte
            1 => gmem.write(addr as usize, &data_bytes[0..2]), // store half
            2 => gmem.write(addr as usize, &data_bytes[0..4]), // store word
            _ => panic!("unimplemented store type"),
        }
        .expect("store failed");
    }

    pub fn finished(&self) -> bool {
        self.sm.cores.iter().all(MuonCore::all_warps_retired)
    }
}

impl ModuleBehaviors for CyclotronTop {
    fn tick_one(&mut self) {
        self.schedule();
        self.cycle += 1;
        self.sm.tick(self.cycle);
        if let Some(err) = self.sm.errs.iter().find_map(Option::as_ref) {
            panic!("{err:?}");
        }
    }

    fn reset(&mut self) {
        self.sm.cores.iter_mut().for_each(MuonCore::reset);
    }
}
