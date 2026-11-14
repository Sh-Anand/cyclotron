use crate::base::behavior::*;
use crate::base::mem::HasMemory;
use crate::base::module::{module, IsModule, ModuleBase};
use crate::info;
use crate::muon::config::MuonConfig;
use crate::muon::core::CtaId;
use crate::muon::csr::CSRFile;
use crate::muon::decode::{DecodeUnit, MicroOp, RegFile};
use crate::muon::execute::ExecuteUnit;
use crate::muon::scheduler::{Schedule, Scheduler};
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::zip;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct WarpState {
    pub reg_file: Vec<RegFile>,
    pub csr_file: CSRFile,
}

pub struct Warp {
    pub base: ModuleBase<WarpState, MuonConfig>,
    pub wid: usize,
    pub ibuf: VecDeque<MicroOp>, // decoded, not yet issued; all but a trailing control op already executed
    pub reg_ready: Vec<u64>,     // cycle each register's pending write lands
    pub fetch_ready: u64,        // first cycle the ibuf may refill; u64::MAX while waiting at a barrier
    pub fetched: u64,            // the instruction cache line it last fetched
    pub gmem_lines: VecDeque<Vec<u64>>, // per decoded, unissued global memory op: the lines it touches
    pub store_done: u64,         // cycle its last issued global store completes
    pub load_done: u64,          // cycle the last of its issued global loads completes
    pub regs: usize,             // regs per thread allocated to this slot
    logger: Arc<Logger>,
}

impl ModuleBehaviors for Warp {
    fn reset(&mut self) {
        self.state_mut().reg_file.iter_mut().for_each(|x| x.reset());
        self.state_mut().csr_file.reset();
        self.ibuf.clear();
        self.reg_ready.fill(0);
        self.fetch_ready = 0;
        self.gmem_lines.clear();
        (self.store_done, self.load_done) = (0, 0);
    }
}

module!(Warp, WarpState, MuonConfig,);

/// What the execute stage produced: rd data per lane, and the memory requests still to serve.
#[derive(Debug)]
pub struct ExWriteback {
    pub rd_addr: u8,
    pub rd_data: Vec<Option<u32>>,
    pub mem_req: Vec<Option<MemRequest>>,
}

#[derive(Clone, Debug)]
pub struct MemRequest {
    pub addr: u32,
    pub data: Option<u32>,
    pub size: u32,
    pub is_sext: bool,
    pub is_store: bool,
    pub is_smem: bool,
    pub is_amo: bool,
    pub amo_op: u8,
}

/// one lane's access to shared or global memory
#[derive(Clone, Debug)]
pub struct MemAccess {
    pub is_smem: bool,
    pub address: u32,
    pub size: u32,
}

#[derive(Clone, Debug)]
pub struct MemResponse {
    pub data: Option<[u8; 4]>,
    pub is_sext: bool,
}

#[derive(Debug, Default, Clone)]
pub struct ExecErr {
    pub pc: u32,
    pub warp_id: usize,
    pub message: Option<String>,
}

impl Warp {
    pub fn new(
        config: Arc<MuonConfig>,
        logger: &Arc<Logger>,
    ) -> Warp {
        let num_lanes = config.num_lanes;
        let mut me = Warp {
            base: ModuleBase {
                state: WarpState {
                    reg_file: (0..num_lanes)
                        .map(|i| RegFile::new(config.clone(), i))
                        .collect(),
                    csr_file: CSRFile::new(*config),
                },
                ..ModuleBase::default()
            },
            wid: config.lane_config.warp_id,
            ibuf: VecDeque::new(),
            reg_ready: vec![0; config.num_regs],
            fetch_ready: 0,
            fetched: u64::MAX,
            gmem_lines: VecDeque::new(),
            store_done: 0,
            load_done: 0,
            regs: 0,
            logger: logger.clone(),
        };
        me.init_conf(config);
        me
    }

    pub fn frontend(&mut self, schedule: Schedule, gmem: &FlatMemory) -> MicroOp {
        // fetch
        let inst = u64::from_le_bytes(gmem.read_n::<8>(schedule.pc as usize).expect("failed to fetch instruction"));
        // decode
        let inst = DecodeUnit::decode(inst, schedule.pc);
        MicroOp { inst, tmask: schedule.mask, smem: 0 }
    }

    /// Execute `uop` functionally: read its operands, execute it, serve its memory requests and
    /// write rd.  Returns the memory its lanes accessed.
    pub fn backend(
        &mut self,
        uop: MicroOp,
        scheduler: &mut Scheduler,
        smem: &mut FlatMemory,
        gmem: &mut FlatMemory,
    ) -> Result<Vec<MemAccess>, ExecErr> {
        let (pc, tmask, wid) = (uop.inst.pc, uop.tmask, self.wid);
        assert!(scheduler.state().thread_masks[wid] == tmask, "scheduler tmask was not set correctly");

        let core_id = self.conf().lane_config.core_id;
        let state = &mut self.base.state;
        let issued = ExecuteUnit::collect(&uop, &state.reg_file);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let ex = ExecuteUnit::execute(issued, core_id, wid, tmask, &mut state.reg_file, &mut state.csr_file, scheduler);
            Self::mem(ex, smem, gmem)
        }));

        match result {
            Ok((rd_addr, rd_data, accesses)) => {
                for (lrf, ldata) in zip(&mut state.reg_file, rd_data) {
                    ldata.map(|data| lrf.write_gpr(rd_addr, data));
                }
                info!(self.logger, "[core {} warp {}] PC=0x{:08x}, rd={:3}", core_id, wid, pc, rd_addr);
                Ok(accesses)
            }
            Err(payload) => Err(ExecErr { pc, warp_id: wid, message: payload.downcast::<String>().ok().map(|s| *s) }),
        }
    }

    /// Serve the memory requests of an executed instruction: its rd address, its rd data merged
    /// with what it loaded, and the memory its lanes accessed.
    fn mem(ex: ExWriteback, smem: &mut FlatMemory, gmem: &mut FlatMemory) -> (u8, Vec<Option<u32>>, Vec<MemAccess>) {
        // most instructions touch no memory
        if ex.mem_req.iter().all(Option::is_none) {
            return (ex.rd_addr, ex.rd_data, vec![]);
        }
        let (mut accesses, mut rd_mem) = (vec![], vec![]);
        for req in &ex.mem_req {
            rd_mem.push(req.as_ref().and_then(|req| {
                accesses.push(MemAccess { is_smem: req.is_smem, address: req.addr, size: req.size });
                ExecuteUnit::mem_writeback(req, &Self::mem_response(req, smem, gmem))
            }));
        }
        let rd_data = zip(ex.rd_data, rd_mem)
            .map(|(lrd_ex, lrd_mem)| {
                assert!(lrd_mem.is_none() || lrd_ex.is_none(), "mem req generated for a lane that already has an EX rd writeback");
                lrd_mem.or(lrd_ex)
            })
            .collect();
        (ex.rd_addr, rd_data, accesses)
    }

    /// Handle a per-lane memory request and generate a MemResponse.
    pub fn mem_response(mem_req: &MemRequest, smem: &mut FlatMemory, gmem: &mut FlatMemory) -> MemResponse {
        let mem = if mem_req.is_smem { smem } else { gmem };

        let addr = mem_req.addr;
        let addr_aligned = addr >> 2 << 2;
        let size = mem_req.size;

        if mem_req.is_amo {
            let operand = mem_req.data.expect("amo req missing operand");
            let old_bytes = mem.read_n::<4>(addr_aligned as usize).expect("amo read failed");
            let old_val = u32::from_le_bytes(old_bytes);
            let new_val = match mem_req.amo_op {
                0b00000 => old_val.wrapping_add(operand), // amoadd
                0b00001 => operand,                       // amoswap
                _ => panic!("unsupported AMO op {}", mem_req.amo_op),
            };
            mem.write(addr_aligned as usize, &new_val.to_le_bytes())
                .expect("amo write failed");

            MemResponse {
                data: Some(old_bytes),
                is_sext: false,
            }
        } else if mem_req.is_store {
            let store_data = mem_req.data.expect("store req missing a data field");
            let store_data_bytes = store_data.to_le_bytes();

            match size {
                1 => mem.write(addr as usize, &store_data_bytes[0..1]), // store byte
                2 => mem.write(addr as usize, &store_data_bytes[0..2]), // store half
                4 => mem.write(addr as usize, &store_data_bytes[0..4]), // store word
                _ => panic!("unimplemented store size"),
            }
            .expect("store failed");

            MemResponse {
                data: None,
                is_sext: mem_req.is_sext, // pass-through
            }
        } else {
            let load_data = mem.read_n::<4>(addr_aligned as usize).expect("load failed");

            // byte-masking will be handled in Execute::mem
            MemResponse {
                data: Some(load_data),
                is_sext: mem_req.is_sext, // pass-through
            }
        }
    }

    pub fn cta(&self) -> CtaId {
        let csrs = &self.base.state.csr_file;
        (csrs.cmd_id, csrs.block_idx)
    }

    pub fn set_block_threads_bp(
        &mut self,
        block_idx: (u32, u32, u32),
        thread_idxs: &[(u32, u32, u32)],
        bp: u32,
        smem_base: u32,
        cmd_id: u64,
    ) {
        assert!(thread_idxs.len() > 0);
        self.base.state.csr_file.set_block_threads_bp(block_idx, thread_idxs, bp, smem_base, cmd_id);
    }

}
