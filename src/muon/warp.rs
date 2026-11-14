use crate::base::mem::HasMemory;
use crate::info;
use crate::muon::config::{MuonConfig, Unit};
use crate::muon::core::CtaId;
use crate::muon::csr::CSRFile;
use crate::muon::decode::{DecodeUnit, Lanes, MicroOp, RegFile};
use crate::muon::execute::ExecuteUnit;
use crate::muon::scheduler::{Schedule, Scheduler};
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::zip;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

pub struct Warp {
    conf: MuonConfig,
    pub reg_file: Vec<RegFile>, // per lane
    pub csr_file: CSRFile,
    pub wid: usize,
    pub ibuf: VecDeque<MicroOp>, // decoded, not yet issued; all but a trailing control op already executed
    pub reg_ready: Vec<u64>,     // cycle each register's pending write lands
    pub fetch_ready: u64,        // first cycle the ibuf may refill; u64::MAX while waiting at a barrier
    pub fetched: u64,            // the instruction cache line it last fetched
    pub gmem_lines: VecDeque<u64>, // the lines its decoded, unissued global memory ops touch, oldest op first
    pub store_done: u64,         // cycle its last issued global store completes
    pub load_done: u64,          // cycle the last of its issued global loads completes
    pub regs: usize,             // regs per thread allocated to this slot
    logger: Arc<Logger>,
}

impl Warp {
    pub fn reset(&mut self) {
        self.reg_file.iter_mut().for_each(|x| x.reset());
        self.csr_file.reset();
        self.ibuf.clear();
        self.reg_ready.fill(0);
        self.fetch_ready = 0;
        self.gmem_lines.clear();
        (self.store_done, self.load_done) = (0, 0);
    }
}

#[derive(Clone, Copy, Debug)]
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
    pub fn new(config: MuonConfig, logger: &Arc<Logger>) -> Warp {
        Warp {
            conf: config,
            reg_file: (0..config.num_lanes).map(|_| RegFile::new(config.num_regs)).collect(),
            csr_file: CSRFile::new(config),
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
        }
    }

    pub fn frontend(&mut self, schedule: Schedule, gmem: &FlatMemory) -> MicroOp {
        // fetch
        let inst = u64::from_le_bytes(gmem.read_n::<8>(schedule.pc as usize).expect("failed to fetch instruction"));
        // decode
        let inst = DecodeUnit::decode(inst, schedule.pc);
        MicroOp { inst, tmask: schedule.mask, smem: 0, gmem: 0 }
    }

    /// Execute `uop` functionally: read its operands, execute it, serve its memory requests and
    /// write rd.  Returns its `MicroOp::smem` and `MicroOp::gmem` counts.
    pub fn backend(
        &mut self,
        uop: MicroOp,
        scheduler: &mut Scheduler,
        smem: &mut FlatMemory,
        gmem: &mut FlatMemory,
    ) -> Result<(u8, u8), ExecErr> {
        let (pc, tmask, wid, rd_addr) = (uop.inst.pc, uop.tmask, self.wid, uop.inst.rd_addr);
        assert!(scheduler.state.thread_masks[wid] == tmask, "scheduler tmask was not set correctly");

        catch_unwind(AssertUnwindSafe(|| {
            let issued = ExecuteUnit::collect(&uop, &self.reg_file);
            let (mut rd_data, reqs) =
                ExecuteUnit::execute(&issued, wid, tmask, &mut self.reg_file, &mut self.csr_file, scheduler);
            let counts = if uop.inst.unit() == Unit::Lsu { self.mem(&mut rd_data, &reqs, smem, gmem) } else { (0, 0) };
            for (lrf, ldata) in zip(&mut self.reg_file, rd_data) {
                ldata.map(|data| lrf.write_gpr(rd_addr, data));
            }
            info!(self.logger, "[core {} warp {}] PC=0x{:08x}, rd={:3}", self.conf.lane_config.core_id, wid, pc, rd_addr);
            counts
        }))
        .map_err(|payload| ExecErr { pc, warp_id: wid, message: payload.downcast::<String>().ok().map(|s| *s) })
    }

    /// Serve an instruction's memory requests; what a lane loads is its rd data.  Returns the distinct
    /// words in its fullest shared memory bank, and how many distinct global lines it queued.
    fn mem(&mut self, rd_data: &mut Lanes<u32>, reqs: &Lanes<MemRequest>, smem: &mut FlatMemory, gmem: &mut FlatMemory) -> (u8, u8) {
        let (banks, line) = (self.conf.smem.banks, self.conf.gmem.line as u32);
        // (bank, word) of each shared memory access; the lines each global access's first and last byte fall in
        let (mut words, mut lines, mut nw, mut nl) = ([(0, 0); 32], [0u64; 64], 0, 0);
        for (rd, r) in zip(rd_data, reqs) {
            let Some(r) = r else { continue };
            assert!(rd.is_none(), "mem req generated for a lane that already has an EX rd writeback");
            *rd = ExecuteUnit::mem_writeback(r, &Self::mem_response(r, smem, gmem));
            if r.is_smem {
                words[nw] = (r.addr as usize / 4 % banks, r.addr / 4);
                nw += 1;
            } else {
                (lines[nl], lines[nl + 1]) = ((r.addr / line) as u64, ((r.addr + r.size - 1) / line) as u64);
                nl += 2;
            }
        }
        words[..nw].sort_unstable();
        lines[..nl].sort_unstable();
        // lanes on the same word, or on the same line, count once
        let distinct = |bank: &[(usize, u32)]| 1 + bank.windows(2).filter(|w| w[0] != w[1]).count();
        let smem = words[..nw].chunk_by(|a, b| a.0 == b.0).map(distinct).max().unwrap_or(0) as u8;
        let mut gmem = 0;
        for i in (0..nl).filter(|&i| smem == 0 && (i == 0 || lines[i] != lines[i - 1])) {
            self.gmem_lines.push_back(lines[i]);
            gmem += 1;
        }
        (smem, gmem)
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
        let csrs = &self.csr_file;
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
        self.csr_file.set_block_threads_bp(block_idx, thread_idxs, bp, smem_base, cmd_id);
    }

}
