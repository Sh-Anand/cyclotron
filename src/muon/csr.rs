use crate::muon::config::MuonConfig;

const MISA: u32 = (1 << 30) // MXL: 32-bit
    | (1 << 5) // F - Single precision floating-point extension
    | (1 << 6) // G - Additional standard extensions present
    | (1 << 8) // I - RV32I/64I/128I base ISA
    | (1 << 12) // M - Integer Multiply/Divide extension
    | (1 << 23); // X - Non-standard extensions present

/// A warp's CSRs.  Only the lane ids and thread_idx differ across lanes; everything
/// else is per warp, per core or constant.
#[derive(Debug, Default)]
pub struct CSRFile {
    conf: MuonConfig,
    pub block_idx: (u32, u32, u32),
    pub thread_idx: Vec<(u32, u32, u32)>, // per lane
    pub bp: u32,
    pub smem_base: u32,
    pub cmd_id: u64, // launching command: stream id << 56 | per-stream command id
    fcsr: u32, // frm[7:5] | fflags[4:0]
    acc: u32,  // cisc accelerator
}

impl CSRFile {
    pub fn new(conf: MuonConfig) -> Self {
        CSRFile { thread_idx: vec![(0, 0, 0); conf.num_lanes], conf, ..Default::default() }
    }

    pub fn reset(&mut self) {
        self.fcsr = 0;
        self.acc = 0;
    }

    /// csr `addr` as lane `lane` reads it
    pub fn read(&self, addr: u32, lane: usize, tmask: u32, active_warps: u32) -> u32 {
        let c = &self.conf;
        let (wid, cid) = (c.lane_config.warp_id, c.lane_config.core_id);
        match addr {
            0x001 => self.fcsr & 0x1f, // fflags
            0x002 => self.fcsr >> 5,   // frm
            0x003 => self.fcsr,
            0xacc => self.acc,
            0x301 => MISA,
            // mvendorid, marchid, mimpid, satp, mstatus, medeleg, mideleg, mie, mtvec,
            // mepc, pmpcfg0, pmpaddr0, mpm_reserved; mcycle and minstret are not counted
            0xf11..=0xf13 | 0x180 | 0x300 | 0x302..=0x305 | 0x341 | 0x3a0 | 0x3b0 | 0xb01 | 0xb81 => 0,
            0xb00 | 0xb80 | 0xb02 | 0xb82 => 0,
            0xf14 => (c.num_lanes * c.num_warps * cid + c.num_lanes * wid + lane) as u32, // mhartid
            0xcc0 => lane as u32,
            0xcc1 => wid as u32,
            0xcc2 => cid as u32,
            0xcc3 => active_warps,
            0xcc4 => tmask,
            0xcd0 => c.lane_config.cluster_id as u32,
            0xfc0 => c.num_lanes as u32,
            0xfc1 => c.num_warps as u32,
            0xfc2 => c.num_cores as u32, // per cluster
            0xfc3 => self.block_idx.0,
            0xfc4 => self.block_idx.1,
            0xfc5 => self.block_idx.2,
            0xfc6 => self.thread_idx[lane].0,
            0xfc7 => self.thread_idx[lane].1,
            0xfc8 => self.thread_idx[lane].2,
            0xfc9 => self.bp,
            0xfca => self.smem_base,
            0xfcb => self.cmd_id as u32,
            0xfcc => (self.cmd_id >> 32) as u32,
            _ => panic!("reading nonexistent csr 0x{:x}", addr),
        }
    }

    /// write a user-writable csr; writes to read-only csrs are ignored
    pub fn write(&mut self, addr: u32, value: u32) {
        match addr {
            0x001 => self.fcsr = (self.fcsr & 0xe0) | (value & 0x1f),
            0x002 => self.fcsr = (self.fcsr & 0x1f) | ((value & 0x7) << 5),
            0x003 => self.fcsr = value & 0xff,
            0xacc => self.acc = value,
            _ => {}
        }
    }

    pub fn set_block_threads_bp(
        &mut self,
        block_idx: (u32, u32, u32),
        thread_idxs: &[(u32, u32, u32)],
        bp: u32,
        smem_base: u32,
        cmd_id: u64,
    ) {
        self.block_idx = block_idx;
        self.thread_idx[..thread_idxs.len()].copy_from_slice(thread_idxs);
        self.bp = bp;
        self.smem_base = smem_base;
        self.cmd_id = cmd_id;
    }
}
