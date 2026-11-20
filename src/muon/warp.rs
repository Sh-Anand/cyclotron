use crate::base::mem::HasMemory;
use crate::info;
use crate::muon::config::{MuonConfig, Unit};
use crate::muon::core::CtaId;
use crate::muon::csr::CSRFile;
use crate::muon::decode::{sign_ext, DecodeUnit, DecodedInst, IssuedInst, Lanes, MicroOp, RegFile};
use crate::muon::execute::{ExecuteUnit, Opcode};
use crate::muon::scheduler::{Schedule, Scheduler};
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use crate::utils::BitSlice;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::zip;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

pub struct Warp {
    conf: MuonConfig,
    pub reg_file: RegFile,
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
    pub cta: CtaId,              // the CTA this slot's warp belongs to
    logger: Arc<Logger>,
}

impl Warp {
    pub fn reset(&mut self) {
        self.reg_file.reset();
        self.csr_file.reset();
        self.ibuf.clear();
        self.reg_ready.fill(0);
        self.fetch_ready = 0;
        self.gmem_lines.clear();
        (self.store_done, self.load_done) = (0, 0);
    }
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
            reg_file: RegFile::new(config.num_regs),
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
            cta: 0,
            logger: logger.clone(),
        }
    }

    pub fn frontend(&mut self, schedule: Schedule, gmem: &FlatMemory) -> MicroOp {
        // fetch
        let inst = u64::from_le_bytes(gmem.read_n::<8>(schedule.pc as usize).expect("failed to fetch instruction"));
        // decode
        let inst = DecodeUnit::decode(inst, schedule.pc);
        let mut rs = [0; 10];
        zip(&mut rs, inst.rs()).for_each(|(s, r)| *s = r);
        let (rd, rd_run) = (inst.rd(), inst.rd_run());
        MicroOp { inst, tmask: schedule.mask, rd, rd_run, rs, unit: inst.unit(), ready: u64::MAX, smem: 0, gmem: 0 }
    }

    /// vx_mma.bf16: D = A * B + C by the whole warp.  A is 16x16 and B 16x8 in bf16, two elements to a register
    /// with the first in its low half; C and D are 16x8 in fp32.  rs1, rs2, rs3 and rd name the first of A's
    /// four registers, B's two, C's four and D's four.  Each lane holds a share of every matrix as in Nvidia's
    /// mma.m16n8k16; with g = lane / 4 and t = lane % 4, register r of
    ///   A holds row g + 8 * (r % 2), columns 2t and 2t + 1, 8 further on for r >= 2
    ///   B holds column g, rows 2t and 2t + 1, 8 further on for r = 1
    ///   C and D holds row g + 8 * (r / 2), column 2t + r % 2
    /// An element of D is its C plus the sixteen products, each exact in fp32, added in order of k
    fn mma(&mut self, inst: &DecodedInst, tmask: u32) {
        assert!(inst.f7 & 3 == 0, "unimplemented vx_mma format {}", inst.f7 & 3);
        fn run<const N: usize>(rf: &RegFile, first: u8) -> [Lanes; N] {
            std::array::from_fn(|r| rf.read_gpr(first + r as u8))
        }
        let rf = &self.reg_file;
        let (ra, rb, rc): ([Lanes; 4], [Lanes; 2], [Lanes; 4]) = (run(rf, inst.rs1_addr), run(rf, inst.rs2_addr), run(rf, inst.rs3_addr));
        let bf16 = |word: u32, half: usize| f32::from_bits(word >> (16 * half) << 16);
        let (mut a, mut b) = ([[0f32; 16]; 16], [[0f32; 8]; 16]);
        for (lane, half) in (0..32).flat_map(|lane| [(lane, 0), (lane, 1)]) {
            let (g, t) = (lane / 4, lane % 4);
            for r in 0..4 {
                a[g + 8 * (r % 2)][2 * t + half + 8 * (r / 2)] = bf16(ra[r][lane], half);
            }
            for r in 0..2 {
                b[2 * t + half + 8 * r][g] = bf16(rb[r][lane], half);
            }
        }
        for r in 0..4 {
            let mut rd = [0; 32];
            for lane in 0..32 {
                let (row, col) = (lane / 4 + 8 * (r / 2), 2 * (lane % 4) + r % 2);
                rd[lane] = (0..16).fold(f32::from_bits(rc[r][lane]), |sum, k| sum + a[row][k] * b[k][col]).to_bits();
            }
            self.reg_file.write_gpr(inst.rd_addr + r as u8, tmask, &rd);
        }
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
            let issued = ExecuteUnit::collect(&uop.inst, &self.reg_file);
            let ((mask, rd_data), counts) = if uop.unit == Unit::Lsu {
                self.mem(&issued, tmask, smem, gmem)
            } else if uop.unit == Unit::Mma {
                self.mma(&uop.inst, tmask); // writes its four registers itself
                ((0, [0; 32]), (0, 0))
            } else {
                (ExecuteUnit::execute(&issued, wid, tmask, &mut self.csr_file, scheduler), (0, 0))
            };
            self.reg_file.write_gpr(rd_addr, mask, &rd_data);
            info!(self.logger, "[core {} warp {}] PC=0x{:08x}, rd={:3}", self.conf.lane_config.core_id, wid, pc, rd_addr);
            counts
        }))
        .map_err(|payload| ExecErr { pc, warp_id: wid, message: payload.downcast::<String>().ok().map(|s| *s) })
    }

    /// Run a load, store or AMO in each lane of `tmask`.  Returns the lanes that write rd and their data, the
    /// distinct words in its fullest shared memory bank, and how many global lines it queued: the distinct ones,
    /// or for an AMO one per lane.
    fn mem(&mut self, inst: &IssuedInst, tmask: u32, smem: &mut FlatMemory, gmem: &mut FlatMemory) -> ((u32, Lanes), (u8, u8)) {
        let (amo, store) = (inst.opcode == Opcode::AMO, inst.opcode == Opcode::STORE);
        match (inst.opcode, inst.f3, inst.opext) {
            // lb, lh, lw, ld, lbu, lhu, lwu and sb, sh, sw; .global or .shared
            (Opcode::LOAD, 0..=6, 0 | 1) | (Opcode::STORE, 0..=2, 0 | 1) => {}
            (Opcode::LOAD, ..) => unimplemented!("unknown load instruction"),
            (Opcode::STORE, ..) => unimplemented!("unknown store instruction"),
            // amoadd.w, amoswap.w
            (Opcode::AMO, ..) => assert!(inst.f7 >> 2 <= 1, "unsupported AMO funct5={:#07b}", inst.f7 >> 2),
            _ => panic!("unimplemented opcode 0x{:x}", inst.opcode),
        }
        let (shared, sext) = (inst.opext == 1 && !amo, !inst.f3.bit(2));
        let size = if amo { 4 } else { 1u32 << (inst.f3 & 3) };
        let mem = if shared { smem } else { gmem };
        let (banks, line) = (self.conf.smem.banks, self.conf.gmem.line as u32);
        // (bank, word) of each shared memory access; the lines each global access's first and last byte fall in
        let (mut words, mut lines, mut nw, mut nl) = ([(0, 0); 32], [0u64; 64], 0, 0);
        let rd = ExecuteUnit::collect_lanes(
            |lane| {
                let (rs1, rs2) = (inst.rs1_data[lane], inst.rs2_data[lane]);
                let addr = if amo { rs1 } else { rs1.wrapping_add(inst.imm32) };
                if shared {
                    words[nw] = (addr as usize / 4 % banks, addr / 4);
                    nw += 1;
                } else {
                    (lines[nl], lines[nl + 1]) = ((addr / line) as u64, ((addr + size - 1) / line) as u64);
                    nl += 2;
                }
                if store {
                    mem.write(addr as usize, &rs2.to_le_bytes()[..size as usize]).expect("store failed");
                    return None;
                }
                let word = addr as usize & !3;
                let raw = u32::from_le_bytes(mem.read_n::<4>(word).expect(if amo { "amo read failed" } else { "load failed" }));
                if amo {
                    let new = if inst.f7 >> 2 == 0 { raw.wrapping_add(rs2) } else { rs2 }; // amoadd, amoswap
                    mem.write(word, &new.to_le_bytes()).expect("amo write failed");
                }
                assert_eq!(addr >> 2, (addr + size - 1) >> 2, "misaligned load");
                let field = raw >> ((addr & 3) * 8);
                Some(match size {
                    1 if sext => sign_ext::<8>(field & 0xff) as u32, // load byte
                    1 => field & 0xff,
                    2 if sext => sign_ext::<16>(field & 0xffff) as u32, // load half
                    2 => field & 0xffff,
                    4 => raw, // load word
                    _ => panic!("unimplemented load type"),
                })
            },
            tmask,
        );
        words[..nw].sort_unstable();
        lines[..nl].sort_unstable();
        // lanes on the same word, or on the same line, count once
        let distinct = |bank: &[(usize, u32)]| 1 + bank.windows(2).filter(|w| w[0] != w[1]).count();
        let smem = words[..nw].chunk_by(|a, b| a.0 == b.0).map(distinct).max().unwrap_or(0) as u8;
        let mut gmem = 0;
        // an atomic sends each lane on its own: every lane put its line in twice, so every other entry
        for i in (0..nl).filter(|&i| smem == 0 && if amo { i % 2 == 0 } else { i == 0 || lines[i] != lines[i - 1] }) {
            self.gmem_lines.push_back(lines[i]);
            gmem += 1;
        }
        (rd, (smem, gmem))
    }

    pub fn cta(&self) -> CtaId {
        self.cta
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
