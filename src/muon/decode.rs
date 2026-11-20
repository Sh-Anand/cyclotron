use crate::muon::config::Unit;
use crate::muon::execute::Opcode;
use crate::utils::*;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug, Clone, Copy, Default)]
pub struct HasRegs {
    pub rs1: bool,
    pub rs2: bool,
    pub rs3: bool,
    pub rs4: bool,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct DecodedInst {
    pub opcode: u8,
    pub opext: u8,
    pub rd_addr: u8,
    pub f3: u8,
    pub rs1_addr: u8,
    pub rs2_addr: u8,
    pub rs3_addr: u8,
    pub rs4_addr: u8,
    pub f7: u8,
    pub imm32: u32,
    pub imm24: i32,
    pub csr_imm: u8,
    pub pc: u32,
    pub raw: u64,
}

/// one value per lane; a warp has at most 32
pub type Lanes = [u32; 32];

/// A decoded instruction with its source operands read
#[derive(Debug, Clone)]
pub struct IssuedInst {
    pub inst: DecodedInst,
    pub rs1_data: Lanes,
    pub rs2_data: Lanes,
    pub rs3_data: Lanes,
}

impl std::ops::Deref for IssuedInst {
    type Target = DecodedInst;
    fn deref(&self) -> &DecodedInst {
        &self.inst
    }
}

impl DecodedInst {
    /// changes pc/tmask/active warps or reads CSRs: ends the ibuf, executes at issue
    pub fn is_control(&self) -> bool {
        matches!(
            self.opcode,
            Opcode::BRANCH | Opcode::JAL | Opcode::JALR | Opcode::SYSTEM | Opcode::MISC_MEM | Opcode::CUSTOM0
        )
    }

    /// writes memory.  An AMO is timed as a store whose rd, the old value, comes back with it
    pub fn is_store(&self) -> bool {
        matches!(self.opcode, Opcode::STORE | Opcode::STORE_FP | Opcode::AMO)
    }

    pub fn is_amo(&self) -> bool {
        self.opcode == Opcode::AMO
    }

    pub fn is_barrier(&self) -> bool {
        self.opcode == Opcode::CUSTOM0 && self.f3 == 4 && self.f7 == 0
    }

    /// vx_launch: rs1-rs4 hold a launch entry's four words, and the rd bits the kernel's id
    pub fn is_launch(&self) -> bool {
        self.opcode == Opcode::CUSTOM0 && self.f3 == 6 && self.f7 == 0
    }

    /// vx_mma, the warp matrix multiply: rs1, rs2, rs3 and rd each name the first register of a matrix's run
    pub fn is_mma(&self) -> bool {
        self.opcode == Opcode::CUSTOM1 && self.f3 == 2
    }

    /// the functional unit this instruction issues to (branches and jumps use the ALU, as in Muon)
    pub fn unit(&self) -> Unit {
        match self.opcode {
            Opcode::OP if self.f7 == 1 && self.f3 < 4 => Unit::Mul, // mul, mulh, mulhsu, mulhu
            Opcode::OP if self.f7 == 1 => Unit::Div,                // div, divu, rem, remu
            Opcode::OP_FP | Opcode::MADD | Opcode::MSUB | Opcode::NM_SUB | Opcode::NM_ADD => {
                let divsqrt = self.opcode == Opcode::OP_FP && matches!(self.f7 >> 2, 0b00011 | 0b01011); // fdiv, fsqrt
                // .h, but for the conversions (fcvt.h.s, fcvt.w.h, fcvt.h.w), which use the fp32 lanes as in Muon
                let half = self.f7 & 3 == 0b10
                    && !(self.opcode == Opcode::OP_FP && matches!(self.f7 >> 2, 0b01000 | 0b11000 | 0b11010));
                match (half, divsqrt) {
                    (false, false) => Unit::Fp,
                    (false, true) => Unit::FpDivSqrt,
                    (true, false) => Unit::FpH,
                    (true, true) => Unit::FpHDivSqrt,
                }
            }
            Opcode::CUSTOM3 => Unit::FpExp,
            Opcode::LOAD | Opcode::LOAD_FP | Opcode::STORE | Opcode::STORE_FP | Opcode::AMO => Unit::Lsu,
            Opcode::CUSTOM0 if self.is_launch() => Unit::Launch,
            Opcode::CUSTOM1 if self.is_mma() => Unit::Mma,
            Opcode::SYSTEM | Opcode::MISC_MEM | Opcode::CUSTOM0 => Unit::Sfu,
            _ => Unit::Alu,
        }
    }

    /// destination register; none for x0 or when the rd bits are not a destination
    pub fn rd(&self) -> Option<usize> {
        let no_rd = matches!(
            self.opcode,
            Opcode::BRANCH | Opcode::STORE | Opcode::STORE_FP | Opcode::MISC_MEM | Opcode::CUSTOM0
        ) || (self.opcode == Opcode::SYSTEM && self.f3 == 0); // ecall
        (!no_rd && self.rd_addr != 0).then_some(self.rd_addr as usize)
    }

    /// how many registers it writes from rd on
    pub fn rd_run(&self) -> usize {
        if self.is_mma() { 4 } else { 1 }
    }

    /// source registers, without x0.  vx_mma reads a run from each of its three: A's four, B's two, C's four
    pub fn rs(&self) -> impl Iterator<Item = usize> {
        let h = self.has_regs();
        let runs = if self.is_mma() { [4, 2, 4, 0] } else { [1; 4] };
        [(h.rs1, self.rs1_addr), (h.rs2, self.rs2_addr), (h.rs3, self.rs3_addr), (h.rs4, self.rs4_addr)]
            .into_iter()
            .zip(runs)
            .filter(|&((used, _), _)| used)
            .flat_map(|((_, r), run)| r as usize..r as usize + run)
            .filter(|&r| r != 0)
    }

    pub fn has_regs(&self) -> HasRegs {
        match self.opcode {
            Opcode::LOAD
            | Opcode::LOAD_FP
            | Opcode::OP_IMM
            | Opcode::STORE
            | Opcode::STORE_FP
            | Opcode::OP
            | Opcode::OP_FP
            | Opcode::MADD
            | Opcode::MSUB
            | Opcode::NM_SUB
            | Opcode::NM_ADD
            | Opcode::BRANCH
            | Opcode::JALR
            | Opcode::AMO => HasRegs {
                rs1: true,
                rs2: matches!(
                    self.opcode,
                    Opcode::AMO
                        | Opcode::STORE
                        | Opcode::STORE_FP
                        | Opcode::OP
                        | Opcode::OP_FP
                        | Opcode::MADD
                        | Opcode::MSUB
                        | Opcode::NM_SUB
                        | Opcode::NM_ADD
                        | Opcode::BRANCH
                ),
                rs3: matches!(
                    self.opcode,
                    Opcode::MADD | Opcode::MSUB | Opcode::NM_SUB | Opcode::NM_ADD
                ),
                rs4: false,
            },
            Opcode::SYSTEM => HasRegs {
                rs1: matches!(self.f3, 1 | 2 | 3),
                rs2: false,
                rs3: false,
                rs4: false,
            },
            Opcode::CUSTOM0 => HasRegs {
                rs1: matches!(
                    self.sfu_key(),
                    0b000_0000000 | 0b001_0000000 | 0b010_0000000 | 0b100_0000000 | 0b101_0000000 | 0b110_0000000
                ),
                rs2: matches!(
                    self.sfu_key(),
                    0b001_0000000 | 0b100_0000000 | 0b101_0000000 | 0b110_0000000
                ),
                rs3: self.is_launch(),
                rs4: self.is_launch(),
            },
            Opcode::CUSTOM2 => HasRegs {
                rs1: matches!(
                    self.extended_opcode(),
                    Opcode::NU_INVOKE | Opcode::NU_PAYLOAD
                ),
                rs2: self.extended_opcode() == Opcode::NU_INVOKE && self.rs2_addr != 0,
                rs3: false,
                rs4: false,
            },
            Opcode::CUSTOM1 if self.is_mma() => HasRegs { rs1: true, rs2: true, rs3: true, rs4: false },
            Opcode::CUSTOM3 => HasRegs {
                rs1: true,
                rs2: false,
                rs3: false,
                rs4: false,
            },
            _ => HasRegs::default(),
        }
    }

    fn extended_opcode(&self) -> u16 {
        self.opcode as u16 | ((self.opext as u16) << 7)
    }

    fn sfu_key(&self) -> u16 {
        ((self.f3 as u16) << 7) | self.f7 as u16
    }
}

impl Display for IssuedInst {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IssuedInst: {{ pc: {:#010x} inst: {:#010x} [ op: 0x{:x}, f3: {}, f7: {}, rd: x{}, rs: [x{}, x{}, x{}], rs1_data: {:?}, rs2_data: {:?}, rs3_data: {:?} ] }}",
            self.pc, self.raw, self.opcode, self.f3, self.f7,
            self.rd_addr, self.rs1_addr, self.rs2_addr, self.rs3_addr,
            self.rs1_data, self.rs2_data, self.rs3_data
        )
    }
}

// per-warp
#[derive(Debug, Clone, Copy)]
pub struct MicroOp {
    pub inst: DecodedInst,
    pub tmask: u32,
    pub rd: Option<usize>, // `inst.rd()`
    pub rd_run: usize,     // `inst.rd_run()`: it writes this many registers from rd on
    pub rs: [usize; 10],   // `inst.rs()`, then 0s; 0 is never a destination
    pub unit: Unit,        // `inst.unit()`
    pub ready: u64,        // cycle it can issue once its memory is free; u64::MAX while an older ibuf entry blocks it
    pub smem: u8, // shared memory request: distinct words in its fullest bank; 0 if not one
    pub gmem: u8, // global memory request: its lines waiting in `Warp::gmem_lines`
}

impl MicroOp {
    /// the registers it writes
    pub fn dests(&self) -> std::ops::Range<usize> {
        self.rd.map_or(0..0, |rd| rd..rd + self.rd_run)
    }

    /// the registers it reads
    pub fn srcs(&self) -> impl Iterator<Item = usize> + '_ {
        self.rs.iter().copied().take_while(|&r| r != 0)
    }
}

impl std::fmt::Display for DecodedInst {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DecodedInst: {{ inst {:#010x} [ op: 0x{:x}, f3: {}, f7: {}, rd: x{}, rs: [x{}, x{}, x{}, x{}] ] }}",
            self.raw,
            self.opcode,
            self.f3,
            self.f7,
            self.rd_addr,
            self.rs1_addr,
            self.rs2_addr,
            self.rs3_addr,
            self.rs4_addr
        )
    }
}

pub fn sign_ext<const W: u8>(from: u32) -> i32 {
    assert!(
        W <= 32,
        "cannot extend a two's complement number that is more than 32 bits"
    );
    ((from << (32 - W)) as i32) >> (32 - W)
}

/// A warp's registers: each register's value in every lane.
#[derive(Debug)]
pub struct RegFile {
    gpr: Vec<Lanes>,
    num_regs: usize,
}

impl RegFile {
    pub fn new(num_regs: usize) -> RegFile {
        assert!(num_regs <= 256, "currently at most 256 regs due to addr size");
        RegFile { gpr: vec![[0; 32]; 256], num_regs }
    }

    pub fn reset(&mut self) {
        self.gpr.fill([0; 32]);
    }

    pub fn read_gpr(&self, addr: u8) -> Lanes {
        self.gpr[addr as usize] // x0 is never written
    }

    /// write `data` to register `addr` in the lanes of `mask`
    pub fn write_gpr(&mut self, addr: u8, mask: u32, data: &Lanes) {
        assert!(mask == 0 || (addr as usize) < self.num_regs, "invalid gpr value {}", addr);
        if addr == 0 {
            return;
        }
        for lane in (0..32).filter(|&l| mask.bit(l)) {
            self.gpr[addr as usize][lane] = data[lane];
        }
    }
}

#[derive(Debug)]
pub struct DecodeUnit;

impl DecodeUnit {
    pub fn decode(inst: u64, pc: u32) -> DecodedInst {
        let _pred: u8 = inst.sel(63, 60) as u8;
        let rs1_addr: u8 = inst.sel(27, 20) as u8;
        let rs2_addr: u8 = inst.sel(35, 28) as u8;
        let rs3_addr: u8 = inst.sel(43, 36) as u8;
        let rs4_addr: u8 = inst.sel(51, 44) as u8;

        let csr_imm: u8 = rs1_addr; // duplicate, because rs1 can be renamed

        let opcode: u8 = inst.sel(6, 0) as u8;
        let is_sb_type: bool = (opcode == Opcode::BRANCH) || (opcode == Opcode::STORE);

        let imm24: i32 = sign_ext::<24>(inst.sel(59, 36) as u32);
        let uimm32: u32 = ((if is_sb_type {
            inst.sel(16, 9)
        } else {
            inst.sel(35, 28)
        } as u32)
            << 24)
            | (inst.sel(59, 36) as u32);

        let _imm12_1: i32 = sign_ext::<12>(inst.sel(47, 36) as u32);
        let _imm12_2: i32 = sign_ext::<12>(inst.sel(59, 48) as u32);

        DecodedInst {
            opcode,
            opext: inst.sel(8, 7) as u8,
            rd_addr: inst.sel(16, 9) as u8,
            f3: inst.sel(19, 17) as u8,
            rs1_addr,
            rs2_addr,
            rs3_addr,
            rs4_addr,
            f7: inst.sel(58, 52) as u8,
            imm32: uimm32,
            imm24,
            csr_imm,
            pc,
            raw: inst,
        }
    }
}
