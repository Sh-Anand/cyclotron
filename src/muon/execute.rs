use crate::muon::csr::CSRFile;
use crate::muon::decode::{sign_ext, IssuedInst, Lanes, MicroOp, RegFile};
use crate::muon::scheduler::Scheduler;
use crate::utils::BitSlice;
use half::bf16;
use num_derive::FromPrimitive;
use num_traits::ToPrimitive;
pub use num_traits::WrappingAdd;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct Opcode;
impl Opcode {
    pub const LOAD: u8 = 0b0000011u8;
    pub const LOAD_FP: u8 = 0b0000111u8;
    pub const CUSTOM0: u8 = 0b0001011u8;
    pub const MISC_MEM: u8 = 0b0001111u8;
    pub const OP_IMM: u8 = 0b0010011u8;
    pub const AUIPC: u8 = 0b0010111u8;
    //  pub const OpImm32 : u8 = 0b0011011u8;
    pub const STORE: u8 = 0b0100011u8;
    pub const STORE_FP: u8 = 0b0100111u8;
    pub const CUSTOM1: u8 = 0b0101011u8;
    pub const AMO: u8 = 0b0101111u8;
    pub const OP: u8 = 0b0110011u8;
    pub const LUI: u8 = 0b0110111u8;
    pub const OP32: u8 = 0b0111011u8;
    pub const MADD: u8 = 0b1000011u8;
    pub const MSUB: u8 = 0b1000111u8;
    pub const NM_SUB: u8 = 0b1001011u8;
    pub const NM_ADD: u8 = 0b1001111u8;
    pub const OP_FP: u8 = 0b1010011u8;
    //  pub const OpV     : u8 = 0b1010111u8;
    pub const CUSTOM2: u8 = 0b1011011u8;
    pub const BRANCH: u8 = 0b1100011u8;
    pub const JALR: u8 = 0b1100111u8;
    pub const JAL: u8 = 0b1101111u8;
    pub const SYSTEM: u8 = 0b1110011u8;
    pub const CUSTOM3: u8 = 0b1111011u8;

    pub const NU_INVOKE: u16 = 0b001011011u16;
    pub const NU_INVOKE_IMM: u16 = 0b001111011u16;
    pub const NU_PAYLOAD: u16 = 0b011011011u16;
    pub const NU_COMPLETE: u16 = 0b101011011u16;
}

macro_rules! f3_f7_mask {
    ($f3:expr, $f7:expr) => {
        ((($f3 as u16) << 7) | ($f7 as u16))
    };
}

#[derive(Debug, FromPrimitive, Clone, Copy)]
pub enum SFUType {
    TMC = 0,
    WSPAWN = 1,
    SPLIT = 2,
    JOIN = 3,
    BAR = 4,
    PRED = 5,
    KILL = 6,
    ECALL = 7,
}

#[derive(Debug, FromPrimitive, Clone, Copy, PartialEq)]
pub enum CSRType {
    RW = 1,
    RS = 2,
    RC = 3,
    RWI = 5,
    RSI = 6,
    RCI = 7,
}

#[derive(Debug)]
pub struct ExecuteUnit;

impl ExecuteUnit {
    /// look up the op once per instruction; the returned closure executes it for one lane
    pub fn alu(issued: &IssuedInst) -> impl Fn(usize) -> Option<u32> + '_ {
        fn check_zero(b: u32) -> bool {
            if b == 0 {
                // we should not panic - real riscv returns -1 on division by 0
                eprintln!("divide by zero");
                true
            } else {
                false
            }
        }

        const fn check_overflow(a: u32, b: i32) -> i32 {
            if (a == 0x8000_0000u32) && (b == -1) {
                1
            } else {
                b
            }
        }

        fn op_insts(key: u16) -> Option<fn([u32; 2]) -> u32> {
            let op: fn([u32; 2]) -> u32 = match key {
                0b000_0000000u16 => |[a, b]| { a.wrapping_add(b) }, // add
                0b000_0100000u16 => |[a, b]| { (a as i32).wrapping_sub(b as i32) as u32 }, // sub
                0b001_0000000u16 => |[a, b]| { a << (b & 31) }, // sll
                0b010_0000000u16 => |[a, b]| { if (a as i32) < (b as i32) { 1 } else { 0 } }, // slt
                0b011_0000000u16 => |[a, b]| { if a < b { 1 } else { 0 } }, // sltu
                0b100_0000000u16 => |[a, b]| { a ^ b }, // xor
                0b101_0000000u16 => |[a, b]| { a >> (b & 31) }, // srl
                0b101_0100000u16 => |[a, b]| { ((a as i32) >> (b & 31)) as u32 }, // sra
                0b110_0000000u16 => |[a, b]| { a | b }, // or
                0b111_0000000u16 => |[a, b]| { a & b }, // and
                0b000_0000001u16 => |[a, b]| { a.wrapping_mul(b) }, // mul
                0b001_0000001u16 => |[a, b]| { ((((a as i32) as i64).wrapping_mul((b as i32) as i64)) >> 32) as u32 }, // mulh
                0b010_0000001u16 => |[a, b]| { ((((a as i32) as i64).wrapping_mul((b as u64) as i64)) >> 32) as u32 }, // mulhsu
                0b011_0000001u16 => |[a, b]| { (((a as u64).wrapping_mul(b as u64)) >> 32) as u32 }, // mulhu
                0b100_0000001u16 => |[a, b]| { if check_zero(b) { u32::MAX } else { ((a as i32) / check_overflow(a, b as i32)) as u32 } }, // div
                0b101_0000001u16 => |[a, b]| { if check_zero(b) { u32::MAX } else { a / b } }, // divu
                0b110_0000001u16 => |[a, b]| { if check_zero(b) { a } else { ((a as i32) % check_overflow(a, b as i32)) as u32 } }, // rem
                0b111_0000001u16 => |[a, b]| { if check_zero(b) { a } else { a % b } }, // remu
                _ => return None,
            };
            Some(op)
        }

        fn opimm_f3_insts(key: u8) -> Option<fn([u32; 2]) -> u32> {
            let op: fn([u32; 2]) -> u32 = match key {
                0u8 => |[a, b]| { a.wrapping_add(b) }, // addi
                2u8 => |[a, b]| { if (a as i32) < (b as i32) { 1 } else { 0 } }, // slti
                3u8 => |[a, b]| { if a < b { 1 } else { 0 } }, // sltiu
                4u8 => |[a, b]| { a ^ b }, // xori
                6u8 => |[a, b]| { a | b }, // ori
                7u8 => |[a, b]| { a & b }, // andi
                _ => return None,
            };
            Some(op)
        }

        fn opimm_f3f7_insts(key: u16) -> Option<fn([u32; 2]) -> u32> {
            let op: fn([u32; 2]) -> u32 = match key {
                0b001_0000000u16 => |[a, b]| { a << (b & 31) }, // slli
                0b101_0000000u16 => |[a, b]| { a >> (b & 31) }, // srli
                0b101_0100000u16 => |[a, b]| { ((a as i32) >> (b & 31)) as u32 }, // srai
                _ => return None,
            };
            Some(op)
        }

        let op: Option<fn([u32; 2]) -> u32> = match issued.opcode {
            Opcode::OP => op_insts(f3_f7_mask!(issued.f3, issued.f7)),
            Opcode::OP_IMM => {
                opimm_f3_insts(issued.f3).or_else(|| opimm_f3f7_insts(f3_f7_mask!(issued.f3, issued.f7)))
            }
            Opcode::AUIPC => Some(|[a, b]| a.wrapping_add(b)),
            Opcode::LUI => Some(|[a, _]| a << 12),
            _ => panic!("unreachable"),
        };

        move |lane| {
            op.map(|op| {
                let operands = match issued.opcode {
                    Opcode::OP => [issued.rs1_data[lane], issued.rs2_data[lane]],
                    Opcode::OP_IMM => [issued.rs1_data[lane], issued.imm32],
                    Opcode::AUIPC => [issued.pc, issued.imm32],
                    _ => [issued.imm32, 0], // LUI
                };
                op(operands)
            })
        }
    }

    pub fn fpu(issued_inst: &IssuedInst, lane: usize) -> Option<u32> {
        fn fp32_op(a: u32, b: u32, op: fn(f32, f32) -> f32) -> u32 {
            let result = op(f32::from_bits(a), f32::from_bits(b)).to_bits();
            // info!("result of the fp operation is {:08x}", result);
            if [0xffc00000, 0x7fffffff, 0xffffffff].contains(&result) {
                0x7fc00000 // risc-v only ever generates this qNaNf
            } else {
                result
            }
        }

        fn bf16_op(a: u32, b: u32, op: fn(bf16, bf16) -> bf16) -> u32 {
            let result = op(
                bf16::from_bits((a & 0xffff) as u16),
                bf16::from_bits((b & 0xffff) as u16),
            )
            .to_bits();
            if [0xffc0, 0x7fff, 0xffff].contains(&result) {
                0x7fc0 // canonical qNaN in bfloat16 encoding
            } else {
                sign_ext::<16>(result as u32) as u32
            }
        }

        fn fp32_op_3(a: u32, b: u32, c: u32, op: fn(f32, f32, f32) -> f32) -> u32 {
            let result = op(f32::from_bits(a), f32::from_bits(b), f32::from_bits(c)).to_bits();
            if [0xffc00000, 0x7fffffff, 0xffffffff].contains(&result) {
                0x7fc00000 // risc-v only ever generates this qNaNf
            } else {
                result
            }
        }

        fn bf16_op_3(a: u32, b: u32, c: u32, op: fn(bf16, bf16, bf16) -> bf16) -> u32 {
            let result = op(
                bf16::from_bits((a & 0xffff) as u16),
                bf16::from_bits((b & 0xffff) as u16),
                bf16::from_bits((c & 0xffff) as u16),
            )
            .to_bits();
            if [0xffc0, 0x7fff, 0xffff].contains(&result) {
                0x7fc0 // canonical qNaN in bfloat16 encoding
            } else {
                sign_ext::<16>(result as u32) as u32
            }
        }

        fn fp32_cmp_op(a: u32, b: u32, op: fn(f32, f32) -> bool) -> u32 {
            op(f32::from_bits(a), f32::from_bits(b)) as u32
        }

        fn fp32_fsgn_op(a: u32, b: u32, op: fn(f32, u32) -> f32) -> u32 {
            op(f32::from_bits(a), b).to_bits()
        }

        // NOTE: this really should just be the f32::minimum/maximum functions, but
        // they are unstable. we should swap that in when that goes into stable.
        fn fp32_fminmax(a: f32, b: f32, min: bool) -> f32 {
            if a.is_nan() {
                return b;
            }
            if b.is_nan() {
                return a;
            }
            if (a == 0f32) && (b == 0f32) {
                if (min && a.is_sign_negative()) || (!min && a.is_sign_positive()) {
                    a
                } else {
                    b
                }
            } else {
                if min {
                    a.min(b)
                } else {
                    a.max(b)
                }
            }
        }

        fn fp32_fcvt_saturate(a: u32, unsigned: bool, rm: u32) -> u32 {
            // round per the instruction's rm; dyn (7) would read frm, which stays at rne
            let f = f32::from_bits(a);
            let f = match rm {
                1 => f.trunc(),          // rtz
                2 => f.floor(),          // rdn
                3 => f.ceil(),           // rup
                4 => f.round(),          // rmm: ties away from zero
                _ => f.round_ties_even(), // rne, dyn
            };
            if unsigned {
                f.to_u32()
                    .unwrap_or_else(|| if f < 0f32 { 0u32 } else { u32::MAX })
            } else {
                f.to_i32().unwrap_or_else(|| {
                    if f < i32::MIN as f32 {
                        i32::MIN
                    } else {
                        i32::MAX
                    }
                }) as u32
            }
        }

        fn fp32_fclass(a: u32) -> u32 {
            let f = f32::from_bits(a);
            let conds = [
                f == f32::NEG_INFINITY,            // 0
                f < 0f32 && f.is_normal(),         // 1
                f < 0f32 && f.is_subnormal(),      // 2
                f == 0f32 && f.is_sign_negative(), // 3
                f == 0f32 && f.is_sign_positive(), // 4
                f > 0f32 && f.is_subnormal(),      // 5
                f > 0f32 && f.is_normal(),         // 6
                f == f32::INFINITY,                // 7
                a == 0x7f800001,                   // sNaNf           // 8
                a == 0x7fc00000,                   // qNaNf           // 9
            ];
            (1 << conds.iter().position(|&c| c).unwrap()) as u32
        }

        fn opfp_f3f7_insts(key: u16) -> Option<fn([u32; 2]) -> u32> {
            let op: fn([u32; 2]) -> u32 = match key {
                0b000_0010000u16 => |[a, b]| { fp32_fsgn_op(a, b, |x, y| { if y.bit(31) { -x.abs() } else { x.abs() } }) }, // fsgnj.s
                0b001_0010000u16 => |[a, b]| { fp32_fsgn_op(a, b, |x, y| { if y.bit(31) { x.abs() } else { -x.abs() } }) }, // fsgnjn.s
                0b010_0010000u16 => |[a, b]| { fp32_fsgn_op(a, b, |x, y| { if y.bit(31) { -1.0 * x } else { x } }) }, // fsgnjx.s
                0b000_0010100u16 => |[a, b]| { fp32_op(a, b, |x, y| { fp32_fminmax(x, y, true) }) }, // fmin.s
                0b001_0010100u16 => |[a, b]| { fp32_op(a, b, |x, y| { fp32_fminmax(x, y, false) }) }, // fmax.s
                0b000_0010110u16 => |[a, b]| { bf16_op(a, b, |x, y| { if x.to_f32() < y.to_f32() { x } else { y } }) }, // fmin.h
                0b001_0010110u16 => |[a, b]| { bf16_op(a, b, |x, y| { if x.to_f32() > y.to_f32() { x } else { y } }) }, // fmax.h
                0b010_1010000u16 => |[a, b]| { fp32_cmp_op(a, b, |x, y| { x == y }) }, // feq.s
                0b001_1010000u16 => |[a, b]| { fp32_cmp_op(a, b, |x, y| { x < y }) }, // flt.s
                0b000_1010000u16 => |[a, b]| { fp32_cmp_op(a, b, |x, y| { x <= y }) }, // fle.s
                0b001_1110000u16 => |[a, _b]| { fp32_fclass(a) }, // fclass.s
                _ => return None,
            };
            Some(op)
        }

        fn opfp_f7_fp32_insts(key: u8) -> Option<fn([u32; 4]) -> u32> {
            let op: fn([u32; 4]) -> u32 = match key {
                0b0000000u8 => |[a, b, _rs2_addr, _rm]| { fp32_op(a, b, |x, y| { x + y }) }, // fadd.s
                0b0000100u8 => |[a, b, _rs2_addr, _rm]| { fp32_op(a, b, |x, y| { x - y }) }, // fsub.s
                0b0001000u8 => |[a, b, _rs2_addr, _rm]| { fp32_op(a, b, |x, y| { x * y }) }, // fmul.s
                0b0001100u8 => |[a, b, _rs2_addr, _rm]| { fp32_op(a, b, |x, y| { x / y }) }, // fdiv.s
                0b0101100u8 => |[a, b, _rs2_addr, _rm]| { fp32_op(a, b, |x, _y| { x.sqrt() }) }, // fsqrt.s
                0b1100000u8 => |[a, _b, rs2_addr, rm]| { fp32_fcvt_saturate(a, rs2_addr > 0, rm) }, // fcvt.*.s
                0b1101000u8 => |[a, _b, rs2_addr, _rm]| { if rs2_addr > 0 { f32::to_bits(a as f32) } else { f32::to_bits(a as i32 as f32) } }, // fcvt.s.*
                _ => return None,
            };
            Some(op)
        }

        fn opfp_f7_bf16_insts(key: u8) -> Option<fn([u32; 4]) -> u32> {
            let op: fn([u32; 4]) -> u32 = match key {
                0b0000000u8 => |[a, b, _rs2_addr, _rm]| { bf16_op(a, b, |x, y| { x + y }) }, // fadd.h
                0b0000100u8 => |[a, b, _rs2_addr, _rm]| { bf16_op(a, b, |x, y| { x - y }) }, // fsub.h
                0b0001000u8 => |[a, b, _rs2_addr, _rm]| { bf16_op(a, b, |x, y| { x * y }) }, // fmul.h
                0b0001100u8 => |[a, b, _rs2_addr, _rm]| { bf16_op(a, b, |x, y| { x / y }) }, // fdiv.h
                0b0101100u8 => |[a, b, _rs2_addr, _rm]| { bf16_op(a, b, |x, _y| { bf16::from_f32(x.to_f32().sqrt()) }) }, // fsqrt.h
                _ => return None,
            };
            Some(op)
        }

        let fmt = issued_inst.f7 & 0b11;
        let f7_base = issued_inst.f7 & !0b11;
        let rd_data = match issued_inst.opcode {
            Opcode::OP_FP => opfp_f3f7_insts(f3_f7_mask!(issued_inst.f3, issued_inst.f7))
                .map(|op| op([issued_inst.rs1_data[lane], issued_inst.rs2_data[lane]]))
                .or_else(|| {
                    let op = if fmt == 0b10 { opfp_f7_bf16_insts(f7_base) } else { opfp_f7_fp32_insts(f7_base) };
                    op.map(|op| {
                        op([
                            issued_inst.rs1_data[lane],
                            issued_inst.rs2_data[lane],
                            issued_inst.rs2_addr as u32,
                            issued_inst.f3 as u32,
                        ])
                    })
                }),
            Opcode::MADD | Opcode::MSUB | Opcode::NM_ADD | Opcode::NM_SUB => {
                let op: fn([u32; 3]) -> u32 = if fmt == 0b10 {
                    match issued_inst.opcode {
                        Opcode::MADD => |[a, b, c]| {
                            bf16_op_3(a, b, c, |x, y, z| {
                                bf16::from_f32(x.to_f32().mul_add(y.to_f32(), z.to_f32()))
                            })
                        },
                        Opcode::MSUB => |[a, b, c]| {
                            bf16_op_3(a, b, c, |x, y, z| {
                                bf16::from_f32(x.to_f32().mul_add(y.to_f32(), -z.to_f32()))
                            })
                        },
                        Opcode::NM_ADD => |[a, b, c]| {
                            bf16_op_3(a, b, c, |x, y, z| {
                                bf16::from_f32((-x.to_f32()).mul_add(y.to_f32(), -z.to_f32()))
                            })
                        },
                        Opcode::NM_SUB => |[a, b, c]| {
                            bf16_op_3(a, b, c, |x, y, z| {
                                bf16::from_f32((-x.to_f32()).mul_add(y.to_f32(), z.to_f32()))
                            })
                        },
                        _ => panic!(),
                    }
                } else {
                    match issued_inst.opcode {
                        Opcode::MADD => |[a, b, c]| {
                            fp32_op_3(a, b, c, |x, y, z| f32::mul_add(x, y, z))
                        },
                        Opcode::MSUB => |[a, b, c]| {
                            fp32_op_3(a, b, c, |x, y, z| f32::mul_add(x, y, -z))
                        },
                        Opcode::NM_ADD => |[a, b, c]| {
                            fp32_op_3(a, b, c, |x, y, z| f32::mul_add(-x, y, -z))
                        },
                        Opcode::NM_SUB => |[a, b, c]| {
                            fp32_op_3(a, b, c, |x, y, z| f32::mul_add(-x, y, z))
                        },
                        _ => panic!(),
                    }
                };
                Some(op([
                    issued_inst.rs1_data[lane],
                    issued_inst.rs2_data[lane],
                    issued_inst.rs3_data[lane],
                ]))
            }
            _ => {
                panic!("unreachable");
            }
        };

        rd_data
    }

    /// Returns Some(target PC) if branch is taken, None otherwise.
    pub fn branch(issued_inst: &IssuedInst, lane: usize) -> Option<u32> {
        let rs1 = issued_inst.rs1_data[lane];
        let rs2 = issued_inst.rs2_data[lane];
        let branch_offset = issued_inst.imm32;
        let branch_target = issued_inst.pc.wrapping_add(branch_offset);
        let taken = match issued_inst.f3 {
            0b000 => rs1 == rs2,                   // beq
            0b001 => rs1 != rs2,                   // bne
            0b100 => (rs1 as i32) < (rs2 as i32),  // blt
            0b101 => (rs1 as i32) >= (rs2 as i32), // bge
            0b110 => rs1 < rs2,                    // bltu
            0b111 => rs1 >= rs2,                   // bgeu
            _ => panic!("unimplemented branch instruction"),
        };
        taken.then_some(branch_target)
    }

    /// every active lane reads the old value; the first active lane's operand updates the csr
    pub fn csr(
        issued_inst: &IssuedInst,
        tmask: u32,
        csrf: &mut CSRFile,
        active_warps: u32,
    ) -> (u32, Lanes) {
        let csr_type = match issued_inst.f3 {
            1 => CSRType::RW,  // csrrw
            2 => CSRType::RS,  // csrrs
            3 => CSRType::RC,  // csrrc
            5 => CSRType::RWI, // csrrwi
            6 => CSRType::RSI, // csrrsi
            7 => CSRType::RCI, // csrrci
            _ => panic!("unimplemented"),
        };
        let first_lid = tmask.trailing_zeros() as usize;
        let operand = match csr_type {
            CSRType::RW | CSRType::RS | CSRType::RC => issued_inst.rs1_data[first_lid],
            CSRType::RWI | CSRType::RSI | CSRType::RCI => issued_inst.csr_imm as u32,
        };
        let csrr = matches!(csr_type, CSRType::RS | CSRType::RSI) && operand == 0;
        let addr = issued_inst.imm32;
        if [0xcc3, 0xcc4].contains(&addr) && !csrr {
            panic!("unimplemented thread mask write using csr");
        }
        let rds = Self::collect_lanes(|lane| Some(csrf.read(addr, lane, tmask, active_warps)), tmask);
        let old = csrf.read(addr, first_lid, tmask, active_warps);
        csrf.write(addr, match csr_type {
            CSRType::RW | CSRType::RWI => operand,
            CSRType::RS | CSRType::RSI => old | operand,
            CSRType::RC | CSRType::RCI => old & !operand,
        });
        rds
    }

    pub fn sfu(
        issued_inst: &IssuedInst,
        wid: usize,
        first_lid: usize,
        num_lanes: usize,
        scheduler: &mut Scheduler,
    ) {
        let sfu_type = if issued_inst.opcode == Opcode::SYSTEM {
            SFUType::ECALL // tohost: signals the result of a test, used only for isa tests
        } else {
            match f3_f7_mask!(issued_inst.f3, issued_inst.f7) {
                // sets thread mask to rs1[NT-1:0]
                0b000_0000000 => SFUType::TMC,
                // spawns rs1 warps, except the executing warp, and set their pc's to rs2
                0b001_0000000 => SFUType::WSPAWN,
                // collect rs1[0] for then mask. set tmc, push else mask to ipdom. skip pushing else mask when non-divergent
                0b010_0000000 => SFUType::SPLIT,
                // pop ipdom and set tmc
                0b011_0000000 => SFUType::JOIN,
                // rs1 indicates barrier id, rs2 indicates num warps participating in each core
                0b100_0000000 => SFUType::BAR,
                // sets thread mask to current tmask & then_mask, same rules as split. if no lanes take branch, set mask to rs2.
                0b101_0000000 => SFUType::PRED,
                _ => panic!("unimplemented sfu instruction"),
            }
        };

        scheduler.sfu(
            wid,
            first_lid,
            sfu_type,
            issued_inst,
            issued_inst.rs1_data[..num_lanes].to_vec(),
            issued_inst.rs2_data[..num_lanes].to_vec(),
        )
    }

    pub fn custom3(issued_inst: &IssuedInst, lane: usize) -> Option<u32> {
        match f3_f7_mask!(issued_inst.f3, issued_inst.f7) {
            // fexp.h
            0b111_0101110 => {
                let a = issued_inst.rs1_data[lane];
                let result = bf16::from_f32(bf16::from_bits((a & 0xffff) as u16).to_f32().exp()).to_bits();
                Some(if [0xffc0, 0x7fff, 0xffff].contains(&result) { 0x7fc0 } else { sign_ext::<16>(result as u32) as u32 })
            }
            _ => panic!("unimplemented custom3 instruction"),
        }
    }

    /// Collect source operand values from the regfile.
    pub fn collect(ibuf: &MicroOp, rf: &RegFile) -> IssuedInst {
        let decoded = ibuf.inst;
        IssuedInst {
            opcode: decoded.opcode,
            opext: decoded.opext,
            rd_addr: decoded.rd_addr,
            f3: decoded.f3,
            rs1_addr: decoded.rs1_addr,
            rs2_addr: decoded.rs2_addr,
            rs3_addr: decoded.rs3_addr,
            rs4_addr: decoded.rs4_addr,
            rs1_data: rf.read_gpr(decoded.rs1_addr),
            rs2_data: rf.read_gpr(decoded.rs2_addr),
            rs3_data: rf.read_gpr(decoded.rs3_addr),
            f7: decoded.f7,
            imm32: decoded.imm32,
            imm24: decoded.imm24,
            csr_imm: decoded.csr_imm,
            pc: decoded.pc,
            raw: decoded.raw,
        }
    }

    /// `func` for each lane of `tmask`: the lanes it gave a value for, and those values
    #[inline]
    pub fn collect_lanes(mut func: impl FnMut(usize) -> Option<u32>, tmask: u32) -> (u32, Lanes) {
        let (mut mask, mut data) = (0, [0; 32]);
        for lane in (0..32).filter(|&l| tmask.bit(l)) {
            if let Some(d) = func(lane) {
                mask |= 1 << lane;
                data[lane] = d;
            }
        }
        (mask, data)
    }

    /// Execute one instruction other than a load, store or AMO: the lanes that write rd, and their rd data.
    pub fn execute(
        issued: &IssuedInst,
        wid: usize,
        tmask: u32,
        num_lanes: usize,
        csrf: &mut CSRFile,
        scheduler: &mut Scheduler,
    ) -> (u32, Lanes) {
        let first_lid = tmask.trailing_zeros() as usize; // lane id of first active thread
        let empty = (0, [0; 32]);

        match issued.opcode {
            Opcode::OP | Opcode::OP_IMM | Opcode::LUI | Opcode::AUIPC => Self::collect_lanes(ExecuteUnit::alu(issued), tmask),
            Opcode::OP_FP | Opcode::MADD | Opcode::MSUB | Opcode::NM_ADD | Opcode::NM_SUB => {
                Self::collect_lanes(|lane| ExecuteUnit::fpu(issued, lane), tmask)
            }
            Opcode::BRANCH => {
                if let Some(target) = ExecuteUnit::branch(issued, first_lid) {
                    scheduler.take_branch(wid, target);
                }
                empty
            }
            Opcode::JAL => {
                scheduler.take_branch(wid, issued.pc.wrapping_add(issued.imm32));
                Self::collect_lanes(|_| Some(issued.pc + 8), tmask)
            }
            Opcode::JALR => {
                let target = issued.rs1_data[first_lid].wrapping_add(issued.imm32);
                scheduler.take_branch(wid, target);
                Self::collect_lanes(|_| Some(issued.pc + 8), tmask)
            }
            Opcode::MISC_MEM => {
                // fence, fence.i, fence.s
                if issued.f3 > 2 {
                    panic!("unimplemented")
                }
                // TODO fence
                empty
            }
            Opcode::SYSTEM => {
                if issued.f3 == 0 {
                    // tohost
                    // FIXME: tmask?
                    ExecuteUnit::sfu(issued, wid, first_lid, num_lanes, scheduler);
                    empty
                } else {
                    // csr
                    ExecuteUnit::csr(issued, tmask, csrf, scheduler.active_warp_mask())
                }
            }
            Opcode::CUSTOM0 => {
                // FIXME: tmask?
                ExecuteUnit::sfu(issued, wid, first_lid, num_lanes, scheduler);
                empty
            }
            Opcode::CUSTOM1 => {
                // vx_tex, vx_cmov, vx_rop
                todo!("graphics ops unimplemented")
            }
            Opcode::CUSTOM2 => {
                panic!("nu.* instructions (Neutrino) are not supported")
            }
            Opcode::CUSTOM3 => Self::collect_lanes(|lane| ExecuteUnit::custom3(issued, lane), tmask),
            _ => {
                panic!("unimplemented opcode 0x{:x}", issued.opcode);
            }
        }
    }
}
