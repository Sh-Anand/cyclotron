use crate::sim::config::Config;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(default)]
pub struct MuonConfig {
    pub num_lanes: usize,
    pub num_warps: usize,
    pub num_cores: usize, // per-cluster
    pub num_regs: usize,
    pub start_pc: u32,
    pub smem_size: usize,
    pub ibuf_depth: usize, // decoded instructions per warp available for issue
    pub units: UnitsConfig,
    #[serde(skip)]
    pub lane_config: LaneConfig,
}

impl Config for MuonConfig {}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            num_lanes: 16,
            num_warps: 8,
            num_cores: 1,
            num_regs: 256,
            start_pc: 0x10000000u32,
            smem_size: 0x10_0000, // includes MMIO space
            ibuf_depth: 4,
            units: UnitsConfig::default(),
            lane_config: LaneConfig::default(),
        }
    }
}

/// Functional units; all are fully pipelined.  An instruction occupies its unit for
/// ceil(num_lanes / lanes) cycles, and its rd is ready `latency` cycles after the last of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Alu,
    Mul,
    Div,
    Fp,
    FpDivSqrt,
    FpExp,
    Sfu,
    Lsu,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct UnitConfig {
    pub lanes: usize,
    pub latency: u64, // issue to dependent issue
}

/// Defaults follow Muon's RTL for a 32-lane warp: unit widths from the Configs builder,
/// latencies from each pipe plus 2 cycles of collector read and RS wakeup.
#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(default)]
pub struct UnitsConfig {
    pub alu: UnitConfig,
    pub mul: UnitConfig,
    pub div: UnitConfig,
    pub fp: UnitConfig,
    pub fp_divsqrt: UnitConfig,
    pub fp_exp: UnitConfig,
    pub sfu: UnitConfig,
    pub lsu: UnitConfig,
}

impl Default for UnitsConfig {
    fn default() -> Self {
        let u = |lanes, latency| UnitConfig { lanes, latency };
        Self {
            alu: u(32, 3),         // rocket ALU, recomposer register
            mul: u(32, 36),        // rocket MulDiv, mulUnroll = 1: 32 iterations + 2
            div: u(32, 37),        // rocket MulDiv, divUnroll = 1: 32 iterations + sign fixup + 2
            fp: u(16, 5),          // CVFPU addmul/noncomp/conv: 2 pipe regs + recomposer
            fp_divsqrt: u(4, 16),  // CVFPU PULP divsqrt, 3 iteration units: ~9 iterations + pre/post + 2 pipe regs
            fp_exp: u(8, 9),       // FPEX: 6 stages + recomposer
            sfu: u(32, 3),
            lsu: u(32, 4),         // placeholder until the memory hierarchy is modelled
        }
    }
}

impl UnitsConfig {
    pub fn get(&self, unit: Unit) -> UnitConfig {
        match unit {
            Unit::Alu => self.alu,
            Unit::Mul => self.mul,
            Unit::Div => self.div,
            Unit::Fp => self.fp,
            Unit::FpDivSqrt => self.fp_divsqrt,
            Unit::FpExp => self.fp_exp,
            Unit::Sfu => self.sfu,
            Unit::Lsu => self.lsu,
        }
    }
}

impl MuonConfig {
    /// cycles a warp-instruction occupies `unit`
    pub fn passes(&self, unit: Unit) -> u64 {
        self.num_lanes.div_ceil(self.units.get(unit).lanes) as u64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LaneConfig {
    pub lane_id: usize,
    pub warp_id: usize,
    pub core_id: usize,
    pub cluster_id: usize,
}

impl Default for LaneConfig {
    fn default() -> Self {
        Self {
            lane_id: 0,
            warp_id: 0,
            core_id: 0,
            cluster_id: 0,
        }
    }
}
