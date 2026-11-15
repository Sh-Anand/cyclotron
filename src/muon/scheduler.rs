use crate::muon::config::MuonConfig;
use crate::muon::decode::IssuedInst;
use crate::muon::execute::SFUType;
use crate::utils::{BitMask, BitSlice};
use std::collections::VecDeque;

#[derive(Debug, Default)]
pub struct IpdomEntry {
    pub pc: u32,
    pub tmask: u32,
}

#[derive(Debug, Default)]
pub struct SchedulerState {
    pub active_warps: u32,
    pub allocated: u32, // warp slots held by a CTA; freed only when the whole CTA is
    pub thread_masks: Vec<u32>,
    pub tohost: Option<u32>,
    pc: Vec<u32>,
    ipdom_stack: Vec<VecDeque<IpdomEntry>>,
}

/// Per-warp info of which instruction to fetch next
#[derive(Debug, Default, Clone, Copy)]
pub struct Schedule {
    pub pc: u32,
    pub mask: u32,
}

// instantiated per core
#[derive(Debug, Default)]
pub struct Scheduler {
    conf: MuonConfig,
    pub state: SchedulerState,
}

impl Scheduler {
    pub fn new(config: MuonConfig) -> Self {
        let num_warps = config.num_warps;
        Scheduler {
            conf: config,
            state: SchedulerState {
                active_warps: 0,
                allocated: 0,
                thread_masks: vec![0u32; num_warps],
                tohost: None,
                pc: vec![0; num_warps], // set when a warp spawns
                ipdom_stack: (0..num_warps).map(|_| VecDeque::new()).collect(),
            },
        }
    }

    fn all_one_bitmask(&self) -> u32 {
        // hack: left shifting by 32 undefined for u32. ideally use u64 everywhere
        if self.conf.num_lanes >= 32 {
            u32::MAX
        } else {
            (1u32 << self.conf.num_lanes) - 1
        }
    }

    /// start warp `wid` at `pc` with `lanes` active threads; other warps are untouched
    pub fn spawn_warp(&mut self, wid: usize, pc: u32, lanes: usize) {
        assert!(!self.state.allocated.bit(wid), "warp {wid} still allocated");
        self.state.pc[wid] = pc;
        self.state.thread_masks[wid] = self.all_one_bitmask() >> (self.conf.num_lanes - lanes);
        self.state.active_warps.mut_bit(wid, true);
        self.state.allocated.mut_bit(wid, true);
    }

    pub fn take_branch(&mut self, wid: usize, target_pc: u32) {
        self.state.pc[wid] = target_pc;
        if target_pc == 0 {
            // returned from main
            self.state.thread_masks[wid] = 0;
            self.state.active_warps.mut_bit(wid, false);
        }
    }

    pub fn sfu(&mut self, wid: usize, first_lid: usize, sfu: SFUType, decoded_inst: &IssuedInst) {
        let (rs1, rs2) = (&decoded_inst.rs1_data, &decoded_inst.rs2_data);
        // for warp-wide operations, we take lane 0 to be the truth
        self.state.pc[wid] = decoded_inst.pc + 8; // flush
        match sfu {
            SFUType::TMC => {
                let tmask = rs1[first_lid] & self.all_one_bitmask();
                self.state.thread_masks[wid] = tmask;
                if tmask == 0 {
                    self.state.active_warps.mut_bit(wid, false);
                }
            }
            SFUType::WSPAWN => {
                let start_pc = rs2[first_lid];
                let count = rs1[first_lid];
                for i in 1..count as usize {
                    if self.state.active_warps.bit(i) {
                        continue;
                    }

                    self.state.active_warps.mut_bit(i, true);
                    self.state.pc[i] = start_pc;
                    self.state.thread_masks[i] = self.all_one_bitmask();
                }
            }
            SFUType::SPLIT => {
                // deviation from vortex behavior: instead of checking for 1,
                // check for nonzero. works with renaming front end
                let invert = decoded_inst.rs2_addr != 0;
                let then_mask = rs1.iter().map(|r| r.bit(0)).collect::<Vec<_>>().to_u32()
                    & self.state.thread_masks[wid];
                let else_mask = rs1.iter().map(|r| !r.bit(0)).collect::<Vec<_>>().to_u32()
                    & self.state.thread_masks[wid];


                // NOTE: we should not need to push the else-path of
                // non-divergent branches to the ipdom stack; this way, the
                // later vx_join will just see the restored mask & invalid PC at
                // stack top, and skip serialization.  This also makes writing
                // back divergence to RD unnecessary.
                //
                // Therefore, we are NOT writing back the divergence result to
                // RD, which differs from vortex behavior for now.
                self.state.ipdom_stack[wid].push_back(IpdomEntry {
                    pc: 0u32,
                    tmask: then_mask | else_mask,
                });

                let diverges = (then_mask != 0) && (else_mask != 0);
                let now_mask = if invert { else_mask } else { then_mask };
                let later_mask = if invert { then_mask } else { else_mask };

                if diverges {
                    self.state.ipdom_stack[wid].push_back(IpdomEntry {
                        pc: decoded_inst.pc + 8,
                        tmask: later_mask,
                    });
                    self.state.thread_masks[wid] = now_mask;
                }
            }
            SFUType::JOIN => {
                let entry = self.state.ipdom_stack[wid]
                    .pop_back()
                    .expect("join without split");
                if entry.pc > 0 {
                    self.state.pc[wid] = entry.pc;
                }
                self.state.thread_masks[wid] = entry.tmask;
            }
            SFUType::BAR => {}
            SFUType::PRED => {
                let invert = decoded_inst.rd_addr != 0;
                let pred_mask = rs1
                    .iter()
                    .map(|r| r.bit(0) ^ invert)
                    .collect::<Vec<_>>()
                    .to_u32();
                self.state.thread_masks[wid] &= pred_mask;

                // if all threads are not active, set thread mask to rs2 of warp leader
                if self.state.thread_masks[wid] == 0 {
                    let tmask = rs2[first_lid];
                    self.state.thread_masks[wid] = tmask;
                    if tmask == 0 {
                        self.state.active_warps.mut_bit(wid, false);
                    }
                }
            }
            SFUType::KILL => {
                self.state.thread_masks[wid] = 0;
                self.state.active_warps.mut_bit(wid, false);
            }
            SFUType::ECALL => {
                let a0 = rs1[first_lid];
                self.state.tohost = Some(a0);
                self.state.thread_masks[wid] = 0;
                self.state.active_warps.mut_bit(wid, false);
            }
        }
    }

    pub fn schedule(&mut self, wid: usize) -> Option<Schedule> {
        self.state.active_warps.bit(wid).then(|| {
            let pc = self.state.pc[wid];
            let sched = Schedule { pc, mask: self.state.thread_masks[wid] };
            self.state.pc[wid] = pc.wrapping_add(8);
            sched
        })
    }

    /// next pc warp `wid` fetches
    pub fn pc(&self, wid: usize) -> u32 {
        self.state.pc[wid]
    }

    pub fn tohost(&self) -> Option<u32> {
        self.state.tohost
    }

    pub fn active_warp_mask(&self) -> u32 {
        self.state.active_warps
    }

    pub fn reset(&mut self) {
        self.state.thread_masks = [self.all_one_bitmask()].repeat(self.conf.num_warps);
        self.state.active_warps = 0;
    }
}
