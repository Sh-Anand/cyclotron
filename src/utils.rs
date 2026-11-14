use num_traits::{PrimInt, Unsigned};

pub trait BitSlice {
    fn sel(&self, msb: usize, lsb: usize) -> Self;
    fn bit(&self, idx: usize) -> bool;
    fn mut_bit(&mut self, idx: usize, to: bool);
}

pub trait BitSlice64 {
    fn sel64(&self, msb: usize, lsb: usize) -> Self;
}

impl BitSlice64 for u64 {
    fn sel64(&self, msb: usize, lsb: usize) -> u64 {
        assert!(msb >= lsb, "invalid bit slice");
        let mask: u64 = if msb - lsb >= 63 {
            0xffffffffffffffffu64
        } else {
            (1u64 << (msb - lsb + 1)) - 1u64
        };
        (*self >> lsb) & mask
    }
}

impl<T: PrimInt + Unsigned + TryFrom<u64>> BitSlice for T {
    fn sel(&self, msb: usize, lsb: usize) -> T {
        let self64: u64 = self.to_u64().unwrap();
        self64.sel64(msb, lsb).try_into().map_err(|_| "").unwrap()
    }

    fn bit(&self, idx: usize) -> bool {
        let self64: u64 = self.to_u64().unwrap();
        self64.sel64(idx, idx) > 0
    }

    fn mut_bit(&mut self, idx: usize, to: bool) {
        let self64: u64 = self.to_u64().unwrap();
        *self = (self64 & (!(1u64 << idx)) | ((to as u64) << idx))
            .try_into()
            .map_err(|_| "")
            .unwrap();
    }
}

pub trait BitMask {
    fn to_u32(&self) -> u32;
}

impl BitMask for [bool] {
    #[inline]
    fn to_u32(&self) -> u32 {
        assert!(self.len() <= 32, "Bitmask too large for u32");
        let mut mask_int = 0;
        for (i, bit) in self.iter().enumerate().take(32) {
            mask_int |= (*bit as u32) << i;
        }

        mask_int
    }
}

impl BitMask for Vec<bool> {
    fn to_u32(&self) -> u32 {
        self.as_slice().to_u32()
    }
}
