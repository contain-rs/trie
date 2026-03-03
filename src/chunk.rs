pub(crate) const SHIFT: u8 = 4;
pub(crate) const SIZE: usize = 1 << SHIFT;
pub(crate) const MASK: usize = SIZE - 1;
// The number of chunks that the key is divided into. Also the maximum depth of the map.
pub(crate) const MAX_DEPTH: usize = usize::BITS as usize / SHIFT as usize;

/// Allows us to extract nybbles (4 bits at a time).
///
/// Should be implemented for lists of bytes, byte strings,
/// as well as basic integer types.
///
/// Note: the `PartialEq` bound is included as a convenience,
/// since we always need it in practice.
///
/// Note: we accept and return `usize` rather than `u8` as a convenience because
/// we wish to use these for indices without casting.
pub trait Chunk: PartialEq {
    /// Note: we accept and return `usize` rather than `u8` as a convenience because
    /// we wish to use these for indices without casting.
    fn chunk(&self, idx: usize) -> usize;
}

impl Chunk for Vec<u8> {
    fn chunk(&self, idx: usize) -> usize {
        (self[idx / 2] >> ((idx % 2) * 4)) as usize
    }
}

impl<const N: usize> Chunk for [u8; N] {
    fn chunk(&self, idx: usize) -> usize {
        (self[idx / 2] >> ((idx % 2) * 4)) as usize
    }
}

impl Chunk for [u8] {
    fn chunk(&self, idx: usize) -> usize {
        (self[idx / 2] >> ((idx % 2) * 4)) as usize
    }
}

impl Chunk for usize {
    fn chunk(&self, idx: usize) -> usize {
        let sh = usize::BITS as u8 - (SHIFT * (idx as u8 + 1));
        (self >> sh) & MASK
    }
}

impl Chunk for u32 {
    fn chunk(&self, idx: usize) -> usize {
        let sh = u32::BITS as u8 - (SHIFT * (idx as u8 + 1));
        ((self >> sh) & MASK as u32) as usize
    }
}

impl Chunk for i32 {
    fn chunk(&self, idx: usize) -> usize {
        let sh = i32::BITS as u8 - (SHIFT * (idx as u8 + 1));
        ((*self as u32 >> sh) & MASK as u32) as usize
    }
}

impl Chunk for u64 {
    fn chunk(&self, idx: usize) -> usize {
        let sh = u64::BITS as u8 - (SHIFT * (idx as u8 + 1));
        ((self >> sh) & MASK as u64) as usize
    }
}

impl Chunk for i64 {
    fn chunk(&self, idx: usize) -> usize {
        let sh = i64::BITS as u8 - (SHIFT * (idx as u8 + 1));
        (((*self) as u64 >> sh) & MASK as u64) as usize
    }
}
