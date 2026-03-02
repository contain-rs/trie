
#[cfg(target_pointer_width = "32")]
pub const USIZE_BITS: usize = 32;

#[cfg(target_pointer_width = "64")]
pub const USIZE_BITS: usize = 64;

// FIXME(conventions): implement bounded iterators
// FIXME(conventions): implement into_iter
// FIXME(conventions): replace each_reverse by making iter DoubleEnded

// FIXME: #5244: need to manually update the InternalNode constructor
pub const SHIFT: usize = 4;
pub const SIZE: usize = 1 << SHIFT;
pub const MASK: usize = SIZE - 1;
// The number of chunks that the key is divided into. Also the maximum depth of the map.
pub const MAX_DEPTH: usize = USIZE_BITS / SHIFT;

pub trait Chunk: PartialEq {
    fn chunk(&self, idx: usize) -> usize;
}

impl Chunk for Vec<u8> {
    fn chunk(&self, idx: usize) -> usize {
        (self[idx / 2] >> (idx % 2 * 4)) as usize
    }
}

impl<const N: usize> Chunk for [u8; N] {
    fn chunk(&self, idx: usize) -> usize {
        (self[idx / 2] >> (idx % 2 * 4)) as usize
    }
}

impl Chunk for usize {
    fn chunk(&self, idx: usize) -> usize {
        let sh = USIZE_BITS - (SHIFT * (idx + 1));
        (self >> sh) & MASK
    }
}
