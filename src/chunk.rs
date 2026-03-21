use std::ops::Deref;

use crate::map_trait::{MapTrait, RootNode};

/// Allows us to extract bytes or parts of a byte (meaning, up to 8 bits
/// at a time).
///
/// Note: the `PartialEq` bound is included as a convenience,
/// since we always need it in practice.
pub trait Chunk: PartialEq {
    const VARSIZED: bool;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>>: RootNode<M>;
    const CONST_LEN: i64;

    fn chunk(&self, idx: u64, len: u8) -> u8;
    fn bits(&self) -> u64;
}

impl Chunk for Vec<u8> {
    const VARSIZED: bool = true;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = T;
    const CONST_LEN: i64 = -1;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        self.deref().chunk(idx, len)
    }

    fn bits(&self) -> u64 {
        self.len() as u64 * 8
    }
}

impl<const N: usize> Chunk for [u8; N] {
    const VARSIZED: bool = false;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = I;
    const CONST_LEN: i64 = N as i64 * 8;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        self[..].chunk(idx, len)
    }

    fn bits(&self) -> u64 {
        N as u64 * 8
    }
}

impl Chunk for [u8] {
    const VARSIZED: bool = true;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = T;
    const CONST_LEN: i64 = -1;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        debug_assert!(len <= 8);
        let first = self[(idx / 8) as usize] >> (idx % 8);
        let second = self.get((idx / 8 + 1) as usize).copied().unwrap_or(0) << (idx % 8);
        let mask = (1 << len) - 1;
        (first | second) & mask
    }

    fn bits(&self) -> u64 {
        self.len() as u64 * 8
    }
}

impl Chunk for String {
    const VARSIZED: bool = true;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = T;
    const CONST_LEN: i64 = -1;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        self.as_bytes().chunk(idx, len)
    }

    fn bits(&self) -> u64 {
        self.as_bytes().bits()
    }
}

impl Chunk for str {
    const VARSIZED: bool = true;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = T;
    const CONST_LEN: i64 = -1;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        self.as_bytes().chunk(idx, len)
    }

    fn bits(&self) -> u64 {
        self.as_bytes().bits()
    }
}

impl Chunk for usize {
    const VARSIZED: bool = false;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = I;
    const CONST_LEN: i64 = usize::BITS as i64;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        self.to_be_bytes().chunk(idx, len)
    }

    fn bits(&self) -> u64 {
        Self::BITS as u64
    }
}

impl Chunk for u8 {
    const VARSIZED: bool = false;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = I;
    const CONST_LEN: i64 = 8;
    fn chunk(&self, idx: u64, len: u8) -> u8 {
        [*self].chunk(idx, len)
    }

    fn bits(&self) -> u64 {
        Self::BITS as u64
    }
}

impl Chunk for u32 {
    const VARSIZED: bool = false;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = I;
    const CONST_LEN: i64 = 32;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        self.to_be_bytes().chunk(idx, len)
    }

    fn bits(&self) -> u64 {
        Self::BITS as u64
    }
}

impl Chunk for i32 {
    const VARSIZED: bool = false;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = I;
    const CONST_LEN: i64 = 32;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        self.to_be_bytes().chunk(idx, len)
    }

    fn bits(&self) -> u64 {
        Self::BITS as u64
    }
}

impl Chunk for u64 {
    const VARSIZED: bool = false;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = I;
    const CONST_LEN: i64 = 64;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        self.to_be_bytes().chunk(idx, len)
    }

    fn bits(&self) -> u64 {
        Self::BITS as u64
    }
}

impl Chunk for i64 {
    const VARSIZED: bool = false;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = I;
    const CONST_LEN: i64 = 64;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        self.to_be_bytes().chunk(idx, len)
    }

    fn bits(&self) -> u64 {
        Self::BITS as u64
    }
}

#[cfg(feature = "bit-vec")]
impl Chunk for bit_vec::BitVec {
    const VARSIZED: bool = true;
    type Root<M: MapTrait, T: RootNode<M>, I: RootNode<M>> = T;
    const CONST_LEN: i64 = -1;

    fn chunk(&self, idx: u64, len: u8) -> u8 {
        todo!()
    }

    fn bits(&self) -> u64 {
        self.len() as u64
    }
}
