use std::{collections::HashMap, marker::PhantomData};

use crate::{Chunk, node::{AnyNode, InternalNode, TrieNode}};

pub trait PerfHint<K, V> where K: Chunk {
    const PREFETCH: bool = true;
    const SHIFT: u8;
    const FANOUT: u8 = (1 << Self::SHIFT) as u8;
    type MaybeInner;
    type Root;
}

pub struct BasicTrieHint<K, V>(PhantomData<(K, V)>);
pub struct HatTrieHint<K, V>(PhantomData<(K, V)>);

struct NoInner;

impl<K: Chunk, V> PerfHint<K, V> for BasicTrieHint<K, V> where Root<{ K::VARSIZED }, K, V, Self>: RootChoice<K, V, Self> {
    const PREFETCH: bool = true;
    const SHIFT: u8 = 4;
    type MaybeInner = NoInner;
    type Root = <Root<{ K::VARSIZED }, K, V, Self> as RootChoice<K, V, Self>>::Out;
}


impl<K: Chunk, V> PerfHint<K, V> for HatTrieHint<K, V> where Root<{ K::VARSIZED }, K, V, Self>: RootChoice<K, V, Self> {
    const PREFETCH: bool = true;
    const SHIFT: u8 = 4;
    type MaybeInner = HashMap<K, V>;
    type Root = <Root<{ K::VARSIZED }, K, V, Self> as RootChoice<K, V, Self>>::Out;
}

pub(crate) struct Root<const VARSIZED: bool, K, V, P>(PhantomData<(K, V, P)>);

pub(crate) trait RootChoice<K, V, P> where K: Chunk, P: PerfHint<K, V> {
    type Out: RootNode<K, V, P>;
}

pub(crate) trait RootNode<K, V, P> where K: Chunk, P: PerfHint<K, V> {
    fn nothing() -> Self;
    fn into_any(self) -> AnyNode<K, V, P>;
}

impl<K, V, P> RootNode<K, V, P> for TrieNode<K, V, P> where K: Chunk, P: PerfHint<K, V> {
    fn nothing() -> Self {
        TrieNode::Nothing
    }

    fn into_any(self) -> AnyNode<K, V, P> {
        match self {
            Self::Internal(bx) => AnyNode::Internal(bx),
            Self::External(k, v) => AnyNode::External(k, v),
            Self::Nothing => AnyNode::Nothing,
        }
    }
}

impl<K, V, P> RootNode<K, V, P> for InternalNode<K, V, P> where K: Chunk, P: PerfHint<K, V>, [(); P::FANOUT as usize]: Sized {
    fn nothing() -> Self {
        InternalNode {
            count: 0,
            children: [TrieNode::Nothing; P::FANOUT as usize],
        }
    }

    fn into_any(self) -> AnyNode<K, V, P> {
        AnyNode::Branch { count: self.count, children: self.children }
    }
}

impl<K, V, P> RootChoice<K, V, P> for Root<true, K, V, P> where K: Chunk, P: PerfHint<K, V> {
    type Out = TrieNode<K, V, P>;
}

impl<K, V, P> RootChoice<K, V, P> for Root<false, K, V, P> where K: Chunk, P: PerfHint<K, V> {
    type Out = InternalNode<K, V, P>;
}
