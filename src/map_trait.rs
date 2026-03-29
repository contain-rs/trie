use std::{
    array,
    collections::{BTreeMap, HashMap},
    hash::Hash,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use crate::{
    Chunk,
    inner::Inner,
    node::{InternalNode, TrieNode},
    root_node::RootNode,
};

pub trait MapTrait {
    const PREFETCH: bool = true;
    const SHIFT: u8;
    type Children<T>: Children<T> = [T; 16];
    type MaybeInner: Inner<Self::Key, Self::Value>;
    type Key: Chunk;
    type Value;
    type Root: RootNode<Self>
    where
        Self: Sized;
}

pub trait Children<T>: Index<usize, Output = T> + IndexMut<usize> {
    fn from_fn<F: Fn(usize) -> T>(f: F) -> Self;

    fn as_slice(&self) -> &[T];
    fn as_slice_mut(&mut self) -> &mut [T];
}

impl<T, const N: usize> Children<T> for [T; N] {
    fn from_fn<F: Fn(usize) -> T>(f: F) -> Self {
        array::from_fn(f)
    }

    fn as_slice(&self) -> &[T] {
        &self[..]
    }

    fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

pub struct BasicTrieHint<K, V>(PhantomData<(K, V)>);
pub struct HatTrieHint<K, V>(PhantomData<(K, V)>);
pub struct BTrieHint<K, V>(PhantomData<(K, V)>);

pub struct Unit<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

impl<K: Chunk, V> MapTrait for BasicTrieHint<K, V> {
    const PREFETCH: bool = true;
    const SHIFT: u8 = 4;
    type MaybeInner = Unit<K, V>;
    type Key = K;
    type Value = V;
    type Root = K::Root<Self, TrieNode<Self>, InternalNode<Self>>;
}

impl<K: Chunk, V> MapTrait for HatTrieHint<K, V>
where
    K: Eq + Hash,
{
    const PREFETCH: bool = true;
    const SHIFT: u8 = 4;
    type MaybeInner = HashMap<K, V>;
    type Key = K;
    type Value = V;
    type Root = K::Root<Self, TrieNode<Self>, InternalNode<Self>>;
}

impl<K: Chunk, V> MapTrait for BTrieHint<K, V>
where
    K: Ord,
{
    const PREFETCH: bool = true;
    const SHIFT: u8 = 4;
    type MaybeInner = BTreeMap<K, V>;
    type Key = K;
    type Value = V;
    type Root = K::Root<Self, TrieNode<Self>, InternalNode<Self>>;
}
