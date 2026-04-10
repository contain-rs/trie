use std::{
    array, collections::{BTreeMap, HashMap}, fmt::Debug, hash::Hash, marker::PhantomData, ops::{AddAssign, Index, IndexMut, SubAssign}
};

use crate::{
    Chunk,
    inner::Inner,
    node::{InternalNode, TrieNode},
    root_node::RootNode,
};

pub trait SubCount: TryInto<usize> + AddAssign<usize> + SubAssign<usize> + PartialEq + PartialEq<usize> + Copy + Default + Debug {}

pub trait MapTrait {
    const PREFETCH: bool = true;
    const SHIFT: u8;
    const INNER_LIMIT: usize = 1;
    type Count: SubCount;
    type Children<T>: Children<T> = [T; 16];
    type MaybeInner: Inner<Self::Key, Self::Value> + Debug;
    type Key: Chunk + Debug;
    type Value: Debug;
    type Root: RootNode<Self>
    where
        Self: Sized;
    type MaybeValue: MaybeValue<Self> where Self: Sized;
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

pub struct BasicTrieHint<K, V, Count>(PhantomData<(K, V, Count)>);
pub struct HatTrieHint<const LIMIT: usize, K, V, Count>(PhantomData<(K, V, Count)>);
pub struct BTrieHint<const LIMIT: usize, K, V, Count>(PhantomData<(K, V, Count)>);

#[derive(Debug)]
pub struct Unit<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

impl<K: Chunk + Debug, V: Debug, Count: SubCount> MapTrait for BasicTrieHint<K, V, Count> {
    const PREFETCH: bool = true;
    const SHIFT: u8 = 4;
    type Count = Count;
    type MaybeInner = Unit<K, V>;
    type Key = K;
    type Value = V;
    type Root = K::Root<Self, TrieNode<Self>, InternalNode<Self>>;
    type MaybeValue = K::MaybeValue<Self, Option<V>, NullValue<Self>>;
}

impl<const LIMIT: usize, K: Chunk + Debug, V: Debug, Count: SubCount> MapTrait for HatTrieHint<LIMIT, K, V, Count>
where
    K: Eq + Hash,
{
    const PREFETCH: bool = true;
    const SHIFT: u8 = 4;
    const INNER_LIMIT: usize = LIMIT;
    type Count = Count;
    type MaybeInner = HashMap<K, V>;
    type Key = K;
    type Value = V;
    type Root = K::Root<Self, TrieNode<Self>, InternalNode<Self>>;
    type MaybeValue = K::MaybeValue<Self, Option<V>, NullValue<Self>>;
}

impl<const LIMIT: usize, K: Chunk + Debug, V: Debug, Count: SubCount> MapTrait for BTrieHint<LIMIT, K, V, Count>
where
    K: Ord,
{
    const PREFETCH: bool = true;
    const SHIFT: u8 = 4;
    const INNER_LIMIT: usize = LIMIT;
    type Count = Count;
    type MaybeInner = BTreeMap<K, V>;
    type Key = K;
    type Value = V;
    type Root = K::Root<Self, TrieNode<Self>, InternalNode<Self>>;
    type MaybeValue = K::MaybeValue<Self, Option<V>, NullValue<Self>>;
}

#[derive(Default, Clone, Copy, Debug)]
pub struct NullCount;

impl AddAssign<usize> for NullCount {
    fn add_assign(&mut self, _rhs: usize) {
        // nothing to do
    }
}

impl SubAssign<usize> for NullCount {
    fn sub_assign(&mut self, _rhs: usize) {
        // nothing to do
    }
}

impl TryInto<usize> for NullCount {
    type Error = ();

    fn try_into(self) -> Result<usize, Self::Error> {
        Err(())
    }
}

impl PartialEq<usize> for NullCount {
    fn eq(&self, other: &usize) -> bool {
        false
    }
}

impl PartialEq for NullCount {
    fn eq(&self, other: &Self) -> bool {
        true
    }
}

impl SubCount for NullCount {}
impl SubCount for usize {}

// trait ValueChoice<M> where M: MapTrait {
//     type Out: Into<MaybeValue<M>>;
// }

// pub(crate) struct Value<const VARSIZED: bool, M>(PhantomData<M>);

pub struct NullValue<M>(PhantomData<M>);
// pub(crate) struct MaybeValue<M: MapTrait>(Option<M::Value>);
pub trait MaybeValue<M: MapTrait>: From<Option<M::Value>> {
    fn into_option(self) -> Option<M::Value>;
    fn as_ref(&self) -> Option<&M::Value>;
    fn as_mut(&mut self) -> Option<&mut M::Value>;
    fn replace(&mut self, elem: M::Value) -> Option<M::Value>;
    fn take(&mut self) -> Option<M::Value>;
}

// impl<M> ValueChoice<M> for Value<false, M> where M: MapTrait {
//     type Out = NullValue<M>;
// }

// impl<M> ValueChoice<M> for Value<true, M> where M: MapTrait {
//     type Out = MaybeValue<M>;
// }

impl<M> MaybeValue<M> for NullValue<M> where M: MapTrait {
    fn into_option(self) -> Option<M::Value> {
        None
    }

    fn as_mut(&mut self) -> Option<&mut M::Value> {
        None
    }

    fn as_ref(&self) -> Option<&<M as MapTrait>::Value> {
        None
    }

    fn replace(&mut self, elem: <M as MapTrait>::Value) -> Option<<M as MapTrait>::Value> {
        unreachable!()
    }

    fn take(&mut self) -> Option<M::Value> {
        None
    }
}

impl<M: MapTrait> From<Option<M::Value>> for NullValue<M> {
    fn from(value: Option<M::Value>) -> Self {
        NullValue(PhantomData)
    }
}

impl<M> MaybeValue<M> for Option<M::Value> where M: MapTrait {
    fn into_option(self) -> Option<<M as MapTrait>::Value> {
        self
    }

    fn as_mut(&mut self) -> Option<&mut <M as MapTrait>::Value> {
        self.as_mut()
    }

    fn as_ref(&self) -> Option<&<M as MapTrait>::Value> {
        self.as_ref()
    }

    fn replace(&mut self, elem: <M as MapTrait>::Value) -> Option<<M as MapTrait>::Value> {
        self.replace(elem)
    }

    fn take(&mut self) -> Option<M::Value> {
        self.take()
    }
}
