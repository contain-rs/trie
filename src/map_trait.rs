use std::{array, collections::HashMap, marker::PhantomData, ops::{Index, IndexMut}};

use crate::{Chunk, node::{AnyNode, AnyNodeMut, AnyNodeRef, EitherNode, EitherNodeRef, InternalNode, TrieNode}};

pub trait MapTrait {
    const PREFETCH: bool = true;
    const SHIFT: u8;
    type Children<T>: Children<T> = [T; 16];
    type MaybeInner;
    type Key: Chunk;
    type Value;
    type Root: RootNode<Self> where Self: Sized;
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

pub struct NoInner;

impl<K: Chunk, V> MapTrait for BasicTrieHint<K, V> {
    const PREFETCH: bool = true;
    const SHIFT: u8 = 4;
    type MaybeInner = NoInner;
    type Key = K;
    type Value = V;
    type Root = K::Root<Self, TrieNode<Self>, InternalNode<Self>>;
}


impl<K: Chunk, V> MapTrait for HatTrieHint<K, V> where Root<{ K::VARSIZED }, Self>: RootChoice<Self> {
    const PREFETCH: bool = true;
    const SHIFT: u8 = 4;
    type MaybeInner = HashMap<K, V>;
    // type Root = <Root<{ K::VARSIZED }, Self> as RootChoice<Self>>::Out;
    type Key = K;
    type Value = V;
    type Root = K::Root<Self, TrieNode<Self>, InternalNode<Self>>;
}

pub(crate) struct Root<const VARSIZED: bool, M>(PhantomData<M>);

pub(crate) trait RootChoice<M> where M: MapTrait {
    type Out: RootNode<M>;
}

pub trait RootNode<M> where M: MapTrait {
    fn nothing() -> Self;
    fn into_any(self) -> AnyNode<M>;
    fn into_any_ref(&self) -> AnyNodeRef<M>;
    fn as_any_mut(&mut self) -> AnyNodeMut<M>;
    fn as_either(&mut self) -> EitherNode<M>;
    fn as_either_ref(&self) -> EitherNodeRef<M>;
    fn from_any_ref(any: AnyNodeRef<M>) -> Self where M::Key: Clone, M::Value: Clone, M::Children<TrieNode<M>>: Clone;
}

impl<M> RootNode<M> for TrieNode<M> where M: MapTrait {
    fn nothing() -> Self {
        TrieNode::Nothing
    }

    fn into_any(self) -> AnyNode<M> {
        match self {
            Self::Internal(bx) => AnyNode::Internal(bx),
            Self::External(k, v) => AnyNode::External(k, v),
            Self::Nothing => AnyNode::Nothing,
        }
    }

    fn from_any_ref(any: AnyNodeRef<M>) -> Self where M::Key: Clone, M::Value: Clone, M::Children<TrieNode<M>>: Clone {
        match any {
            AnyNodeRef::Branch { count, children } => Self::Internal(Box::new(InternalNode { count, children: M::Children::from_fn(|i| children[i].clone()) })),
            AnyNodeRef::External(k, v) => Self::External(k.clone(), v.clone()),
            AnyNodeRef::Nothing => Self::Nothing,
        }
    }

    fn into_any_ref(&self) -> AnyNodeRef<M> {
        match self {
            Self::Internal(bx) => AnyNodeRef::Branch { count: bx.count, children: bx.children.as_slice() },
            Self::External(k, v) => AnyNodeRef::External(k, v),
            Self::Nothing => AnyNodeRef::Nothing,
        }
    }

    fn as_any_mut(&mut self) -> AnyNodeMut<M> {
        match self {
            Self::Internal(bx) => AnyNodeMut::Branch { count: bx.count, children: bx.children.as_slice_mut() },
            Self::External(k, v) => AnyNodeMut::External(k, v),
            Self::Nothing => AnyNodeMut::Nothing,
        }
    }

    fn as_either(&mut self) -> EitherNode<M> {
        EitherNode::Trie(self)
    }

    fn as_either_ref(&self) -> EitherNodeRef<M> {
        EitherNodeRef::Trie(self)
    }
}

impl<M> RootNode<M> for InternalNode<M> where M: MapTrait {
    fn nothing() -> Self {
        InternalNode {
            count: 0,
            children: Children::from_fn(|_| TrieNode::Nothing),
        }
    }

    fn into_any(self) -> AnyNode<M> {
        AnyNode::Branch { count: self.count, children: self.children }
    }

    fn into_any_ref(&self) -> AnyNodeRef<M> {
        AnyNodeRef::Branch { count: self.count, children: self.children.as_slice() }
    }

    fn from_any_ref(any: AnyNodeRef<M>) -> Self where M::Key: Clone, M::Value: Clone, M::Children<TrieNode<M>>: Clone {
        match any {
            AnyNodeRef::Branch { count, children } => Self { count, children: M::Children::from_fn(|i| children[i].clone()) },
            AnyNodeRef::External(k, v) => unreachable!(),
            AnyNodeRef::Nothing => unreachable!(),
        }
    }

    fn as_any_mut(&mut self) -> AnyNodeMut<M> {
        AnyNodeMut::Branch { count: self.count, children: self.children.as_slice_mut() }
    }

    fn as_either(&mut self) -> EitherNode<M> {
        EitherNode::Internal(self)
    }

    fn as_either_ref(&self) -> EitherNodeRef<M> {
        EitherNodeRef::Internal(self)
    }
}

impl<M> RootChoice<M> for Root<true, M> where M: MapTrait {
    type Out = TrieNode<M>;
}

impl<M> RootChoice<M> for Root<false, M> where M: MapTrait {
    type Out = InternalNode<M>;
}
