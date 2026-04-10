use std::{marker::PhantomData, mem::MaybeUninit};

use crate::{
    inner::Inner, map_trait::{Children, MapTrait, MaybeValue}, node::{InternalNode, TrieNode}
};

impl<M> RootNode<M> for TrieNode<M>
where
    M: MapTrait,
{
    fn nothing() -> Self {
        TrieNode::Nothing
    }

    fn into_any(self) -> AnyNode<M> {
        match self {
            Self::Internal(bx) => AnyNode::Internal(bx),
            Self::External(external) => AnyNode::External(external),
            Self::Nothing => AnyNode::Nothing,
        }
    }

    fn from_any_ref(any: AnyNodeRef<M>) -> Self
    where
        M::Key: Clone,
        M::Value: Clone,
    {
        match any {
            AnyNodeRef::Branch { count, skip, value, children } => Self::Internal(Box::new(InternalNode {
                count,
                value: value.cloned().into(),
                skip: skip.map_or(0, |(n, _)| n),
                maybe_key: if let Some((_, key)) = skip { MaybeUninit::new(key.clone()) } else { MaybeUninit::uninit() },
                children: M::Children::from_fn(|i| children[i].clone()),
            })),
            AnyNodeRef::External(external) => Self::External(external.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            AnyNodeRef::Nothing => Self::Nothing,
        }
    }

    fn into_any_ref(&self) -> AnyNodeRef<M> {
        unsafe {
            match self {
                Self::Internal(bx) => AnyNodeRef::Branch {
                    count: bx.count,
                    value: bx.value.as_ref(),
                    skip: if bx.skip == 0 { None } else { Some((bx.skip, bx.maybe_key.assume_init_ref())) },
                    children: bx.children.as_slice(),
                },
                Self::External(external) => AnyNodeRef::External(external),
                Self::Nothing => AnyNodeRef::Nothing,
            }
        }
    }

    fn as_any_mut(&mut self) -> AnyNodeMut<M> {
        match self {
            Self::Internal(bx) => AnyNodeMut::Branch {
                count: bx.count,
                children: bx.children.as_slice_mut(),
            },
            Self::External(external) => AnyNodeMut::External(external),
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

impl<M> RootNode<M> for InternalNode<M>
where
    M: MapTrait,
{
    fn nothing() -> Self {
        InternalNode {
            count: Default::default(),
            value: None.into(),
            skip: 0,
            maybe_key: MaybeUninit::uninit(),
            children: Children::from_fn(|_| TrieNode::Nothing),
        }
    }

    fn into_any(self) -> AnyNode<M> {
        AnyNode::Branch {
            count: self.count,
            children: self.children,
        }
    }

    fn into_any_ref(&self) -> AnyNodeRef<M> {
        unsafe {
            AnyNodeRef::Branch {
                count: self.count,
                value: self.value.as_ref(),
                skip: if self.skip == 0 { None } else { Some((self.skip, self.maybe_key.assume_init_ref())) },
                children: self.children.as_slice(),
            }
        }
    }

    fn from_any_ref(any: AnyNodeRef<M>) -> Self
    where
        M::Key: Clone,
        M::Value: Clone,
    {
        match any {
            AnyNodeRef::Branch { count, skip, value, children } => Self {
                count,
                value: value.cloned().into(),
                skip: skip.map_or(0, |(n, _)| n),
                maybe_key: if let Some((_n, key)) = skip { MaybeUninit::new(key.clone()) } else { MaybeUninit::uninit() },
                children: M::Children::from_fn(|i| children[i].clone()),
            },
            AnyNodeRef::External(_external) => unreachable!(),
            AnyNodeRef::Nothing => unreachable!(),
        }
    }

    fn as_any_mut(&mut self) -> AnyNodeMut<M> {
        AnyNodeMut::Branch {
            count: self.count,
            children: self.children.as_slice_mut(),
        }
    }

    fn as_either(&mut self) -> EitherNode<M> {
        EitherNode::Internal(self)
    }

    fn as_either_ref(&self) -> EitherNodeRef<M> {
        EitherNodeRef::Internal(self)
    }
}

pub(crate) struct Root<const VARSIZED: bool, M>(PhantomData<M>);

pub(crate) trait RootChoice<M>
where
    M: MapTrait,
{
    type Out: RootNode<M>;
}

pub trait RootNode<M>
where
    M: MapTrait,
{
    fn nothing() -> Self;
    fn into_any(self) -> AnyNode<M>;
    fn into_any_ref(&self) -> AnyNodeRef<M>;
    fn as_any_mut(&mut self) -> AnyNodeMut<M>;
    fn as_either(&mut self) -> EitherNode<M>;
    fn as_either_ref(&self) -> EitherNodeRef<M>;
    fn from_any_ref(any: AnyNodeRef<M>) -> Self
    where
        M::Key: Clone,
        M::Value: Clone;
}

impl<M> RootChoice<M> for Root<true, M>
where
    M: MapTrait,
{
    type Out = TrieNode<M>;
}

impl<M> RootChoice<M> for Root<false, M>
where
    M: MapTrait,
{
    type Out = InternalNode<M>;
}

pub(crate) enum AnyNode<M: MapTrait> {
    Internal(Box<InternalNode<M>>),
    External(M::MaybeInner),
    Branch {
        count: M::Count,
        children: M::Children<TrieNode<M>>,
    },
    Nothing,
}

pub(crate) enum AnyNodeRef<'a, M: MapTrait> {
    External(&'a M::MaybeInner),
    Branch {
        count: M::Count,
        value: Option<&'a M::Value>,
        skip: Option<(usize, &'a M::Key)>,
        children: &'a [TrieNode<M>],
    },
    Nothing,
}

pub(crate) enum AnyNodeMut<'a, M: MapTrait> {
    External(&'a mut M::MaybeInner),
    Branch {
        count: M::Count,
        children: &'a mut [TrieNode<M>],
    },
    Nothing,
}

pub(crate) enum EitherNode<'a, M: MapTrait> {
    Trie(&'a mut TrieNode<M>),
    Internal(&'a mut InternalNode<M>),
}

pub(crate) enum EitherNodeRef<'a, M: MapTrait> {
    Trie(&'a TrieNode<M>),
    Internal(&'a InternalNode<M>),
}
