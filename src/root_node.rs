use std::marker::PhantomData;

use crate::{
    map_trait::{Children, MapTrait},
    node::{InternalNode, TrieNode},
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
        M::Children<TrieNode<M>>: Clone,
        M::MaybeInner: Clone,
    {
        match any {
            AnyNodeRef::Branch { count, children } => Self::Internal(Box::new(InternalNode {
                count,
                children: M::Children::from_fn(|i| children[i].clone()),
            })),
            AnyNodeRef::External(external) => Self::External(external.clone()),
            AnyNodeRef::Nothing => Self::Nothing,
        }
    }

    fn into_any_ref(&self) -> AnyNodeRef<M> {
        match self {
            Self::Internal(bx) => AnyNodeRef::Branch {
                count: bx.count,
                children: bx.children.as_slice(),
            },
            Self::External(external) => AnyNodeRef::External(external),
            Self::Nothing => AnyNodeRef::Nothing,
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
            count: 0,
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
        AnyNodeRef::Branch {
            count: self.count,
            children: self.children.as_slice(),
        }
    }

    fn from_any_ref(any: AnyNodeRef<M>) -> Self
    where
        M::Key: Clone,
        M::Value: Clone,
        M::Children<TrieNode<M>>: Clone,
        M::MaybeInner: Clone,
    {
        match any {
            AnyNodeRef::Branch { count, children } => Self {
                count,
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
        M::Value: Clone,
        M::Children<TrieNode<M>>: Clone,
        M::MaybeInner: Clone;
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
        count: usize,
        children: M::Children<TrieNode<M>>,
    },
    Nothing,
}

pub(crate) enum AnyNodeRef<'a, M: MapTrait> {
    External(&'a M::MaybeInner),
    Branch {
        count: usize,
        children: &'a [TrieNode<M>],
    },
    Nothing,
}

// pub(crate) enum TrieNodeRef<'a, M: MapTrait> {
//     External(&'a M::Key, &'a M::Value),
//     Internal()
// }

pub(crate) enum AnyNodeMut<'a, M: MapTrait> {
    External(&'a mut M::MaybeInner),
    Branch {
        count: usize,
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
