use crate::map_trait::{Children, MapTrait};

// An internal node holds SIZE child nodes, which may themselves contain more internal nodes.
//
// Throughout this implementation, "idx" is used to refer to a section of key that is used
// to access a node. The layer of the tree directly below the root corresponds to idx 0.
pub struct InternalNode<M: MapTrait> {
    // The number of direct children which are external (i.e. that store a value).
    pub(crate) count: usize,
    pub(crate) children: M::Children<TrieNode<M>>,
}

impl<M: MapTrait> Clone for InternalNode<M> where M::Key: Clone, M::Value: Clone {
    fn clone(&self) -> Self {
        Self {
            count: self.count,
            children: M::Children::from_fn(|i| self.children.as_slice()[i].clone()),
        }
    }
}

// Each child of an InternalNode may be internal, in which case nesting continues,
// external (containing a value), or empty.
// The root of the Map is also one of these variants.
pub enum TrieNode<M: MapTrait> {
    Internal(Box<InternalNode<M>>),
    External(M::Key, M::Value),
    Nothing,
}

impl<M: MapTrait> Clone for TrieNode<M> where M::Key: Clone, M::Value: Clone {
    fn clone(&self) -> Self {
        match self {
            TrieNode::External(k, v) => TrieNode::External(k.clone(), v.clone()),
            TrieNode::Internal(b) => TrieNode::Internal((*b).clone()),
            TrieNode::Nothing => TrieNode::Nothing,
        }
    }
}

pub(crate) enum AnyNode<M: MapTrait> {
    Internal(Box<InternalNode<M>>),
    External(M::Key, M::Value),
    Branch {
        count: usize,
        children: M::Children<TrieNode<M>>,
    },
    Nothing,
}

pub(crate) enum AnyNodeRef<'a, M: MapTrait> {
    External(&'a M::Key, &'a M::Value),
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
    External(&'a mut M::Key, &'a mut M::Value),
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

impl<M> InternalNode<M> where M: MapTrait {
    #[inline]
    pub(crate) fn new() -> Self {
        InternalNode {
            count: 0,
            children: Children::from_fn(|_| TrieNode::Nothing),
        }
    }
}
