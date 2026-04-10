use std::mem::{ManuallyDrop, MaybeUninit};

use crate::{inner::Inner, map_trait::{Children, MapTrait, MaybeValue}};

// An internal node holds SIZE child nodes, which may themselves contain more internal nodes.
//
// Throughout this implementation, "idx" is used to refer to a section of key that is used
// to access a node. The layer of the tree directly below the root corresponds to idx 0.
pub struct InternalNode<M: MapTrait> {
    // The number of direct children which are external (i.e. that store a value).
    pub(crate) count: M::Count,
    pub(crate) value: M::MaybeValue,
    pub(crate) skip: usize,
    pub(crate) maybe_key: MaybeUninit<M::Key>,
    pub(crate) children: M::Children<TrieNode<M>>,
}

impl<M: MapTrait> Clone for InternalNode<M>
where
    M::Key: Clone,
    M::Value: Clone,
{
    fn clone(&self) -> Self {
        unsafe {
            Self {
                count: self.count,
                skip: self.skip,
                maybe_key: if self.skip == 0 { MaybeUninit::uninit() } else { MaybeUninit::new(self.maybe_key.assume_init_read()) },
                value: self.value.as_ref().cloned().into(),
                children: M::Children::from_fn(|i| self.children.as_slice()[i].clone()),
            }
        }
    }
}

// Each child of an InternalNode may be internal, in which case nesting continues,
// external (containing a value), or empty.
// The root of the Map is also one of these variants.
pub enum TrieNode<M: MapTrait> {
    Internal(Box<InternalNode<M>>),
    External(M::MaybeInner),
    Nothing,
}

impl<M: MapTrait> Clone for TrieNode<M>
where
    M::Key: Clone,
    M::Value: Clone,
{
    fn clone(&self) -> Self {
        match self {
            TrieNode::External(maybe_inner) => TrieNode::External(maybe_inner.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            TrieNode::Internal(b) => TrieNode::Internal((*b).clone()),
            TrieNode::Nothing => TrieNode::Nothing,
        }
    }
}

impl<M> InternalNode<M>
where
    M: MapTrait,
{
    #[inline]
    pub(crate) fn new() -> Self {
        InternalNode {
            count: Default::default(),
            value: None.into(),
            skip: 0,
            maybe_key: MaybeUninit::uninit(),
            children: Children::from_fn(|_| TrieNode::Nothing),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.children.as_slice().iter().all(|ch| matches!(ch, &TrieNode::Nothing))
    }
}
