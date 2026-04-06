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

impl<M: MapTrait> Clone for InternalNode<M>
where
    M::Key: Clone,
    M::Value: Clone,
    M::MaybeInner: Clone,
{
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
    External(M::MaybeInner),
    Nothing,
}

impl<M: MapTrait> Clone for TrieNode<M>
where
    M::Key: Clone,
    M::Value: Clone,
    M::MaybeInner: Clone,
{
    fn clone(&self) -> Self {
        match self {
            TrieNode::External(maybe_inner) => TrieNode::External(maybe_inner.clone()),
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
            count: 0,
            children: Children::from_fn(|_| TrieNode::Nothing),
        }
    }
}
