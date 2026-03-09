use crate::perf_hint::PerfHint;
use crate::Chunk;

// An internal node holds SIZE child nodes, which may themselves contain more internal nodes.
//
// Throughout this implementation, "idx" is used to refer to a section of key that is used
// to access a node. The layer of the tree directly below the root corresponds to idx 0.
#[derive(Clone)]
pub(crate) struct InternalNode<K, V, P: PerfHint<K, V>> where K: Chunk, [(); P::FANOUT as usize]: Sized {
    // The number of direct children which are external (i.e. that store a value).
    pub(crate) count: usize,
    pub(crate) children: [TrieNode<K, V, P>; P::FANOUT as usize],
}

// Each child of an InternalNode may be internal, in which case nesting continues,
// external (containing a value), or empty.
// The root of the Map is also one of these variants.
#[derive(Clone)]
pub(crate) enum TrieNode<K, V, P: PerfHint<K, V>> where K: Chunk, [(); P::FANOUT as usize]: Sized {
    Internal(Box<InternalNode<K, V, P>>),
    External(K, V),
    Nothing,
}

pub(crate) enum AnyNode<K, V, P> where K: Chunk, P: PerfHint<K, V>, [(); P::FANOUT as usize]: Sized {
    Internal(Box<InternalNode<K, V, P>>),
    External(K, V),
    Branch {
        count: usize,
        children: [TrieNode<K, V, P>; P::FANOUT as usize],
    },
    Nothing,
}
