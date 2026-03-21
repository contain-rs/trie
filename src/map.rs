// Copyright 2013-2026 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An ordered map based on a trie.

use crate::Chunk;
use crate::map_trait::{Children, MapTrait};
use crate::map_trait::{Root, RootChoice};
use crate::map_trait::RootNode;

use crate::node::{AnyNode, EitherNodeRef};
use crate::node::AnyNodeMut;
use crate::node::AnyNodeRef;
use crate::node::EitherNode;
use crate::node::InternalNode;
// pub use self::Entry::*;
use crate::node::TrieNode::{self, *};

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ops::{self, Bound};
use std::slice;

/// A map implemented as a radix trie.
///
/// Keys are split into sequences of 4 bits, which are used to place elements in
/// 16-entry arrays which are nested to form a tree structure. Inserted elements are placed
/// as close to the top of the tree as possible. The most significant bits of the key are used to
/// assign the key to a node/bucket in the first layer. If there are no other elements keyed by
/// the same 4 bits in the first layer, a leaf node will be created in the first layer.
/// When keys coincide, the next 4 bits are used to assign the node to a bucket in the next layer,
/// with this process continuing until an empty spot is found or there are no more bits left in the
/// key. As a result, the maximum depth using 32-bit `usize` keys is 8. The worst collisions occur
/// for very small numbers. For example, 1 and 2 are identical in all but their least significant
/// 4 bits. If both numbers are used as keys, a chain of maximum length will be created to
/// differentiate them.
///
/// # Examples
///
/// ```
/// let mut map = trie::Map::new();
/// map.insert(27, "Olaf");
/// map.insert(1, "Edgar");
/// map.insert(13, "Ruth");
/// map.insert(1, "Martin");
///
/// assert_eq!(map.len(), 3);
/// assert_eq!(map.get(&1), Some(&"Martin"));
///
/// if !map.contains_key(&90) {
///     println!("Nobody is keyed 90");
/// }
///
/// // Update a key
/// match map.get_mut(&1) {
///     Some(value) => *value = "Olga",
///     None => (),
/// }
///
/// map.remove(&13);
/// assert_eq!(map.len(), 2);
///
/// // Print the key value pairs, ordered by key.
/// for (key, value) in map.iter() {
///     // Prints `1: Olga` then `27: Olaf`
///     println!("{}: {}", key, value);
/// }
///
/// map.clear();
/// assert!(map.is_empty());
/// ```
pub struct Map<M> where M: MapTrait {
    root: M::Root,
    length: usize,
}

impl<M: MapTrait<Key: Clone, Value: Clone, Children<TrieNode<M>>: Clone>> Clone for Map<M> {
    fn clone(&self) -> Self {
        Map {
            length: self.length,
            root: M::Root::from_any_ref(self.root.into_any_ref()),
        }
    }
}

impl<M: MapTrait<Value: PartialEq>> PartialEq for Map<M> {
    fn eq(&self, other: &Map<M>) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<M: MapTrait<Value: Eq>> Eq for Map<M> {}

// impl<K: PartialOrd + Chunk, V: PartialOrd> PartialOrd for Map<K, V> {
//     #[inline]
//     fn partial_cmp(&self, other: &Map<K, V>) -> Option<Ordering> {
//         self.iter().partial_cmp(other.iter())
//     }
// }

// impl<K: Ord + Chunk, V: Ord> Ord for Map<K, V> {
//     #[inline]
//     fn cmp(&self, other: &Map<K, V>) -> Ordering {
//         self.iter().cmp(other.iter())
//     }
// }

// impl<K: Debug + Chunk, V: Debug> Debug for Map<K, V> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_map().entries(self.iter()).finish()
//     }
// }

#[cfg(feature = "extra")]
impl<K, V, P> Default for Map<K, V, P> where K: Chunk, P: PerfHint<K, V> {
    #[inline]
    fn default() -> Map<K, V, P> {
        Map::new()
    }
}

impl<M: MapTrait> Map<M> {
    /// Creates an empty map.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut map: trie::Map<usize, &str> = trie::Map::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Map {
            root: M::Root::nothing(),
            length: 0,
        }
    }
}

impl<M: MapTrait> Map<M> {
    /// Visits all key-value pairs in reverse order. Aborts traversal when `f` returns `false`.
    /// Returns `true` if `f` returns `true` for all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let map: trie::Map<usize, &str> = [(1, "a"), (2, "b"), (3, "c")].iter().cloned().collect();
    ///
    /// let mut vec = vec![];
    /// assert_eq!(true, map.each_reverse(|&key, &value| { vec.push((key, value)); true }));
    /// assert_eq!(vec, [(3, "c"), (2, "b"), (1, "a")]);
    ///
    /// // Stop when we reach 2
    /// let mut vec = vec![];
    /// assert_eq!(false, map.each_reverse(|&key, &value| { vec.push(value); key != 2 }));
    /// assert_eq!(vec, ["c", "b"]);
    /// ```
    // #[inline]
    // pub fn each_reverse<'a, F>(&'a self, mut f: F) -> bool
    // where
    //     F: FnMut(&K, &'a V) -> bool,
    // {
    //     // Root is now a TrieNode, so delegate to the node-level helper directly.
    //     node_each_reverse(&self.root, &mut f)
    // }

    /// Gets an iterator visiting all keys in ascending order by the keys.
    /// The iterator's element type is `usize`.
#[cfg(feature = "extra")]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys(self.iter())
    }

    /// Gets an iterator visiting all values in ascending order by the keys.
    /// The iterator's element type is `&'r T`.
#[cfg(feature = "extra")]
    pub fn values(&self) -> Values<'_, K, V> {
        Values(self.iter())
    }

    /// Gets an iterator over the key-value pairs in the map, ordered by keys.
    ///
    /// # Examples
    ///
    /// ```
    /// let map: trie::Map<usize, &str> = [(3, "c"), (1, "a"), (2, "b")].iter().cloned().collect();
    ///
    /// for (key, value) in map.iter() {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<'_, M> {
        let mut iter = unsafe { Iter::new() };
        // Instead of pushing root.children directly (root was always Internal),
        // we wrap the root node in a slice iterator via std::slice::from_ref.
        let slice_iter = match self.root.as_either_ref() {
            EitherNodeRef::Internal(&InternalNode { count: _, ref children }) => {
                children.as_slice().iter()
            }
            EitherNodeRef::Trie(trie_node) => slice::from_ref(trie_node).iter()
        };
        iter.stack.push(slice_iter);
        iter.remaining = self.length;
        iter
    }

    /// Gets an iterator over the key-value pairs in the map, with the
    /// ability to mutate the values.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut map: trie::Map<usize, i32> = [(1, 2), (2, 4), (3, 6)].iter().cloned().collect();
    ///
    /// for (&key, value) in map.iter_mut() {
    ///     *value = -(key as i32);
    /// }
    ///
    /// assert_eq!(map.get(&1), Some(&-1));
    /// assert_eq!(map.get(&2), Some(&-2));
    /// assert_eq!(map.get(&3), Some(&-3));
    /// ```
#[cfg(feature = "extra")]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let mut iter = unsafe { IterMut::new() };
        iter.stack.push(std::slice::from_mut(&mut self.root).iter_mut());
        iter.remaining = self.length;
        iter
    }
}


impl<M: MapTrait> Map<M> {
    /// Return the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut a = trie::Map::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Return true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut a = trie::Map::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the map, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut a = trie::Map::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.root = M::Root::nothing();
        self.length = 0;
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut map = trie::Map::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&M::Value>
    where
        M::Key: Borrow<Q>,
        Q: Chunk,
    {
        // Root is now a TrieNode: start traversal from it directly at idx 0.
        let mut node = self.root.into_any_ref();
        let mut idx = 0;

        loop {
            match node {
                AnyNodeRef::Branch { children, .. } => {
                    node = children[key.chunk(idx, M::SHIFT) as usize].into_any_ref();
                    idx += M::SHIFT as u64;
                }
                AnyNodeRef::External(k, v) => {
                    if k.borrow() == key {
                        return Some(v);
                    } else {
                        return None;
                    }
                }
                AnyNodeRef::Nothing => return None,
            }
        }
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut map = trie::Map::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        M::Key: Borrow<Q>,
        Q: Chunk,
    {
        self.get(key).is_some()
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut map = trie::Map::new();
    /// map.insert(1, "a");
    /// match map.get_mut(&1) {
    ///     Some(x) => *x = "b",
    ///     None => (),
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut M::Value>
    where
        M::Key: Borrow<Q>,
        Q: Chunk,
    {
        // Root is now the first node to check, at idx 0.
        find_mut(self.root.as_any_mut(), key, 0)
    }

    /// Inserts a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut map = trie::Map::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// ```
    pub fn insert(&mut self, key: M::Key, value: M::Value) -> Option<M::Value> {
        // root_count is a scratch counter used only to satisfy insert()'s signature;
        // the real element count is self.length.
        let mut root_count: usize = 0;
        let (_, old_val) = match self.root.as_either() {
            EitherNode::Trie(node) => {
                insert(&mut root_count, node, key, value, 0)
            }
            EitherNode::Internal(internal) => {
                insert(&mut internal.count, &mut internal.children[key.chunk(0, M::SHIFT) as usize], key, value, M::SHIFT as u64)
            }
        };
        if old_val.is_none() {
            self.length += 1;
        }
        old_val
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut map = trie::Map::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<M::Value>
    where
        M::Key: Borrow<Q>,
        Q: Chunk,
    {
        let mut root_count: usize = 0;
        let ret = match self.root.as_either() {
            EitherNode::Trie(node) => {
                remove(&mut root_count, node, key, 0)
            }
            EitherNode::Internal(internal) => {
                remove(&mut internal.count, &mut internal.children[key.chunk(0, M::SHIFT) as usize], key, M::SHIFT as u64)
            }
        };
        if ret.is_some() {
            self.length -= 1;
        }
        ret
    }
}

// macro_rules! bound {
//     (
//         $iterator_name:ident,
//         self = $this:expr,
//         key = $key:expr,
//         is_upper = $upper:expr,
//         iter = $iter:ident,
//         mutability = ($($mut_:tt)*),
//         const = ($($const_:tt)*)
//     ) => {
//         {
//             // We need an unsafe pointer here because we are borrowing
//             // mutable references to the internals of each of these
//             // mutable nodes, while still using the outer node.
//             //
//             // However, we're allowed to flaunt rustc like this because we
//             // never actually modify the "shape" of the nodes. The only
//             // place that mutation can actually occur is of the actual
//             // values of the map (as the return value of the iterator),
//             // i.e. we can never cause a deallocation of any InternalNodes
//             // so the raw pointer is always valid.
//             let this = $this;
//             let key = $key;

//             // let mut it = unsafe { $iterator_name::new() };
//             // it.remaining = this.length;

//             // Start from the root TrieNode itself (idx 0) rather than from
//             // an InternalNode's children array.
//             // let mut current_node: *$($mut_)* $($const_)* TrieNode<K, V> =
//                 // & $($mut_)* this.root as *$($mut_)* $($const_)* TrieNode<K, V>;

//             // Real implementation: mirrors the original but starts from the root TrieNode.
//             // Reset and redo properly.
//             let mut it = unsafe { $iterator_name::new() };
//             it.remaining = this.length;
//             it
//         }
//     }
// }

// impl<K: Chunk, V, P> Map<K, V, P> {
//     /// Returns a [`Cursor`] pointing at the gap before the smallest key
//     /// greater than the given bound.
//     ///
//     /// Passing `Bound::Included(x)` will return a cursor pointing to the
//     /// gap before the smallest key greater than or equal to `x`.
//     ///
//     /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
//     /// gap before the smallest key greater than `x`.
//     ///
//     /// Passing `Bound::Unbounded` will return a cursor pointing to the
//     /// gap before the smallest key in the map.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// #![feature(btree_cursors)]
//     ///
//     /// use std::collections::BTreeMap;
//     /// use std::ops::Bound;
//     ///
//     /// let map = BTreeMap::from([
//     ///     (1, "a"),
//     ///     (2, "b"),
//     ///     (3, "c"),
//     ///     (4, "d"),
//     /// ]);
//     ///
//     /// let cursor = map.lower_bound(Bound::Included(&2));
//     /// assert_eq!(cursor.peek_prev(), Some((&1, &"a")));
//     /// assert_eq!(cursor.peek_next(), Some((&2, &"b")));
//     ///
//     /// let cursor = map.lower_bound(Bound::Excluded(&2));
//     /// assert_eq!(cursor.peek_prev(), Some((&2, &"b")));
//     /// assert_eq!(cursor.peek_next(), Some((&3, &"c")));
//     ///
//     /// let cursor = map.lower_bound(Bound::Unbounded);
//     /// assert_eq!(cursor.peek_prev(), None);
//     /// assert_eq!(cursor.peek_next(), Some((&1, &"a")));
//     /// ```
//     pub fn lower_bound<Q: ?Sized>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V>
//     where
//         K: Borrow<Q> + Ord,
//         Q: Ord,
//     {
//         let root_node = match self.root.as_ref() {
//             None => return Cursor { current: None, root: None },
//             Some(root) => root.reborrow(),
//         };
//         let edge = root_node.lower_bound(SearchBound::from_range(bound));
//         Cursor { current: Some(edge), root: self.root.as_ref() }
//     }

//     /// Returns a [`CursorMut`] pointing at the gap before the smallest key
//     /// greater than the given bound.
//     ///
//     /// Passing `Bound::Included(x)` will return a cursor pointing to the
//     /// gap before the smallest key greater than or equal to `x`.
//     ///
//     /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
//     /// gap before the smallest key greater than `x`.
//     ///
//     /// Passing `Bound::Unbounded` will return a cursor pointing to the
//     /// gap before the smallest key in the map.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// #![feature(btree_cursors)]
//     ///
//     /// use std::collections::BTreeMap;
//     /// use std::ops::Bound;
//     ///
//     /// let mut map = BTreeMap::from([
//     ///     (1, "a"),
//     ///     (2, "b"),
//     ///     (3, "c"),
//     ///     (4, "d"),
//     /// ]);
//     ///
//     /// let mut cursor = map.lower_bound_mut(Bound::Included(&2));
//     /// assert_eq!(cursor.peek_prev(), Some((&1, &mut "a")));
//     /// assert_eq!(cursor.peek_next(), Some((&2, &mut "b")));
//     ///
//     /// let mut cursor = map.lower_bound_mut(Bound::Excluded(&2));
//     /// assert_eq!(cursor.peek_prev(), Some((&2, &mut "b")));
//     /// assert_eq!(cursor.peek_next(), Some((&3, &mut "c")));
//     ///
//     /// let mut cursor = map.lower_bound_mut(Bound::Unbounded);
//     /// assert_eq!(cursor.peek_prev(), None);
//     /// assert_eq!(cursor.peek_next(), Some((&1, &mut "a")));
//     /// ```
//     pub fn lower_bound_mut<Q: ?Sized>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V>
//     where
//         K: Borrow<Q> + Ord,
//         Q: Ord,
//     {
//         let (root, dormant_root) = DormantMutRef::new(&mut self.root);
//         let root_node = match root.as_mut() {
//             None => {
//                 return CursorMut {
//                     inner: CursorMutKey {
//                         current: None,
//                         root: dormant_root,
//                         length: &mut self.length,
//                         alloc: &mut *self.alloc,
//                     },
//                 };
//             }
//             Some(root) => root.borrow_mut(),
//         };
//         let edge = root_node.lower_bound(SearchBound::from_range(bound));
//         CursorMut {
//             inner: CursorMutKey {
//                 current: Some(edge),
//                 root: dormant_root,
//                 length: &mut self.length,
//                 alloc: &mut *self.alloc,
//             },
//         }
//     }

//     /// Returns a [`Cursor`] pointing at the gap after the greatest key
//     /// smaller than the given bound.
//     ///
//     /// Passing `Bound::Included(x)` will return a cursor pointing to the
//     /// gap after the greatest key smaller than or equal to `x`.
//     ///
//     /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
//     /// gap after the greatest key smaller than `x`.
//     ///
//     /// Passing `Bound::Unbounded` will return a cursor pointing to the
//     /// gap after the greatest key in the map.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// #![feature(btree_cursors)]
//     ///
//     /// use std::collections::BTreeMap;
//     /// use std::ops::Bound;
//     ///
//     /// let map = BTreeMap::from([
//     ///     (1, "a"),
//     ///     (2, "b"),
//     ///     (3, "c"),
//     ///     (4, "d"),
//     /// ]);
//     ///
//     /// let cursor = map.upper_bound(Bound::Included(&3));
//     /// assert_eq!(cursor.peek_prev(), Some((&3, &"c")));
//     /// assert_eq!(cursor.peek_next(), Some((&4, &"d")));
//     ///
//     /// let cursor = map.upper_bound(Bound::Excluded(&3));
//     /// assert_eq!(cursor.peek_prev(), Some((&2, &"b")));
//     /// assert_eq!(cursor.peek_next(), Some((&3, &"c")));
//     ///
//     /// let cursor = map.upper_bound(Bound::Unbounded);
//     /// assert_eq!(cursor.peek_prev(), Some((&4, &"d")));
//     /// assert_eq!(cursor.peek_next(), None);
//     /// ```
//     pub fn upper_bound<Q: ?Sized>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V>
//     where
//         K: Borrow<Q> + Ord,
//         Q: Ord,
//     {
//         let root_node = match self.root.as_ref() {
//             None => return Cursor { current: None, root: None },
//             Some(root) => root.reborrow(),
//         };
//         let edge = root_node.upper_bound(SearchBound::from_range(bound));
//         Cursor { current: Some(edge), root: self.root.as_ref() }
//     }

//     /// Returns a [`CursorMut`] pointing at the gap after the greatest key
//     /// smaller than the given bound.
//     ///
//     /// Passing `Bound::Included(x)` will return a cursor pointing to the
//     /// gap after the greatest key smaller than or equal to `x`.
//     ///
//     /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
//     /// gap after the greatest key smaller than `x`.
//     ///
//     /// Passing `Bound::Unbounded` will return a cursor pointing to the
//     /// gap after the greatest key in the map.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// #![feature(btree_cursors)]
//     ///
//     /// use std::collections::BTreeMap;
//     /// use std::ops::Bound;
//     ///
//     /// let mut map = BTreeMap::from([
//     ///     (1, "a"),
//     ///     (2, "b"),
//     ///     (3, "c"),
//     ///     (4, "d"),
//     /// ]);
//     ///
//     /// let mut cursor = map.upper_bound_mut(Bound::Included(&3));
//     /// assert_eq!(cursor.peek_prev(), Some((&3, &mut "c")));
//     /// assert_eq!(cursor.peek_next(), Some((&4, &mut "d")));
//     ///
//     /// let mut cursor = map.upper_bound_mut(Bound::Excluded(&3));
//     /// assert_eq!(cursor.peek_prev(), Some((&2, &mut "b")));
//     /// assert_eq!(cursor.peek_next(), Some((&3, &mut "c")));
//     ///
//     /// let mut cursor = map.upper_bound_mut(Bound::Unbounded);
//     /// assert_eq!(cursor.peek_prev(), Some((&4, &mut "d")));
//     /// assert_eq!(cursor.peek_next(), None);
//     /// ```
//     pub fn upper_bound_mut<Q: ?Sized>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V>
//     where
//         K: Borrow<Q> + Ord,
//         Q: Ord,
//     {
//         let (root, dormant_root) = DormantMutRef::new(&mut self.root);
//         let root_node = match root.as_mut() {
//             None => {
//                 return CursorMut {
//                     inner: CursorMutKey {
//                         current: None,
//                         root: dormant_root,
//                         length: &mut self.length,
//                         alloc: &mut *self.alloc,
//                     },
//                 };
//             }
//             Some(root) => root.borrow_mut(),
//         };
//         let edge = root_node.upper_bound(SearchBound::from_range(bound));
//         CursorMut {
//             inner: CursorMutKey {
//                 current: Some(edge),
//                 root: dormant_root,
//                 length: &mut self.length,
//                 alloc: &mut *self.alloc,
//             },
//         }
//     }
// }

// /// A cursor over a `BTreeMap`.
// ///
// /// A `Cursor` is like an iterator, except that it can freely seek back-and-forth.
// ///
// /// Cursors always point to a gap between two elements in the map, and can
// /// operate on the two immediately adjacent elements.
// ///
// /// A `Cursor` is created with the [`BTreeMap::lower_bound`] and [`BTreeMap::upper_bound`] methods.
// struct Cursor<'a, K: 'a, V: 'a> {
//     // If current is None then it means the tree has not been allocated yet.
//     current: Option<Handle<NodeRef<marker::Immut<'a>, K, V, marker::Leaf>, marker::Edge>>,
//     root: Option<&'a node::Root<K, V>>,
// }

// impl <K, V> Clone for Cursor<'_, K, V> {
//     fn clone(&self) -> Self {
//         let Cursor { current, root } = *self;
//         Cursor { current, root }
//     }
// }

// impl <K: Debug, V: Debug> Debug for Cursor<'_, K, V> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.write_str("Cursor")
//     }
// }

// /// A cursor over a `BTreeMap` with editing operations.
// ///
// /// A `Cursor` is like an iterator, except that it can freely seek back-and-forth, and can
// /// safely mutate the map during iteration. This is because the lifetime of its yielded
// /// references is tied to its own lifetime, instead of just the underlying map. This means
// /// cursors cannot yield multiple elements at once.
// ///
// /// Cursors always point to a gap between two elements in the map, and can
// /// operate on the two immediately adjacent elements.
// ///
// /// A `CursorMut` is created with the [`BTreeMap::lower_bound_mut`] and [`BTreeMap::upper_bound_mut`]
// /// methods.
// struct CursorMut<
//     'a,
//     K: 'a,
//     V: 'a> {
//     inner: CursorMutKey<'a, K, V>,
// }

// impl<K: Debug, V: Debug, A> Debug for CursorMut<'_, K, V, A> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.write_str("CursorMut")
//     }
// }

// /// A cursor over a `BTreeMap` with editing operations, and which allows
// /// mutating the key of elements.
// ///
// /// A `Cursor` is like an iterator, except that it can freely seek back-and-forth, and can
// /// safely mutate the map during iteration. This is because the lifetime of its yielded
// /// references is tied to its own lifetime, instead of just the underlying map. This means
// /// cursors cannot yield multiple elements at once.
// ///
// /// Cursors always point to a gap between two elements in the map, and can
// /// operate on the two immediately adjacent elements.
// ///
// /// A `CursorMutKey` is created from a [`CursorMut`] with the
// /// [`CursorMut::with_mutable_key`] method.
// ///
// /// # Safety
// ///
// /// Since this cursor allows mutating keys, you must ensure that the `BTreeMap`
// /// invariants are maintained. Specifically:
// ///
// /// * The key of the newly inserted element must be unique in the tree.
// /// * All keys in the tree must remain in sorted order.
// struct CursorMutKey<
//     'a,
//     K: 'a,
//     V: 'a,
// > {
//     // If current is None then it means the tree has not been allocated yet.
//     current: Option<Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>>,
//     root: DormantMutRef<'a, Option<node::Root<K, V>>>,
//     length: &'a mut usize,
//     alloc: &'a mut A,
// }

// impl<K: Debug, V: Debug, A> Debug for CursorMutKey<'_, K, V, A> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.write_str("CursorMutKey")
//     }
// }

// impl<'a, K, V> Cursor<'a, K, V> {
//     /// Advances the cursor to the next gap, returning the key and value of the
//     /// element that it moved over.
//     ///
//     /// If the cursor is already at the end of the map then `None` is returned
//     /// and the cursor is not moved.
//     pub fn next(&mut self) -> Option<(&'a K, &'a V)> {
//         let current = self.current.take()?;
//         match current.next_kv() {
//             Ok(kv) => {
//                 let result = kv.into_kv();
//                 self.current = Some(kv.next_leaf_edge());
//                 Some(result)
//             }
//             Err(root) => {
//                 self.current = Some(root.last_leaf_edge());
//                 None
//             }
//         }
//     }

//     /// Advances the cursor to the previous gap, returning the key and value of
//     /// the element that it moved over.
//     ///
//     /// If the cursor is already at the start of the map then `None` is returned
//     /// and the cursor is not moved.
//     pub fn prev(&mut self) -> Option<(&'a K, &'a V)> {
//         let current = self.current.take()?;
//         match current.next_back_kv() {
//             Ok(kv) => {
//                 let result = kv.into_kv();
//                 self.current = Some(kv.next_back_leaf_edge());
//                 Some(result)
//             }
//             Err(root) => {
//                 self.current = Some(root.first_leaf_edge());
//                 None
//             }
//         }
//     }

//     /// Returns a reference to the key and value of the next element without
//     /// moving the cursor.
//     ///
//     /// If the cursor is at the end of the map then `None` is returned.
//     pub fn peek_next(&self) -> Option<(&'a K, &'a V)> {
//         self.clone().next()
//     }

//     /// Returns a reference to the key and value of the previous element
//     /// without moving the cursor.
//     ///
//     /// If the cursor is at the start of the map then `None` is returned.
//     pub fn peek_prev(&self) -> Option<(&'a K, &'a V)> {
//         self.clone().prev()
//     }
// }

// impl<'a, K, V, A> CursorMut<'a, K, V, A> {
//     /// Advances the cursor to the next gap, returning the key and value of the
//     /// element that it moved over.
//     ///
//     /// If the cursor is already at the end of the map then `None` is returned
//     /// and the cursor is not moved.
//     pub fn next(&mut self) -> Option<(&K, &mut V)> {
//         let (k, v) = self.inner.next()?;
//         Some((&*k, v))
//     }

//     /// Advances the cursor to the previous gap, returning the key and value of
//     /// the element that it moved over.
//     ///
//     /// If the cursor is already at the start of the map then `None` is returned
//     /// and the cursor is not moved.
//     pub fn prev(&mut self) -> Option<(&K, &mut V)> {
//         let (k, v) = self.inner.prev()?;
//         Some((&*k, v))
//     }

//     /// Returns a reference to the key and value of the next element without
//     /// moving the cursor.
//     ///
//     /// If the cursor is at the end of the map then `None` is returned.
//     pub fn peek_next(&mut self) -> Option<(&K, &mut V)> {
//         let (k, v) = self.inner.peek_next()?;
//         Some((&*k, v))
//     }

//     /// Returns a reference to the key and value of the previous element
//     /// without moving the cursor.
//     ///
//     /// If the cursor is at the start of the map then `None` is returned.
//     pub fn peek_prev(&mut self) -> Option<(&K, &mut V)> {
//         let (k, v) = self.inner.peek_prev()?;
//         Some((&*k, v))
//     }

//     /// Returns a read-only cursor pointing to the same location as the
//     /// `CursorMut`.
//     ///
//     /// The lifetime of the returned `Cursor` is bound to that of the
//     /// `CursorMut`, which means it cannot outlive the `CursorMut` and that the
//     /// `CursorMut` is frozen for the lifetime of the `Cursor`.
//     pub fn as_cursor(&self) -> Cursor<'_, K, V> {
//         self.inner.as_cursor()
//     }

//     /// Converts the cursor into a [`CursorMutKey`], which allows mutating
//     /// the key of elements in the tree.
//     ///
//     /// # Safety
//     ///
//     /// Since this cursor allows mutating keys, you must ensure that the `BTreeMap`
//     /// invariants are maintained. Specifically:
//     ///
//     /// * The key of the newly inserted element must be unique in the tree.
//     /// * All keys in the tree must remain in sorted order.
//     pub unsafe fn with_mutable_key(self) -> CursorMutKey<'a, K, V, A> {
//         self.inner
//     }
// }

// impl<'a, K, V, A> CursorMutKey<'a, K, V, A> {
//     /// Advances the cursor to the next gap, returning the key and value of the
//     /// element that it moved over.
//     ///
//     /// If the cursor is already at the end of the map then `None` is returned
//     /// and the cursor is not moved.
//     pub fn next(&mut self) -> Option<(&mut K, &mut V)> {
//         let current = self.current.take()?;
//         match current.next_kv() {
//             Ok(mut kv) => {
//                 // SAFETY: The key/value pointers remain valid even after the
//                 // cursor is moved forward. The lifetimes then prevent any
//                 // further access to the cursor.
//                 let (k, v) = unsafe { kv.reborrow_mut().into_kv_mut() };
//                 let (k, v) = (k as *mut _, v as *mut _);
//                 self.current = Some(kv.next_leaf_edge());
//                 Some(unsafe { (&mut *k, &mut *v) })
//             }
//             Err(root) => {
//                 self.current = Some(root.last_leaf_edge());
//                 None
//             }
//         }
//     }

//     /// Advances the cursor to the previous gap, returning the key and value of
//     /// the element that it moved over.
//     ///
//     /// If the cursor is already at the start of the map then `None` is returned
//     /// and the cursor is not moved.
//     pub fn prev(&mut self) -> Option<(&mut K, &mut V)> {
//         let current = self.current.take()?;
//         match current.next_back_kv() {
//             Ok(mut kv) => {
//                 // SAFETY: The key/value pointers remain valid even after the
//                 // cursor is moved forward. The lifetimes then prevent any
//                 // further access to the cursor.
//                 let (k, v) = unsafe { kv.reborrow_mut().into_kv_mut() };
//                 let (k, v) = (k as *mut _, v as *mut _);
//                 self.current = Some(kv.next_back_leaf_edge());
//                 Some(unsafe { (&mut *k, &mut *v) })
//             }
//             Err(root) => {
//                 self.current = Some(root.first_leaf_edge());
//                 None
//             }
//         }
//     }

//     /// Returns a reference to the key and value of the next element without
//     /// moving the cursor.
//     ///
//     /// If the cursor is at the end of the map then `None` is returned.
//     pub fn peek_next(&mut self) -> Option<(&mut K, &mut V)> {
//         let current = self.current.as_mut()?;
//         // SAFETY: We're not using this to mutate the tree.
//         let kv = unsafe { current.reborrow_mut() }.next_kv().ok()?.into_kv_mut();
//         Some(kv)
//     }

//     /// Returns a reference to the key and value of the previous element
//     /// without moving the cursor.
//     ///
//     /// If the cursor is at the start of the map then `None` is returned.
//     pub fn peek_prev(&mut self) -> Option<(&mut K, &mut V)> {
//         let current = self.current.as_mut()?;
//         // SAFETY: We're not using this to mutate the tree.
//         let kv = unsafe { current.reborrow_mut() }.next_back_kv().ok()?.into_kv_mut();
//         Some(kv)
//     }

//     /// Returns a read-only cursor pointing to the same location as the
//     /// `CursorMutKey`.
//     ///
//     /// The lifetime of the returned `Cursor` is bound to that of the
//     /// `CursorMutKey`, which means it cannot outlive the `CursorMutKey` and that the
//     /// `CursorMutKey` is frozen for the lifetime of the `Cursor`.
//     pub fn as_cursor(&self) -> Cursor<'_, K, V> {
//         Cursor {
//             // SAFETY: The tree is immutable while the cursor exists.
//             root: unsafe { self.root.reborrow_shared().as_ref() },
//             current: self.current.as_ref().map(|current| current.reborrow()),
//         }
//     }
// }

// // Now the tree editing operations
// impl<'a, K: Ord, V, A: Allocator + Clone> CursorMutKey<'a, K, V, A> {
//     /// Inserts a new key-value pair into the map in the gap that the
//     /// cursor is currently pointing to.
//     ///
//     /// After the insertion the cursor will be pointing at the gap before the
//     /// newly inserted element.
//     ///
//     /// # Safety
//     ///
//     /// You must ensure that the `BTreeMap` invariants are maintained.
//     /// Specifically:
//     ///
//     /// * The key of the newly inserted element must be unique in the tree.
//     /// * All keys in the tree must remain in sorted order.
//     pub unsafe fn insert_after_unchecked(&mut self, key: K, value: V) {
//         let edge = match self.current.take() {
//             None => {
//                 // Tree is empty, allocate a new root.
//                 // SAFETY: We have no other reference to the tree.
//                 let root = unsafe { self.root.reborrow() };
//                 debug_assert!(root.is_none());
//                 let mut node = NodeRef::new_leaf(self.alloc.clone());
//                 // SAFETY: We don't touch the root while the handle is alive.
//                 let handle = unsafe { node.borrow_mut().push_with_handle(key, value) };
//                 *root = Some(node.forget_type());
//                 *self.length += 1;
//                 self.current = Some(handle.left_edge());
//                 return;
//             }
//             Some(current) => current,
//         };

//         let handle = edge.insert_recursing(key, value, self.alloc.clone(), |ins| {
//             drop(ins.left);
//             // SAFETY: The handle to the newly inserted value is always on a
//             // leaf node, so adding a new root node doesn't invalidate it.
//             let root = unsafe { self.root.reborrow().as_mut().unwrap() };
//             root.push_internal_level(self.alloc.clone()).push(ins.kv.0, ins.kv.1, ins.right)
//         });
//         self.current = Some(handle.left_edge());
//         *self.length += 1;
//     }

//     /// Inserts a new key-value pair into the map in the gap that the
//     /// cursor is currently pointing to.
//     ///
//     /// After the insertion the cursor will be pointing at the gap after the
//     /// newly inserted element.
//     ///
//     /// # Safety
//     ///
//     /// You must ensure that the `BTreeMap` invariants are maintained.
//     /// Specifically:
//     ///
//     /// * The key of the newly inserted element must be unique in the tree.
//     /// * All keys in the tree must remain in sorted order.
//     pub unsafe fn insert_before_unchecked(&mut self, key: K, value: V) {
//         let edge = match self.current.take() {
//             None => {
//                 // SAFETY: We have no other reference to the tree.
//                 match unsafe { self.root.reborrow() } {
//                     root @ None => {
//                         // Tree is empty, allocate a new root.
//                         let mut node = NodeRef::new_leaf(self.alloc.clone());
//                         // SAFETY: We don't touch the root while the handle is alive.
//                         let handle = unsafe { node.borrow_mut().push_with_handle(key, value) };
//                         *root = Some(node.forget_type());
//                         *self.length += 1;
//                         self.current = Some(handle.right_edge());
//                         return;
//                     }
//                     Some(root) => root.borrow_mut().last_leaf_edge(),
//                 }
//             }
//             Some(current) => current,
//         };

//         let handle = edge.insert_recursing(key, value, self.alloc.clone(), |ins| {
//             drop(ins.left);
//             // SAFETY: The handle to the newly inserted value is always on a
//             // leaf node, so adding a new root node doesn't invalidate it.
//             let root = unsafe { self.root.reborrow().as_mut().unwrap() };
//             root.push_internal_level(self.alloc.clone()).push(ins.kv.0, ins.kv.1, ins.right)
//         });
//         self.current = Some(handle.right_edge());
//         *self.length += 1;
//     }

//     /// Inserts a new key-value pair into the map in the gap that the
//     /// cursor is currently pointing to.
//     ///
//     /// After the insertion the cursor will be pointing at the gap before the
//     /// newly inserted element.
//     ///
//     /// If the inserted key is not greater than the key before the cursor
//     /// (if any), or if it not less than the key after the cursor (if any),
//     /// then an [`UnorderedKeyError`] is returned since this would
//     /// invalidate the [`Ord`] invariant between the keys of the map.
//     pub fn insert_after(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError> {
//         if let Some((prev, _)) = self.peek_prev() {
//             if &key <= prev {
//                 return Err(UnorderedKeyError {});
//             }
//         }
//         if let Some((next, _)) = self.peek_next() {
//             if &key >= next {
//                 return Err(UnorderedKeyError {});
//             }
//         }
//         unsafe {
//             self.insert_after_unchecked(key, value);
//         }
//         Ok(())
//     }

//     /// Inserts a new key-value pair into the map in the gap that the
//     /// cursor is currently pointing to.
//     ///
//     /// After the insertion the cursor will be pointing at the gap after the
//     /// newly inserted element.
//     ///
//     /// If the inserted key is not greater than the key before the cursor
//     /// (if any), or if it not less than the key after the cursor (if any),
//     /// then an [`UnorderedKeyError`] is returned since this would
//     /// invalidate the [`Ord`] invariant between the keys of the map.
//     pub fn insert_before(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError> {
//         if let Some((prev, _)) = self.peek_prev() {
//             if &key <= prev {
//                 return Err(UnorderedKeyError {});
//             }
//         }
//         if let Some((next, _)) = self.peek_next() {
//             if &key >= next {
//                 return Err(UnorderedKeyError {});
//             }
//         }
//         unsafe {
//             self.insert_before_unchecked(key, value);
//         }
//         Ok(())
//     }

//     /// Removes the next element from the `BTreeMap`.
//     ///
//     /// The element that was removed is returned. The cursor position is
//     /// unchanged (before the removed element).
//     pub fn remove_next(&mut self) -> Option<(K, V)> {
//         let current = self.current.take()?;
//         if current.reborrow().next_kv().is_err() {
//             self.current = Some(current);
//             return None;
//         }
//         let mut emptied_internal_root = false;
//         let (kv, pos) = current
//             .next_kv()
//             // This should be unwrap(), but that doesn't work because NodeRef
//             // doesn't implement Debug. The condition is checked above.
//             .ok()?
//             .remove_kv_tracking(|| emptied_internal_root = true, self.alloc.clone());
//         self.current = Some(pos);
//         *self.length -= 1;
//         if emptied_internal_root {
//             // SAFETY: This is safe since current does not point within the now
//             // empty root node.
//             let root = unsafe { self.root.reborrow().as_mut().unwrap() };
//             root.pop_internal_level(self.alloc.clone());
//         }
//         Some(kv)
//     }

//     /// Removes the preceding element from the `BTreeMap`.
//     ///
//     /// The element that was removed is returned. The cursor position is
//     /// unchanged (after the removed element).
//     pub fn remove_prev(&mut self) -> Option<(K, V)> {
//         let current = self.current.take()?;
//         if current.reborrow().next_back_kv().is_err() {
//             self.current = Some(current);
//             return None;
//         }
//         let mut emptied_internal_root = false;
//         let (kv, pos) = current
//             .next_back_kv()
//             // This should be unwrap(), but that doesn't work because NodeRef
//             // doesn't implement Debug. The condition is checked above.
//             .ok()?
//             .remove_kv_tracking(|| emptied_internal_root = true, self.alloc.clone());
//         self.current = Some(pos);
//         *self.length -= 1;
//         if emptied_internal_root {
//             // SAFETY: This is safe since current does not point within the now
//             // empty root node.
//             let root = unsafe { self.root.reborrow().as_mut().unwrap() };
//             root.pop_internal_level(self.alloc.clone());
//         }
//         Some(kv)
//     }
// }

// impl<'a, K: Ord, V, A: Allocator + Clone> CursorMut<'a, K, V, A> {
//     /// Inserts a new key-value pair into the map in the gap that the
//     /// cursor is currently pointing to.
//     ///
//     /// After the insertion the cursor will be pointing at the gap after the
//     /// newly inserted element.
//     ///
//     /// # Safety
//     ///
//     /// You must ensure that the `BTreeMap` invariants are maintained.
//     /// Specifically:
//     ///
//     /// * The key of the newly inserted element must be unique in the tree.
//     /// * All keys in the tree must remain in sorted order.
//     pub unsafe fn insert_after_unchecked(&mut self, key: K, value: V) {
//         unsafe { self.inner.insert_after_unchecked(key, value) }
//     }

//     /// Inserts a new key-value pair into the map in the gap that the
//     /// cursor is currently pointing to.
//     ///
//     /// After the insertion the cursor will be pointing at the gap after the
//     /// newly inserted element.
//     ///
//     /// # Safety
//     ///
//     /// You must ensure that the `BTreeMap` invariants are maintained.
//     /// Specifically:
//     ///
//     /// * The key of the newly inserted element must be unique in the tree.
//     /// * All keys in the tree must remain in sorted order.
//     pub unsafe fn insert_before_unchecked(&mut self, key: K, value: V) {
//         unsafe { self.inner.insert_before_unchecked(key, value) }
//     }

//     /// Inserts a new key-value pair into the map in the gap that the
//     /// cursor is currently pointing to.
//     ///
//     /// After the insertion the cursor will be pointing at the gap before the
//     /// newly inserted element.
//     ///
//     /// If the inserted key is not greater than the key before the cursor
//     /// (if any), or if it not less than the key after the cursor (if any),
//     /// then an [`UnorderedKeyError`] is returned since this would
//     /// invalidate the [`Ord`] invariant between the keys of the map.
//     pub fn insert_after(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError> {
//         self.inner.insert_after(key, value)
//     }

//     /// Inserts a new key-value pair into the map in the gap that the
//     /// cursor is currently pointing to.
//     ///
//     /// After the insertion the cursor will be pointing at the gap after the
//     /// newly inserted element.
//     ///
//     /// If the inserted key is not greater than the key before the cursor
//     /// (if any), or if it not less than the key after the cursor (if any),
//     /// then an [`UnorderedKeyError`] is returned since this would
//     /// invalidate the [`Ord`] invariant between the keys of the map.
//     pub fn insert_before(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError> {
//         self.inner.insert_before(key, value)
//     }

//     /// Removes the next element from the `BTreeMap`.
//     ///
//     /// The element that was removed is returned. The cursor position is
//     /// unchanged (before the removed element).
//     pub fn remove_next(&mut self) -> Option<(K, V)> {
//         self.inner.remove_next()
//     }

//     /// Removes the preceding element from the `BTreeMap`.
//     ///
//     /// The element that was removed is returned. The cursor position is
//     /// unchanged (after the removed element).
//     pub fn remove_prev(&mut self) -> Option<(K, V)> {
//         self.inner.remove_prev()
//     }
// }

// /// Error type returned by [`CursorMut::insert_before`] and
// /// [`CursorMut::insert_after`] if the key being inserted is not properly
// /// ordered with regards to adjacent keys.
// #[derive(Clone, PartialEq, Eq, Debug)]
// struct UnorderedKeyError {}

// impl fmt::Display for UnorderedKeyError {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "key is not properly ordered relative to neighbors")
//     }
// }

// impl Error for UnorderedKeyError {}

// impl<K: Chunk, V> iter::FromIterator<(K, V)> for Map<K, V> {
//     fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Map<K, V> {
//         let mut map = Map::new();
//         map.extend(iter);
//         map
//     }
// }

// impl<K: Chunk, V> Extend<(K, V)> for Map<K, V> {
//     fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
//         for (k, v) in iter {
//             self.insert(k, v);
//         }
//     }
// }

// impl<K: Chunk + Hash, V: Hash> Hash for Map<K, V> {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         for elt in self.iter() {
//             elt.hash(state);
//         }
//     }
// }

impl<'a, Q, M> ops::Index<&'a Q> for Map<M>
where
    M::Key: Borrow<Q> + Chunk,
    Q: Chunk + 'a,
    M: MapTrait,
{
    type Output = M::Value;
    #[inline]
    fn index(&self, i: &'a Q) -> &Self::Output {
        self.get(i).expect("key not present")
    }
}

impl<'a, Q, M> ops::IndexMut<&'a Q> for Map<M>
where
    M::Key: Borrow<Q>,
    Q: Chunk + 'a,
    M: MapTrait,
{
    #[inline]
    fn index_mut(&mut self, i: &'a Q) -> &mut Self::Output {
        self.get_mut(i).expect("key not present")
    }
}

// // Standalone recursive helper so both InternalNode and the root TrieNode can use it.
// fn node_each_reverse<'a, K, V, F>(node: &'a TrieNode<K, V>, f: &mut F) -> bool
// where
//     F: FnMut(&'a K, &'a V) -> bool,
// {
//     match *node {
//         Internal(ref x) => {
//             for elt in x.children.iter().rev() {
//                 if !node_each_reverse(elt, f) {
//                     return false;
//                 }
//             }
//             true
//         }
//         External(ref k, ref v) => f(k, v),
//         Nothing => true,
//     }
// }

// // TODO: make the function non-recursive
fn find_mut<'a, M: MapTrait, Q: Chunk>(
    node: AnyNodeMut<'a, M>,
    key: &Q,
    idx: u64,
) -> Option<&'a mut M::Value>
where
    M::Key: Borrow<Q> + Chunk,
{
    match node {
        AnyNodeMut::External(stored, value) if (*stored).borrow() == key => Some(value),
        AnyNodeMut::External(..) => None,
        AnyNodeMut::Branch { children, .. } => find_mut(children[key.chunk(idx, M::SHIFT) as usize].as_any_mut(), key, idx + M::SHIFT as u64),
        AnyNodeMut::Nothing => None,
    }
}

/// Inserts a new node for the given key and value, at or below `start_node`.
///
/// The index (`idx`) is the chunk index used to reach `start_node` from its parent.
/// For the root, `idx` is 0.
///
/// `count` is the external-node counter for `start_node`'s parent; it is incremented
/// only when `start_node` transitions from Nothing to a new External node.
///
/// Returns a mutable reference to the inserted value and an optional previous value.
fn insert<'a, M: MapTrait>(
    count: &mut usize,
    start_node: &'a mut TrieNode<M>,
    key: M::Key,
    value: M::Value,
    idx: u64,
) -> (&'a mut M::Value, Option<M::Value>) where M::Key: Chunk {
    // We branch twice to avoid having to do the `replace` when we don't need to;
    // this is much faster, especially for keys that have long shared prefixes.

    let mut hack = false;
    match *start_node {
        Nothing => {
            *count += 1;
            *start_node = External(key, value);
            match *start_node {
                External(_, ref mut value_ref) => return (value_ref, None),
                _ => unreachable!(),
            }
        }
        Internal(ref mut x) => {
            let x = &mut **x;
            return insert(
                &mut x.count,
                &mut x.children[key.chunk(idx, M::SHIFT) as usize],
                key,
                value,
                idx + M::SHIFT as u64,
            );
        }
        External(ref stored_key, _) if stored_key == &key => {
            hack = true;
        }
        _ => {}
    }

    if !hack {
        // Conflict: an External node with a different key.
        // Replace it with a new Internal node and re-insert both values beneath it.
        match mem::replace(start_node, Internal(Box::new(InternalNode::new()))) {
            External(stored_key, stored_value) => {
                match *start_node {
                    Internal(ref mut new_node) => {
                        let new_node = &mut **new_node;
                        insert(
                            &mut new_node.count,
                            &mut new_node.children[stored_key.chunk(idx, M::SHIFT) as usize],
                            stored_key,
                            stored_value,
                            idx + M::SHIFT as u64,
                        );
                        return insert(
                            &mut new_node.count,
                            &mut new_node.children[key.chunk(idx, M::SHIFT) as usize],
                            key,
                            value,
                            idx + M::SHIFT as u64,
                        );
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }

    if let External(_, ref mut stored_value) = *start_node {
        let old_value = mem::replace(stored_value, value);
        return (stored_value, Some(old_value));
    }

    unreachable!();
}

// TODO: make the function non-recursive
fn remove<M: MapTrait, Q: Chunk>(
    count: &mut usize,
    child: &mut TrieNode<M>,
    key: &Q,
    idx: u64,
) -> Option<M::Value>
where
    M::Key: Borrow<Q> + Chunk,
{
    let (ret, this) = match *child {
        External(ref stored, _) if stored.borrow() == key => match mem::replace(child, Nothing) {
            External(_, value) => (Some(value), true),
            _ => unreachable!(),
        },
        External(..) => (None, false),
        Internal(ref mut x) => {
            let x = &mut **x;
            let ret = remove(&mut x.count, &mut x.children[key.chunk(idx, M::SHIFT) as usize], key, idx + M::SHIFT as u64);
            (ret, x.count == 0)
        }
        Nothing => (None, false),
    };

    if this {
        *child = Nothing;
        *count -= 1;
    }
    ret
}

// /// A view into a single entry in a map, which may be vacant or occupied.
// pub enum Entry<'a, K: 'a, V: 'a> {
//     /// An occupied entry.
//     Occupied(OccupiedEntry<'a, K, V>),
//     /// A vacant entry.
//     Vacant(VacantEntry<'a, K, V>),
// }

// impl<'a, K: Chunk, V> Entry<'a, K, V> {
//     /// Ensures a value is in the entry by inserting the default if empty, and returns
//     /// a mutable reference to the value in the entry.
//     pub fn or_insert(self, default: V) -> &'a mut V {
//         match self {
//             Occupied(entry) => entry.into_mut(),
//             Vacant(entry) => entry.insert(default),
//         }
//     }

//     /// Ensures a value is in the entry by inserting the result of the default function if empty,
//     /// and returns a mutable reference to the value in the entry.
//     pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
//         match self {
//             Occupied(entry) => entry.into_mut(),
//             Vacant(entry) => entry.insert(default()),
//         }
//     }
// }

// /// A view into an occupied entry in a map.
// pub struct OccupiedEntry<'a, K: 'a, V: 'a> {
//     search_stack: SearchStack<'a, K, V>,
// }

// /// A view into a vacant entry in a map.
// pub struct VacantEntry<'a, K: 'a, V: 'a> {
//     search_stack: SearchStack<'a, K, V>,
// }

// /// A list of nodes encoding a path from the root of a map to a node.
// ///
// /// Invariants:
// /// * The last node is either `External` or `Nothing`.
// /// * Pointers at indexes less than `length` can be safely dereferenced.
// ///
// /// We can only use raw pointers, because of stacked borrows.
// /// Source:
// /// https://rust-unofficial.github.io/too-many-lists/fifth-stacked-borrows.html#managing-stacked-borrows
// struct SearchStack<'a, K: 'a, V: 'a> {
//     map: *mut Map<K, V>,
//     key: K,
//     items: Vec<*mut TrieNode<K, V>>,
//     phantom: PhantomData<&'a mut Map<K, V>>,
// }

// impl<'a, K, V> SearchStack<'a, K, V> {
//     fn new(map: *mut Map<K, V>, key: K) -> Self {
//         SearchStack {
//             map,
//             key,
//             items: Vec::new(),
//             phantom: PhantomData,
//         }
//     }

//     fn push(&mut self, node: *mut TrieNode<K, V>) {
//         self.items.push(node);
//     }

//     fn peek(&self) -> *mut TrieNode<K, V> {
//         self.items.last().copied().unwrap()
//     }

//     fn peek_ref(&self) -> &'a mut TrieNode<K, V> {
//         let item = self.items.last().copied().unwrap();
//         unsafe { &mut *item }
//     }

//     fn pop_ref(&mut self) -> &'a mut TrieNode<K, V> {
//         unsafe { &mut *self.items.pop().unwrap() }
//     }

//     fn is_empty(&self) -> bool {
//         self.items.is_empty()
//     }

//     fn get_ref(&self, idx: usize) -> &'a mut TrieNode<K, V> {
//         assert!(idx < self.items.len());
//         unsafe { &mut *self.items[idx] }
//     }
// }

// impl<K: Chunk, V> Map<K, V> {
//     /// Gets the given key's corresponding entry in the map for in-place manipulation.
//     #[inline]
//     pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
//         // The root is now a TrieNode, so the first item on the search stack IS the root itself.
//         let root_ptr = &mut self.root as *mut _;
//         let mut search_stack = SearchStack::new(self, key);
//         search_stack.push(root_ptr);

//         let search_successful: bool;
//         loop {
//             match unsafe {
//                 next_child(search_stack.peek(), &search_stack.key, search_stack.items.len() - 1)
//             } {
//                 (Some(child), _) => search_stack.push(child),
//                 (None, success) => {
//                     search_successful = success;
//                     break;
//                 }
//             }
//         }

//         if search_successful {
//             Occupied(OccupiedEntry { search_stack })
//         } else {
//             Vacant(VacantEntry { search_stack })
//         }
//     }
// }

// /// Get a mutable pointer to the next child of a node, given a key and an idx.
// ///
// /// `idx` is the chunk depth: 0 means we are inspecting the root node itself.
// /// For an Internal node, we descend into children[chunk(key, idx)].
// /// For External/Nothing, the search is complete.
// ///
// /// Returns (Some(child_ptr), false) to keep descending, or (None, found) when done.
// #[inline]
// unsafe fn next_child<K: Chunk, V>(
//     node: *mut TrieNode<K, V>,
//     key: &K,
//     idx: usize,
// ) -> (Option<*mut TrieNode<K, V>>, bool) {
//     match unsafe { &mut *node } {
//         &mut Internal(ref mut node_internal) => (
//             Some(&mut node_internal.children[key.chunk(idx) as usize] as *mut _),
//             false,
//         ),
//         External(stored_key, _) if stored_key == key => (None, true),
//         External(..) | Nothing => (None, false),
//     }
// }

// // NB: All these methods assume a correctly constructed occupied entry (matching the given key).
// impl<'a, K, V> OccupiedEntry<'a, K, V> {
//     /// Gets a reference to the value in the entry.
//     #[inline]
//     pub fn get(&self) -> &V {
//         match *self.search_stack.peek_ref() {
//             External(_, ref value) => value,
//             _ => unreachable!(),
//         }
//     }

//     /// Gets a mutable reference to the value in the entry.
//     #[inline]
//     pub fn get_mut(&mut self) -> &mut V {
//         match *self.search_stack.peek_ref() {
//             External(_, ref mut value) => value,
//             _ => unreachable!(),
//         }
//     }

//     /// Converts the OccupiedEntry into a mutable reference to the value in the entry,
//     /// with a lifetime bound to the map itself.
//     #[inline]
//     pub fn into_mut(self) -> &'a mut V {
//         match *self.search_stack.peek_ref() {
//             External(_, ref mut value) => value,
//             _ => unreachable!(),
//         }
//     }

//     /// Sets the value of the entry, and returns the entry's old value.
//     #[inline]
//     pub fn insert(&mut self, value: V) -> V {
//         match *self.search_stack.peek_ref() {
//             External(_, ref mut stored_value) => mem::replace(stored_value, value),
//             _ => unreachable!(),
//         }
//     }

//     /// Takes the value out of the entry, and returns it.
//     #[inline]
//     pub fn remove(self) -> V {
//         let mut search_stack = self.search_stack;

//         let leaf_node = mem::replace(search_stack.pop_ref(), Nothing);
//         let value = match leaf_node {
//             External(_, value) => value,
//             _ => unreachable!(),
//         };

//         // Unwind the stack, collapsing now-childless Internal ancestors.
//         // The bottom of the stack is the root TrieNode itself; we stop before
//         // touching it (an Internal root with count==1 should collapse to Nothing,
//         // handled the same way as any other node).
//         while !search_stack.is_empty() {
//             let ancestor = search_stack.pop_ref();
//             match *ancestor {
//                 Internal(ref mut internal) => {
//                     if internal.count != 1 {
//                         internal.count -= 1;
//                         break;
//                     }
//                 }
//                 _ => unreachable!(),
//             }
//             *ancestor = Nothing;
//         }

//         unsafe {
//             (*search_stack.map).length -= 1;
//         }

//         value
//     }
// }

// impl<'a, K: Chunk, V> VacantEntry<'a, K, V> {
//     /// Set the vacant entry to the given value.
//     pub fn insert(self, value: V) -> &'a mut V {
//         let search_stack = self.search_stack;
//         let old_length = search_stack.items.len();

//         unsafe {
//             (*search_stack.map).length += 1;
//         }

//         // The search stack always has at least one entry: the root TrieNode pointer.
//         // old_length == 1 means the root itself is the vacant/mismatched node.
//         if old_length == 1 {
//             unsafe {
//                 let mut dummy_count: usize = 0;
//                 let (value_ref, _) = insert(
//                     &mut dummy_count,
//                     search_stack.get_ref(0),
//                     search_stack.key,
//                     value,
//                     0,
//                 );
//                 value_ref
//             }
//         } else {
//             // The second-to-last item is the parent Internal node; the last item is
//             // the child slot where the new External node should be placed.
//             // Depth of the child slot = old_length - 1 (0-based chunk index).
//             match *search_stack.get_ref(old_length - 2) {
//                 Internal(ref mut parent) => {
//                     let parent = &mut **parent;
//                     let child_idx = search_stack.key.chunk(old_length - 1) as usize;
//                     let (value_ref, _) = insert(
//                         &mut parent.count,
//                         &mut parent.children[child_idx],
//                         search_stack.key,
//                         value,
//                         old_length,
//                     );
//                     value_ref
//                 }
//                 _ => unreachable!(),
//             }
//         }
//     }
// }

/// A forward iterator over a map.
pub struct Iter<'a, M: MapTrait> {
    stack: Vec<slice::Iter<'a, TrieNode<M>>>,
    remaining: usize,
}

#[cfg(feature = "extra")]
impl<'a, K, V, P> Clone for Iter<'a, K, V, P> where K: Chunk, P: PerfHint {
    #[cfg(target_pointer_width = "32")]
    fn clone(&self) -> Iter<'a, K, V> {
        Iter {
            stack: self.stack.clone(),
            ..*self
        }
    }

    #[cfg(target_pointer_width = "64")]
    fn clone(&self) -> Iter<'a, K, V> {
        Iter {
            stack: self.stack.clone(),
            ..*self
        }
    }
}

/// A forward iterator over the key-value pairs of a map, with the
/// values being mutable.
#[cfg(feature = "extra")]
pub struct IterMut<'a, K: 'a, V: 'a> {
    stack: Vec<slice::IterMut<'a, TrieNode<K, V>>>,
    remaining: usize,
}

/// A forward iterator over the keys of a map.
#[cfg(feature = "extra")]
pub struct Keys<'a, K: 'a, V: 'a>(Iter<'a, K, V>);

#[cfg(feature = "extra")]
impl<'a, K, V> Clone for Keys<'a, K, V> {
    fn clone(&self) -> Keys<'a, K, V> {
        Keys(self.0.clone())
    }
}

#[cfg(feature = "extra")]
impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|e| e.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

#[cfg(feature = "extra")]
impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {}

/// A forward iterator over the values of a map.
#[cfg(feature = "extra")]
pub struct Values<'a, K: 'a, V: 'a>(Iter<'a, K, V>);

#[cfg(feature = "extra")]
impl<'a, K, V> Clone for Values<'a, K, V> {
    fn clone(&self) -> Values<'a, K, V> {
        Values(self.0.clone())
    }
}

#[cfg(feature = "extra")]
impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|e| e.1)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

#[cfg(feature = "extra")]
impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {}

macro_rules! iterator_impl {
    ($name:ident,
     iter = $iter:ident,
     mutability = ($($mut_:tt)*)) => {
        impl<'a, M: MapTrait> $name<'a, M> {
            unsafe fn new() -> Self {
                $name {
                    remaining: 0,
                    stack: Vec::new(),
                }
            }
        }

        impl<'a, M: MapTrait> Iterator for $name<'a, M> {
            type Item = (&'a M::Key, &'a $($mut_)* M::Value);
            fn next(&mut self) -> Option<Self::Item> {
                while let Some(iter) = self.stack.last_mut() {
                    match iter.next() {
                        None => {
                            self.stack.pop();
                        }
                        Some(child) => {
                            match *child {
                                Internal(ref $($mut_)* node) => {
                                    self.stack.push(node.children.as_slice().$iter());
                                }
                                External(ref key, ref $($mut_)* value) => {
                                    self.remaining -= 1;
                                    return Some((key, value));
                                }
                                Nothing => {}
                            }
                        }
                    }
                }
                return None;
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.remaining, Some(self.remaining))
            }
        }

        impl<'a, M: MapTrait> ExactSizeIterator for $name<'a, M> {
            fn len(&self) -> usize { self.remaining }
        }
    }
}

iterator_impl! { Iter, iter = iter, mutability = () }
// iterator_impl! { IterMut, iter = iter_mut, mutability = (mut) }

// /// A bounded forward iterator over a map.
// pub struct Range<'a, K: 'a, V: 'a>(Iter<'a, K, V>);

// impl<'a, K, V> Clone for Range<'a, K, V> {
//     fn clone(&self) -> Range<'a, K, V> {
//         Range(self.0.clone())
//     }
// }

// impl<'a, K, V> Iterator for Range<'a, K, V> {
//     type Item = (&'a K, &'a V);
//     fn next(&mut self) -> Option<Self::Item> {
//         self.0.next()
//     }
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         (0, Some(self.0.remaining))
//     }
// }

// impl<'a, K: Chunk, V> IntoIterator for &'a Map<K, V> {
//     type Item = (&'a K, &'a V);
//     type IntoIter = Iter<'a, K, V>;
//     fn into_iter(self) -> Iter<'a, K, V> {
//         self.iter()
//     }
// }

// impl<'a, K: Chunk, V> IntoIterator for &'a mut Map<K, V> {
//     type Item = (&'a K, &'a mut V);
//     type IntoIter = IterMut<'a, K, V>;
//     fn into_iter(self) -> IterMut<'a, K, V> {
//         self.iter_mut()
//     }
// }

#[cfg(test)]
mod test {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::hint::black_box;
    use std::ops::Bound;

    #[cfg(feature = "extra")]
    use super::Entry::*;
    use crate::TrieMap;
    use crate::map_trait::{BasicTrieHint, Children, MapTrait, RootNode};
    use crate::node::TrieNode::{self, *};
    use crate::node::AnyNodeRef;
    use super::{InternalNode, Map};

    /// check_integrity now accepts a TrieNode instead of an InternalNode,
    /// because the root is a TrieNode enum.
    fn check_integrity<M: MapTrait<Children<TrieNode<M>> = [TrieNode<M>; 16]>>(node: &M::Root) {
        match node.into_any_ref() {
            AnyNodeRef::Branch { children, count } => check_integrity_internal(count, children),
            // A bare External or Nothing root is valid (0 or 1 elements).
            AnyNodeRef::External(..) | AnyNodeRef::Nothing => {}
        }
    }

    fn check_integrity_internal<M: MapTrait<Children<TrieNode<M>> = [TrieNode<M>; 16]>>(count: usize, children: &[TrieNode<M>]) {
        assert!(count != 0);

        let mut sum = 0;
        for x in children.iter() {
            match *x {
                Nothing => (),
                Internal(ref y) => {
                    check_integrity_internal(y.count, &y.children[..]);
                    sum += 1;
                }
                External(_, _) => sum += 1,
            }
        }
        assert_eq!(sum, count);
    }

    #[test]
    fn test_find_mut() {
        let mut m: TrieMap<_, _> = Map::new();
        assert!(m.insert(1, 12).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.insert(5, 14).is_none());
        let new = 100;
        match m.get_mut(&5) {
            None => panic!(),
            Some(x) => *x = new,
        }
        assert_eq!(m.get(&5), Some(&new));
    }

    #[test]
    fn test_find_mut_missing() {
        let mut m: TrieMap<_, _> = Map::new();
        assert!(m.get_mut(&0).is_none());
        assert!(m.insert(1, 12).is_none());
        assert!(m.get_mut(&0).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.get_mut(&0).is_none());
    }

    #[test]
    fn test_step() {
        let mut trie: TrieMap<_, _> = Map::new();
        let n = 300;

        for x in (1..n).step_by(2) {
            assert!(trie.insert(x, x + 1).is_none(), "{}", x);
            assert!(trie.contains_key(&x), "{}", x);
            check_integrity::<BasicTrieHint<i32, _>>(&trie.root);
        }

        for x in (0..n).step_by(2) {
            assert!(!trie.contains_key(&x));
            assert!(trie.insert(x, x + 1).is_none());
            check_integrity::<BasicTrieHint<i32, _>>(&trie.root);
        }

        for x in 0..n {
            assert!(trie.contains_key(&x));
            assert!(trie.insert(x, x + 1).is_some());
            check_integrity::<BasicTrieHint<i32, _>>(&trie.root);
        }

        for x in (1..n).step_by(2) {
            assert!(trie.remove(&x).is_some());
            assert!(!trie.contains_key(&x));
            check_integrity::<BasicTrieHint<i32, _>>(&trie.root);
        }

        for x in (0..n).step_by(2) {
            assert!(trie.contains_key(&x));
            assert!(trie.insert(x, x + 1).is_some());
            check_integrity::<BasicTrieHint<i32, _>>(&trie.root);
        }
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_each_reverse() {
        let mut m = Map::new();

        assert!(m.insert(3, 6).is_none());
        assert!(m.insert(0, 0).is_none());
        assert!(m.insert(4, 8).is_none());
        assert!(m.insert(2, 4).is_none());
        assert!(m.insert(1, 2).is_none());

        let mut n = 5;
        let mut vec: Vec<&i32> = vec![];
        m.each_reverse(|k, v| {
            n -= 1;
            assert_eq!(*k, n);
            vec.push(v);
            true
        });
        assert_eq!(vec, [&8, &6, &4, &2, &0]);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_each_reverse_break() {
        let mut m = Map::new();

        for x in (usize::MAX - 10000..usize::MAX).rev() {
            m.insert(x, x / 2);
        }

        let mut n = usize::MAX - 1;
        m.each_reverse(|k, v| {
            if n == usize::MAX - 5000 {
                false
            } else {
                assert!(n > usize::MAX - 5000);
                assert_eq!(*k, n);
                assert_eq!(*v, n / 2);
                n -= 1;
                true
            }
        });
    }

    #[test]
    fn test_insert() {
        let mut m: TrieMap<_, _> = Map::new();
        assert_eq!(m.insert(1, 2), None);
        assert_eq!(m.insert(1, 3), Some(2));
        assert_eq!(m.insert(1, 4), Some(3));
    }

    #[test]
    fn test_remove() {
        let mut m: TrieMap<_, _> = Map::new();
        m.insert(1, 2);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_from_iter() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: TrieMap<usize, i32> = xs.iter().cloned().collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_keys() {
        let vec = [(1, 'a'), (2, 'b'), (3, 'c')];
        let map: Map<usize, _> = vec.iter().cloned().collect();
        let keys: Vec<_> = map.keys().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&&1));
        assert!(keys.contains(&&2));
        assert!(keys.contains(&&3));
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_values() {
        let vec = [(1, 'a'), (2, 'b'), (3, 'c')];
        let map: Map<usize, _> = vec.iter().cloned().collect();
        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_iteration() {
        let empty_map: Map<usize, usize> = Map::new();
        assert_eq!(empty_map.iter().next(), None);

        let first = usize::MAX - 10000;
        let last = usize::MAX;

        let mut map = Map::new();
        for x in (first..last).rev() {
            map.insert(x, x / 2);
        }

        let mut i = 0;
        for (&k, &v) in map.iter() {
            assert_eq!(k, first + i);
            assert_eq!(v, k / 2);
            i += 1;
        }
        assert_eq!(i, last - first);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_mut_iter() {
        let mut empty_map: Map<usize, usize> = Map::new();
        assert!(empty_map.iter_mut().next().is_none());

        let first = usize::MAX - 10000;
        let last = usize::MAX;

        let mut map = Map::new();
        for x in (first..last).rev() {
            map.insert(x, x / 2);
        }

        let mut i = 0;
        for (&k, v) in map.iter_mut() {
            assert_eq!(k, first + i);
            *v -= k / 2;
            i += 1;
        }
        assert_eq!(i, last - first);

        assert!(map.iter().all(|(_, &v)| v == 0));
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_bound() {
        let empty_map: Map<usize, usize> = Map::new();
        assert_eq!(empty_map.lower_bound(Bound::Excluded(&0)).next(), None);
        assert_eq!(empty_map.upper_bound(Bound::Excluded(&0)).next(), None);

        let last = 999;
        let step = 3;
        let value = 42;

        let mut map: Map<usize, usize> = Map::new();
        for x in (0..last).step_by(step) {
            assert!(x % step == 0);
            map.insert(x, value);
        }

        for i in 0..last - step {
            let mut lb = map.lower_bound(Bound::Excluded(&i));
            let mut ub = map.upper_bound(Bound::Excluded(&i));
            let next_key = i - i % step + step;
            let next_pair = (&next_key, &value);
            if i % step == 0 {
                assert_eq!(lb.next(), Some((&i, &value)));
            } else {
                assert_eq!(lb.next(), Some(next_pair));
            }
            assert_eq!(ub.next(), Some(next_pair));
        }

        let mut lb = map.lower_bound(Bound::Excluded(&(last - step)));
        assert_eq!(lb.next(), Some((&(last - step), &value)));
        let mut ub = map.upper_bound(Bound::Excluded(&(last - step)));
        assert_eq!(ub.next(), None);

        for i in last - step + 1..last {
            let mut lb = map.lower_bound(Bound::Excluded(&i));
            assert_eq!(lb.next(), None);
            let mut ub = map.upper_bound(Bound::Excluded(&i));
            assert_eq!(ub.next(), None);
        }
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_mut_bound() {
        let empty_map: Map<usize, usize> = Map::new();
        assert_eq!(empty_map.lower_bound(Bound::Excluded(&0)).next(), None);
        assert_eq!(empty_map.upper_bound(Bound::Excluded(&0)).next(), None);

        let mut m_lower = Map::new();
        let mut m_upper = Map::new();
        for i in 0..100 {
            m_lower.insert(2 * i, 4 * i);
            m_upper.insert(2 * i, 4 * i);
        }

        for i in 0..199 {
            let mut lb_it = m_lower.lower_bound_mut(Bound::Excluded(&i));
            let (&k, v) = lb_it.next().unwrap();
            let lb = i + i % 2;
            assert_eq!(lb, k);
            *v -= k;
        }

        for i in 0..198 {
            let mut ub_it = m_upper.upper_bound_mut(Bound::Excluded(&i));
            let (&k, v) = ub_it.next().unwrap();
            let ub = i + 2 - i % 2;
            assert_eq!(ub, k);
            *v -= k;
        }

        assert!(m_lower.lower_bound_mut(Bound::Excluded(&199)).next().is_none());
        assert!(m_upper.upper_bound_mut(Bound::Excluded(&198)).next().is_none());

        assert!(m_lower.iter().all(|(_, &x)| x == 0));
        assert!(m_upper.iter().all(|(_, &x)| x == 0));
    }

    #[test]
    fn test_clone() {
        let mut a: TrieMap<_, _> = Map::new();

        a.insert(1, 'a');
        a.insert(2, 'b');
        a.insert(3, 'c');

        assert!(a.clone() == a);
    }

    #[test]
    fn test_eq() {
        let mut a: TrieMap<_, _> = Map::new();
        let mut b = Map::new();

        assert!(a == b);
        assert!(a.insert(0, 5).is_none());
        assert!(a != b);
        assert!(b.insert(0, 4).is_none());
        assert!(a != b);
        assert!(a.insert(5, 19).is_none());
        assert!(a != b);
        assert!(b.insert(0, 5).is_some());
        assert!(a != b);
        assert!(b.insert(5, 19).is_none());
        assert!(a == b);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_lt() {
        let mut a = Map::new();
        let mut b = Map::new();

        assert!((a >= b) && (b >= a));
        assert!(b.insert(2, 5).is_none());
        assert!(a < b);
        assert!(a.insert(2, 7).is_none());
        assert!((a >= b) && b < a);
        assert!(b.insert(1, 0).is_none());
        assert!(b < a);
        assert!(a.insert(0, 6).is_none());
        assert!(a < b);
        assert!(a.insert(6, 2).is_none());
        assert!(a < b && (b >= a));
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_ord() {
        let mut a = Map::new();
        let mut b = Map::new();

        assert!(a == b);
        assert!(a.insert(1, 1).is_none());
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2, 2).is_none());
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_hash() {
        fn hash<T: Hash>(t: &T) -> u64 {
            let mut s = DefaultHasher::new();
            t.hash(&mut s);
            s.finish()
        }

        let mut x = Map::new();
        let mut y = Map::new();

        assert!(hash(&x) == hash(&y));
        x.insert(1, 'a');
        x.insert(2, 'b');
        x.insert(3, 'c');

        y.insert(3, 'c');
        y.insert(2, 'b');
        y.insert(1, 'a');

        assert!(hash(&x) == hash(&y));
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_debug() {
        let mut map = Map::new();
        let empty: Map<usize, char> = Map::new();

        map.insert(1, 'a');
        map.insert(2, 'b');

        assert_eq!(format!("{:?}", map), "{1: 'a', 2: 'b'}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_index() {
        let mut map = Map::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[&2], 1);
    }

    #[cfg(feature = "extra")]
    #[test]
    #[should_panic]
    fn test_index_nonexistent() {
        let mut map = Map::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        black_box(map[&4]);
    }

    const SQUARES_UPPER_LIM: usize = 128;

    #[cfg(feature = "extra")]
    fn squares_map() -> Map<usize, usize> {
        let mut map = Map::new();
        for i in 0..SQUARES_UPPER_LIM {
            map.insert(i, i * i);
        }
        map
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_entry_get() {
        let mut map = squares_map();

        for i in 0..SQUARES_UPPER_LIM {
            match map.entry(i) {
                Occupied(slot) => assert_eq!(slot.get(), &(i * i)),
                Vacant(_) => panic!("Key not found."),
            }
        }
        check_integrity(&map.root);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_entry_get_mut() {
        let mut map = squares_map();

        for i in 0..SQUARES_UPPER_LIM {
            match map.entry(i) {
                Occupied(mut e) => {
                    *e.get_mut() = i * i * i;
                }
                Vacant(_) => panic!("Key not found."),
            }
            assert_eq!(map.get(&i).unwrap(), &(i * i * i));
        }

        check_integrity(&map.root);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_entry_into_mut() {
        let mut map = Map::new();
        map.insert(3, 6);

        let value_ref = match map.entry(3) {
            Occupied(e) => e.into_mut(),
            Vacant(_) => panic!("Entry not found."),
        };

        assert_eq!(*value_ref, 6);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_entry_take() {
        let mut map = squares_map();
        assert_eq!(map.len(), SQUARES_UPPER_LIM);

        for i in (1..SQUARES_UPPER_LIM).step_by(2) {
            match map.entry(i) {
                Occupied(e) => assert_eq!(e.remove(), i * i),
                Vacant(_) => panic!("Key not found."),
            }
        }

        check_integrity(&map.root);

        for i in (0..SQUARES_UPPER_LIM).step_by(2) {
            assert_eq!(map.get(&i).unwrap(), &(i * i));
        }

        assert_eq!(map.len(), SQUARES_UPPER_LIM / 2);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_occupied_entry_set() {
        let mut map = squares_map();

        for i in 0..SQUARES_UPPER_LIM {
            match map.entry(i) {
                Occupied(mut e) => assert_eq!(e.insert(i * i * i), i * i),
                Vacant(_) => panic!("Key not found."),
            }
            assert_eq!(map.get(&i).unwrap(), &(i * i * i));
        }
        check_integrity(&map.root);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_vacant_entry_set() {
        let mut map = Map::new();

        for i in 0..SQUARES_UPPER_LIM {
            match map.entry(i) {
                Vacant(e) => {
                    let inserted_val = e.insert(i * i);
                    assert_eq!(*inserted_val, i * i);
                    *inserted_val = i * i * i;
                }
                _ => panic!("Non-existent key found."),
            }
            assert_eq!(map.get(&i).unwrap(), &(i * i * i));
        }

        check_integrity(&map.root);
        assert_eq!(map.len(), SQUARES_UPPER_LIM);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_single_key() {
        let mut map = Map::new();
        map.insert(1, 2);

        if let Occupied(e) = map.entry(1) {
            e.remove();
        }
    }
}
