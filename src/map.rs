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
use crate::inner::{Entry, Inner, Occupied, Vacant};
use crate::map_trait::{Children, MapTrait, MaybeValue};

use crate::node::InternalNode;
use crate::root_node::AnyNodeMut;
use crate::root_node::AnyNodeRef;
use crate::root_node::EitherNode;
use crate::root_node::{AnyNode, EitherNodeRef};
// pub use self::Entry::*;
use crate::node::TrieNode::{self, *};
use crate::root_node::RootNode;

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::btree_map::OccupiedEntry;
use std::fmt::{self, Binary, Debug};
use std::hash::{Hash, Hasher};
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ops::{self, Bound};
use std::slice;
use std::option;

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
pub struct Map<M>
where
    M: MapTrait,
{
    pub(crate) root: M::Root,
    pub(crate) length: usize,
}

impl<M: MapTrait<Key: Clone, Value: Clone>> Clone
    for Map<M>
{
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
impl<K, V, P> Default for Map<K, V, P>
where
    K: Chunk,
    P: PerfHint<K, V>,
{
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
            EitherNodeRef::Internal(&InternalNode {
                count: _,
                ref value,
                ref children,
                ..
            }) => {
                (value.as_ref().into_iter(), children.as_slice().iter())
            }
            EitherNodeRef::Trie(trie_node) => {
                (None.into_iter(), slice::from_ref(trie_node).iter())
            }
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
        iter.stack
            .push(std::slice::from_mut(&mut self.root).iter_mut());
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
        Q: Chunk + Eq + Hash + Ord,
    {
        // Root is now a TrieNode: start traversal from it directly at idx 0.
        let mut node = self.root.into_any_ref();
        let mut idx: Q::KeySize = 0u8.try_into().ok().unwrap();

        loop {
            match node {
                AnyNodeRef::Branch { children, .. } => {
                    node = children
                        [key.chunk(idx.try_into().ok()?.try_into().ok()?, M::SHIFT) as usize]
                        .into_any_ref();
                    idx += M::SHIFT.try_into().ok().unwrap();
                }
                AnyNodeRef::External(external) => {
                    return external.get(key);
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
        Q: Chunk + Eq + Hash + Ord,
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
        find_mut(self.root.as_any_mut(), key, 0u8.try_into().ok().unwrap())
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
        let mut root_count: M::Count = Default::default();
        let (_, old_val) = match self.root.as_either() {
            EitherNode::Trie(node) => insert(
                &mut root_count,
                node,
                key,
                value,
                0u8.try_into().ok().unwrap(),
            ),
            EitherNode::Internal(internal) => {
                if key.bits() == 0.into() {
                    return internal.value.replace(value);
                }
                insert(
                    &mut internal.count,
                    &mut internal.children[key.chunk(0u8.try_into().ok().unwrap(), M::SHIFT) as usize],
                    key,
                    value,
                    M::SHIFT.try_into().ok().unwrap(),
                )
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
        let mut root_count: M::Count = Default::default();
        let ret = match self.root.as_either() {
            EitherNode::Trie(node) => remove(&mut root_count, node, key, 0u8.into()),
            EitherNode::Internal(internal) => remove(
                &mut internal.count,
                &mut internal.children[key.chunk(0u8.into(), M::SHIFT) as usize],
                key,
                M::SHIFT.into(),
            ),
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
    Q: Chunk + Eq + Ord + Hash + 'a,
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
    Q: Chunk + Eq + Ord + Hash + 'a,
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
fn find_mut<'a, M, Q>(node: AnyNodeMut<'a, M>, key: &Q, idx: Q::KeySize) -> Option<&'a mut M::Value>
where
    M: MapTrait,
    M::Key: Borrow<Q> + Chunk,
    Q: Chunk,
{
    match node {
        AnyNodeMut::External(maybe_inner) => maybe_inner.get_mut(key),
        AnyNodeMut::Branch { children, .. } => find_mut(
            children[key.chunk(idx, M::SHIFT) as usize].as_any_mut(),
            key,
            idx + M::SHIFT.into(),
        ),
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
    count: &mut M::Count,
    start_node: &'a mut TrieNode<M>,
    key: M::Key,
    value: M::Value,
    idx: <<M as MapTrait>::Key as Chunk>::KeySize,
) -> (&'a mut M::Value, Option<M::Value>) {
    // We branch twice to avoid having to do the `replace` when we don't need to;
    // this is much faster, especially for keys that have long shared prefixes.

    let hack;

    let key = match *start_node {
        Nothing => {
            *count += 1;
            *start_node = External(M::MaybeInner::new(key, value));
            match *start_node {
                External(ref mut external) => return (external.values_mut().next().unwrap(), None),
                _ => unreachable!(),
            }
        }
        Internal(ref mut x) => {
            let x = &mut **x;
            // TODO: bit-vectors and non-8-multiply lengths
            if key.bits() == idx {
                let old = x.value.replace(value);
                return (x.value.as_mut().unwrap(), old);
            }
            assert!(idx < key.bits(), "idx = {} < key.bits() = {}", idx, key.bits());

            return insert(
                &mut x.count,
                &mut x.children[key.chunk(idx, M::SHIFT) as usize],
                key,
                value,
                idx + M::SHIFT.into(),
            );
        }
        External(ref mut maybe_inner) => {
            match maybe_inner.entry(key) {
                Entry::Occupied(occupied) => {
                    hack = true;
                    occupied.into_key()
                }
                Entry::Vacant(vacant) if vacant.should_split() => {
                    hack = false;
                    vacant.into_key()
                }
                Entry::Vacant(vacant) => {
                    hack = true;
                    vacant.into_key()
                }
            }
        }
    };

    if !hack {
        match mem::replace(start_node, Internal(Box::new(InternalNode::new()))) {
            External(inner) => {
                for (k, v) in inner {
                    insert(count, start_node, k, v, idx);
                }
                return insert(count, start_node, key, value, idx);
            }
            _ => unreachable!(),
        }
    }

    if let External(ref mut maybe_inner) = *start_node {
        match maybe_inner.entry(key) {
            Entry::Occupied(mut occupied) => {
                let prev = occupied.insert(value);
                return (occupied.into_mut(), Some(prev));
            }
            Entry::Vacant(vacant) => {
                *count += 1;
                return (vacant.insert(value), None);
            }
        }
    }

    unreachable!();
}

// TODO: make the function non-recursive
fn remove<M: MapTrait, Q: Chunk>(
    count: &mut M::Count,
    child: &mut TrieNode<M>,
    key: &Q,
    idx: Q::KeySize,
) -> Option<M::Value>
where
    M::Key: Borrow<Q> + Chunk,
{
    // TODO optimize
    let this;
    let mut ret;
    'outer: {
        match *child {
            External(ref mut maybe_inner) => {
                let should_remove = maybe_inner.should_remove::<M>();
                match maybe_inner.get_mut(key) {
                    Some(_) if should_remove => {
                        this = true;
                        ret = None;
                        break 'outer;
                    }
                    Some(_) => {
                        this = false;
                    }
                    None => {
                        return None;
                    }
                }
                ret = maybe_inner.remove(key);
                *count -= 1;
            }
            Internal(ref mut x) => {
                let x = &mut **x;
                if key.bits() == idx {
                    return x.value.take();
                }
                ret = remove(
                    &mut x.count,
                    &mut x.children
                        [key.chunk(idx.try_into().ok()?.try_into().ok()?, M::SHIFT) as usize],
                    key,
                    idx + M::SHIFT.into(),
                );
                this = if let Ok(val) = x.count.try_into() { val == 0 } else { x.is_empty() };
            }
            Nothing => {
                ret = None;
                this = false;
            }
        };
    }

    if this {
        *count -= 1;
        match mem::replace(child, Nothing) {
            External(maybe_inner) => {
                ret = maybe_inner.into_iter().next().map(|(_k, v)| v);
            }
            _ => {}
        };
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

#[cfg(feature = "extra")]
impl<'a, K, V, P> Clone for Iter<'a, K, V, P>
where
    K: Chunk,
    P: PerfHint,
{
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
        /// A forward iterator over a map.
        pub struct $name<'a, M: MapTrait> {
            maybe_inner: Option<<M::MaybeInner as Inner<M::Key, M::Value>>::$name<'a>>,
            stack: Vec<(option::IntoIter<&'a $($mut_)* M::Value>, slice::$name<'a, TrieNode<M>>)>,
            remaining: usize,
        }

        impl<'a, M: MapTrait> $name<'a, M> {
            unsafe fn new() -> Self {
                $name {
                    maybe_inner: None,
                    remaining: 0,
                    stack: Vec::new(),
                }
            }
        }

        impl<'a, M: MapTrait> Iterator for $name<'a, M> {
            type Item = (&'a M::Key, &'a $($mut_)* M::Value);
            fn next(&mut self) -> Option<Self::Item> {
                if let &mut Some(ref mut iter) = &mut self.maybe_inner {
                    if let Some((key, value)) = iter.next() {
                        return Some((key, value));
                    }
                }
                while let Some(&mut (ref node, ref mut iter)) = self.stack.last_mut() {
                    match iter.next() {
                        None => {
                            self.stack.pop();
                        }
                        Some(child) => {
                            match *child {
                                Internal(ref $($mut_)* node) => {
                                    self.stack.push((node.value.as_ref().into_iter(), node.children.as_slice().$iter()));
                                }
                                External(ref $($mut_)* maybe_inner) => {
                                    self.remaining -= 1;
                                    self.maybe_inner = Some(maybe_inner.$iter());
                                    return self.next();
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
