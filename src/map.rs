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
use crate::chunk::{MAX_DEPTH, SIZE};

pub use self::Entry::*;
use self::TrieNode::*;

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ops;
use std::ptr;
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
#[derive(Clone)]
pub struct Map<K, V> {
    root: InternalNode<K, V>,
    length: usize,
}

// An internal node holds SIZE child nodes, which may themselves contain more internal nodes.
//
// Throughout this implementation, "idx" is used to refer to a section of key that is used
// to access a node. The layer of the tree directly below the root corresponds to idx 0.
#[derive(Clone)]
struct InternalNode<K, V> {
    // The number of direct children which are external (i.e. that store a value).
    count: usize,
    children: [TrieNode<K, V>; SIZE],
}

// Each child of an InternalNode may be internal, in which case nesting continues,
// external (containing a value), or empty
#[derive(Clone)]
enum TrieNode<K, V> {
    Internal(Box<InternalNode<K, V>>),
    External(K, V),
    Nothing,
}

impl<K: Chunk, V: PartialEq> PartialEq for Map<K, V> {
    fn eq(&self, other: &Map<K, V>) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<K: Chunk, V: Eq> Eq for Map<K, V> {}

impl<K: PartialOrd + Chunk, V: PartialOrd> PartialOrd for Map<K, V> {
    #[inline]
    fn partial_cmp(&self, other: &Map<K, V>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Ord + Chunk, V: Ord> Ord for Map<K, V> {
    #[inline]
    fn cmp(&self, other: &Map<K, V>) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K: Debug + Chunk, V: Debug> Debug for Map<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V> Default for Map<K, V> {
    #[inline]
    fn default() -> Map<K, V> {
        Map::new()
    }
}

impl<K, V> Map<K, V> {
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
            root: InternalNode::new(),
            length: 0,
        }
    }
}

impl<K: Chunk, V> Map<K, V> {
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
    #[inline]
    pub fn each_reverse<'a, F>(&'a self, mut f: F) -> bool
    where
        F: FnMut(&K, &'a V) -> bool,
    {
        self.root.each_reverse(&mut f)
    }

    /// Gets an iterator visiting all keys in ascending order by the keys.
    /// The iterator's element type is `usize`.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys(self.iter())
    }

    /// Gets an iterator visiting all values in ascending order by the keys.
    /// The iterator's element type is `&'r T`.
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
    pub fn iter(&self) -> Iter<'_, K, V> {
        let mut iter = unsafe { Iter::new() };
        iter.stack[0] = self.root.children.iter();
        iter.length = 1;
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
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let mut iter = unsafe { IterMut::new() };
        iter.stack[0] = self.root.children.iter_mut();
        iter.length = 1;
        iter.remaining = self.length;

        iter
    }

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
        self.root = InternalNode::new();
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
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Chunk,
    {
        let mut node = &self.root;
        let mut idx = 0;
        loop {
            match node.children[key.chunk(idx)] {
                Internal(ref x) => node = &**x,
                External(ref stored, ref value) => {
                    if stored.borrow() == key {
                        return Some(value);
                    } else {
                        return None;
                    }
                }
                Nothing => return None,
            }
            idx += 1;
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
        K: Borrow<Q>,
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
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Chunk,
    {
        find_mut(&mut self.root.children[key.chunk(0)], key, 1)
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
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let (_, old_val) = insert(
            &mut self.root.count,
            &mut self.root.children[key.chunk(0)],
            key,
            value,
            1,
        );
        if old_val.is_none() {
            self.length += 1
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
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Chunk,
    {
        let ret = remove(
            &mut self.root.count,
            &mut self.root.children[key.chunk(0)],
            key,
            1,
        );
        if ret.is_some() {
            self.length -= 1
        }
        ret
    }
}

macro_rules! bound {
    (
        $iterator_name:ident,
        // the current treemap
        self = $this:expr,
        // the key to look for
        key = $key:expr,
        // are we looking at the upper bound?
        is_upper = $upper:expr,

        // method name for iterating.
        iter = $iter:ident,

        // this is just an optional mut, but there's no 0-or-1 repeats yet.
        mutability = ($($mut_:tt)*),
        const = ($($const_:tt)*)
    ) => {
        {
            // We need an unsafe pointer here because we are borrowing
            // mutable references to the internals of each of these
            // mutable nodes, while still using the outer node.
            //
            // However, we're allowed to flaunt rustc like this because we
            // never actually modify the "shape" of the nodes. The only
            // place that mutation is can actually occur is of the actual
            // values of the map (as the return value of the
            // iterator), i.e. we can never cause a deallocation of any
            // InternalNodes so the raw pointer is always valid.
            let this = $this;
            let mut node = & $($mut_)* this.root as *$($mut_)* $($const_)* InternalNode<K, V>;

            let key = $key;

            let mut it = unsafe {$iterator_name::new()};
            // everything else is zero'd, as we want.
            it.remaining = this.length;

            // this addr is necessary for the `Internal` pattern.
            loop {
                let children = unsafe { & $($mut_)* (*node).children };
                // it.length is the current depth in the iterator and the
                // current depth through the `usize` key we've traversed.
                let child_id = key.chunk(it.length);
                let (slice_idx, ret) = match & $($mut_)* children[child_id] {
                    & $($mut_)* Internal(ref $($mut_)* n) => {
                        node = (& $($mut_)* **n) as *$($mut_)* $($const_)* _;
                        (child_id + 1, false)
                    }
                    & $($mut_)* External(ref stored, _) => {
                        (if stored < key || ($upper && stored == key) {
                            child_id + 1
                        } else {
                            child_id
                        }, true)
                    }
                    & $($mut_)* Nothing => {
                        (child_id + 1, true)
                    }
                };
                // push to the stack.
                it.stack[it.length] = children[slice_idx..].$iter();
                it.length += 1;
                if ret { break }
            }

            it
        }
    }
}

impl<K: Chunk + PartialOrd, V> Map<K, V> {
    // If `upper` is true then returns upper_bound else returns lower_bound.
    #[inline]
    fn bound(&self, key: &K, upper: bool) -> Range<'_, K, V> {
        Range(bound!(Iter, self = self,
               key = key, is_upper = upper,
               iter = iter,
               mutability = (), const = (const)))
    }

    /// Gets an iterator pointing to the first key-value pair whose key is not less than `key`.
    /// If all keys in the map are less than `key` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let map: trie::Map<usize, &str> = [(2, "a"), (4, "b"), (6, "c")].iter().cloned().collect();
    ///
    /// assert_eq!(map.lower_bound(&4).next(), Some((&4, &"b")));
    /// assert_eq!(map.lower_bound(&5).next(), Some((&6, &"c")));
    /// assert_eq!(map.lower_bound(&10).next(), None);
    /// ```
    pub fn lower_bound(&self, key: &K) -> Range<'_, K, V> {
        self.bound(key, false)
    }

    /// Gets an iterator pointing to the first key-value pair whose key is greater than `key`.
    /// If all keys in the map are not greater than `key` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let map: trie::Map<usize, &str> = [(2, "a"), (4, "b"), (6, "c")].iter().cloned().collect();
    ///
    /// assert_eq!(map.upper_bound(&4).next(), Some((&6, &"c")));
    /// assert_eq!(map.upper_bound(&5).next(), Some((&6, &"c")));
    /// assert_eq!(map.upper_bound(&10).next(), None);
    /// ```
    pub fn upper_bound(&self, key: &K) -> Range<'_, K, V> {
        self.bound(key, true)
    }
    // If `upper` is true then returns upper_bound else returns lower_bound.
    #[inline]
    fn bound_mut(&mut self, key: &K, upper: bool) -> RangeMut<'_, K, V> {
        RangeMut(bound!(IterMut, self = self,
               key = key, is_upper = upper,
               iter = iter_mut,
               mutability = (mut), const = ()))
    }

    /// Gets an iterator pointing to the first key-value pair whose key is not less than `key`.
    /// If all keys in the map are less than `key` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut map: trie::Map<usize, &str> = [(2, "a"), (4, "b"), (6, "c")].iter().cloned().collect();
    ///
    /// assert_eq!(map.lower_bound_mut(&4).next(), Some((&4, &mut "b")));
    /// assert_eq!(map.lower_bound_mut(&5).next(), Some((&6, &mut "c")));
    /// assert_eq!(map.lower_bound_mut(&10).next(), None);
    ///
    /// for (_key, value) in map.lower_bound_mut(&4) {
    ///     *value = "changed";
    /// }
    ///
    /// assert_eq!(map.get(&2), Some(&"a"));
    /// assert_eq!(map.get(&4), Some(&"changed"));
    /// assert_eq!(map.get(&6), Some(&"changed"));
    /// ```
    pub fn lower_bound_mut(&mut self, key: &K) -> RangeMut<'_, K, V> {
        self.bound_mut(key, false)
    }

    /// Gets an iterator pointing to the first key-value pair whose key is greater than `key`.
    /// If all keys in the map are not greater than `key` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut map: trie::Map<usize, &str> = [(2, "a"), (4, "b"), (6, "c")].iter().cloned().collect();
    ///
    /// assert_eq!(map.upper_bound_mut(&4).next(), Some((&6, &mut "c")));
    /// assert_eq!(map.upper_bound_mut(&5).next(), Some((&6, &mut "c")));
    /// assert_eq!(map.upper_bound_mut(&10).next(), None);
    ///
    /// for (_key, value) in map.upper_bound_mut(&4) {
    ///     *value = "changed";
    /// }
    ///
    /// assert_eq!(map.get(&2), Some(&"a"));
    /// assert_eq!(map.get(&4), Some(&"b"));
    /// assert_eq!(map.get(&6), Some(&"changed"));
    /// ```
    pub fn upper_bound_mut(&mut self, key: &K) -> RangeMut<'_, K, V> {
        self.bound_mut(key, true)
    }
}

impl<K: Chunk, V> iter::FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Map<K, V> {
        let mut map = Map::new();
        map.extend(iter);
        map
    }
}

impl<K: Chunk, V> Extend<(K, V)> for Map<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Chunk + Hash, V: Hash> Hash for Map<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}

impl<'a, Q, K, V> ops::Index<&'a Q> for Map<K, V>
where
    K: Borrow<Q> + Chunk,
    Q: Chunk + 'a,
{
    type Output = V;
    #[inline]
    fn index(&self, i: &'a Q) -> &V {
        self.get(i).expect("key not present")
    }
}

impl<'a, Q, K, V> ops::IndexMut<&'a Q> for Map<K, V>
where
    K: Borrow<Q> + Chunk,
    Q: Chunk + 'a,
{
    #[inline]
    fn index_mut(&mut self, i: &'a Q) -> &mut V {
        self.get_mut(i).expect("key not present")
    }
}

impl<K, V> InternalNode<K, V> {
    #[inline]
    fn new() -> Self {
        InternalNode {
            count: 0,
            children: [const { Nothing }; SIZE],
        }
    }
}

impl<K, V> InternalNode<K, V> {
    fn each_reverse<'a, F>(&'a self, f: &mut F) -> bool
    where
        F: FnMut(&'a K, &'a V) -> bool,
    {
        for elt in self.children.iter().rev() {
            match *elt {
                Internal(ref x) => {
                    if !x.each_reverse(f) {
                        return false;
                    }
                }
                External(ref k, ref v) => {
                    if !f(k, v) {
                        return false;
                    }
                }
                Nothing => (),
            }
        }
        true
    }
}

fn find_mut<'a, K, Q: Chunk, V>(
    child: &'a mut TrieNode<K, V>,
    key: &Q,
    idx: usize,
) -> Option<&'a mut V>
where
    K: Borrow<Q> + Chunk,
{
    match *child {
        External(ref stored, ref mut value) if stored.borrow() == key => Some(value),
        External(..) => None,
        Internal(ref mut x) => find_mut(&mut x.children[key.chunk(idx)], key, idx + 1),
        Nothing => None,
    }
}

/// Inserts a new node for the given key and value, at or below `start_node`.
///
/// The index (`idx`) is the index of the next node, such that the start node
/// was accessed via parent.children[chunk(key, idx - 1)].
///
/// The count is the external node counter for the start node's parent,
/// which will be incremented only if `start_node` is transformed into a *new* external node.
///
/// Returns a mutable reference to the inserted value and an optional previous value.
fn insert<'a, K: Chunk, V>(
    count: &mut usize,
    start_node: &'a mut TrieNode<K, V>,
    key: K,
    value: V,
    idx: usize,
) -> (&'a mut V, Option<V>) {
    // We branch twice to avoid having to do the `replace` when we
    // don't need to; this is much faster, especially for keys that
    // have long shared prefixes.

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
                &mut x.children[key.chunk(idx)],
                key,
                value,
                idx + 1,
            );
        }
        External(ref stored_key, _) if stored_key == &key => {
            hack = true;
        }
        _ => {}
    }

    if !hack {
        // Conflict, an external node with differing keys.
        // We replace the old node by an internal one, then re-insert the two values beneath it.
        match mem::replace(start_node, Internal(Box::new(InternalNode::new()))) {
            External(stored_key, stored_value) => {
                match *start_node {
                    Internal(ref mut new_node) => {
                        let new_node = &mut **new_node;
                        // Re-insert the old value.
                        // TODO: remove recursive call.
                        insert(
                            &mut new_node.count,
                            &mut new_node.children[stored_key.chunk(idx)],
                            stored_key,
                            stored_value,
                            idx + 1,
                        );

                        // Insert the new value, and return a reference to it directly.
                        return insert(
                            &mut new_node.count,
                            &mut new_node.children[key.chunk(idx)],
                            key,
                            value,
                            idx + 1,
                        );
                    }
                    // Value that was just copied disappeared.
                    _ => unreachable!(),
                }
            }
            // Logic error in previous match.
            _ => unreachable!(),
        }
    }

    if let External(_, ref mut stored_value) = *start_node {
        // Swap in the new value and return the old.
        let old_value = mem::replace(stored_value, value);
        return (stored_value, Some(old_value));
    }

    unreachable!();
}

fn remove<K, Q: Chunk, V>(
    count: &mut usize,
    child: &mut TrieNode<K, V>,
    key: &Q,
    idx: usize,
) -> Option<V>
where
    K: Borrow<Q> + Chunk,
{
    let (ret, this) = match *child {
        External(ref stored, _) if stored.borrow() == key => match mem::replace(child, Nothing) {
            External(_, value) => (Some(value), true),
            _ => unreachable!(),
        },
        External(..) => (None, false),
        Internal(ref mut x) => {
            let x = &mut **x;
            let ret = remove(&mut x.count, &mut x.children[key.chunk(idx)], key, idx + 1);
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

/// A view into a single entry in a map, which may be vacant or occupied.
pub enum Entry<'a, K: 'a, V: 'a> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K: Chunk, V> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default()),
        }
    }
}

/// A view into an occupied entry in a map.
pub struct OccupiedEntry<'a, K: 'a, V: 'a> {
    search_stack: SearchStack<'a, K, V>,
}

/// A view into a vacant entry in a map.
pub struct VacantEntry<'a, K: 'a, V: 'a> {
    search_stack: SearchStack<'a, K, V>,
}

/// A list of nodes encoding a path from the root of a map to a node.
///
/// Invariants:
/// * The last node is either `External` or `Nothing`.
/// * Pointers at indexes less than `length` can be safely dereferenced.
///
/// we can only use raw pointers, because of stacked borrows.
/// Source:
/// https://rust-unofficial.github.io/too-many-lists/fifth-stacked-borrows.html#managing-stacked-borrows
struct SearchStack<'a, K: 'a, V: 'a> {
    map: *mut Map<K, V>,
    length: usize,
    key: K,
    items: [*mut TrieNode<K, V>; MAX_DEPTH],
    phantom: PhantomData<&'a mut Map<K, V>>,
}

impl<'a, K, V> SearchStack<'a, K, V> {
    /// Creates a new search-stack with empty entries.
    fn new(map: *mut Map<K, V>, key: K) -> Self {
        SearchStack {
            map,
            length: 0,
            key,
            items: [ptr::null_mut(); MAX_DEPTH],
            phantom: PhantomData,
        }
    }

    fn push(&mut self, node: *mut TrieNode<K, V>) {
        self.length += 1;
        self.items[self.length - 1] = node;
    }

    fn peek(&self) -> *mut TrieNode<K, V> {
        self.items[self.length - 1]
    }

    fn peek_ref(&self) -> &'a mut TrieNode<K, V> {
        let item = self.items[self.length - 1];
        unsafe { &mut *item }
    }

    fn pop_ref(&mut self) -> &'a mut TrieNode<K, V> {
        self.length -= 1;
        unsafe { &mut *self.items[self.length] }
    }

    fn is_empty(&self) -> bool {
        self.length == 0
    }

    fn get_ref(&self, idx: usize) -> &'a mut TrieNode<K, V> {
        assert!(idx < self.length);
        unsafe { &mut *self.items[idx] }
    }
}

// Implementation of SearchStack creation logic.
// Once a SearchStack has been created the Entry methods are relatively straight-forward.
impl<K: Chunk, V> Map<K, V> {
    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        // Unconditionally add the corresponding node from the first layer.
        let first_node = &mut self.root.children[key.chunk(0)] as *mut _;
        // Create an empty search stack.
        let mut search_stack = SearchStack::new(self, key);
        search_stack.push(first_node);

        // While no appropriate slot is found, keep descending down the Trie,
        // adding nodes to the search stack.
        let search_successful: bool;
        loop {
            match unsafe { next_child(search_stack.peek(), &search_stack.key, search_stack.length) }
            {
                (Some(child), _) => search_stack.push(child),
                (None, success) => {
                    search_successful = success;
                    break;
                }
            }
        }

        if search_successful {
            Occupied(OccupiedEntry { search_stack })
        } else {
            Vacant(VacantEntry { search_stack })
        }
    }
}

/// Get a mutable pointer to the next child of a node, given a key and an idx.
///
/// The idx is the index of the next child, such that `node` was accessed via
/// parent.children[chunk(key, idx - 1)].
///
/// Returns a tuple with an optional mutable pointer to the next child, and
/// a boolean flag to indicate whether the external key node was found.
///
/// This function is safe only if `node` points to a valid `TrieNode`.
#[inline]
unsafe fn next_child<K: Chunk, V>(
    node: *mut TrieNode<K, V>,
    key: &K,
    idx: usize,
) -> (Option<*mut TrieNode<K, V>>, bool) {
    match unsafe { &mut *node } {
        // If the node is internal, tell the caller to descend further.
        &mut Internal(ref mut node_internal) => (
            Some(&mut node_internal.children[key.chunk(idx)] as *mut _),
            false,
        ),
        // If the node is external or empty, the search is complete.
        // If the key doesn't match, node expansion will be done upon
        // insertion. If it does match, we've found our node.
        External(stored_key, _) if stored_key == key => (None, true),
        External(..) | Nothing => (None, false),
    }
}

// NB: All these methods assume a correctly constructed occupied entry (matching the given key).
impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry.
    #[inline]
    pub fn get(&self) -> &V {
        match *self.search_stack.peek_ref() {
            External(_, ref value) => value,
            // Invalid SearchStack, non-external last node.
            _ => unreachable!(),
        }
    }

    /// Gets a mutable reference to the value in the entry.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        match *self.search_stack.peek_ref() {
            External(_, ref mut value) => value,
            // Invalid SearchStack, non-external last node.
            _ => unreachable!(),
        }
    }

    /// Converts the OccupiedEntry into a mutable reference to the value in the entry,
    /// with a lifetime bound to the map itself.
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        match *self.search_stack.peek_ref() {
            External(_, ref mut value) => value,
            // Invalid SearchStack, non-external last node.
            _ => unreachable!(),
        }
    }

    /// Sets the value of the entry, and returns the entry's old value.
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        match *self.search_stack.peek_ref() {
            External(_, ref mut stored_value) => mem::replace(stored_value, value),
            // Invalid SearchStack, non-external last node.
            _ => unreachable!(),
        }
    }

    /// Takes the value out of the entry, and returns it.
    #[inline]
    pub fn remove(self) -> V {
        // This function removes the external leaf-node, then unwinds the search-stack
        // deleting now-childless ancestors.
        let mut search_stack = self.search_stack;

        // Extract the value from the leaf-node of interest.
        let leaf_node = mem::replace(search_stack.pop_ref(), Nothing);

        let value = match leaf_node {
            External(_, value) => value,
            // Invalid SearchStack, non-external last node.
            _ => unreachable!(),
        };

        // Iterate backwards through the search stack, deleting nodes if they are childless.
        // We compare each ancestor's parent count to 1 because each ancestor reached has just
        // had one of its children deleted.
        while !search_stack.is_empty() {
            let ancestor = search_stack.pop_ref();
            match *ancestor {
                Internal(ref mut internal) => {
                    // If stopping deletion, update the child count and break.
                    if internal.count != 1 {
                        internal.count -= 1;
                        break;
                    }
                }
                // Invalid SearchStack, non-internal ancestor node.
                _ => unreachable!(),
            }
            *ancestor = Nothing;
        }

        // Decrement the length of the entire map, for the removed node.
        unsafe {
            (*search_stack.map).length -= 1;
        }

        value
    }
}

impl<'a, K: Chunk, V> VacantEntry<'a, K, V> {
    /// Set the vacant entry to the given value.
    pub fn insert(self, value: V) -> &'a mut V {
        let search_stack = self.search_stack;
        let old_length = search_stack.length;

        // Update the map's length for the new element.
        unsafe {
            (*search_stack.map).length += 1;
        }

        // If there's only 1 node in the search stack, insert a new node below it at idx 1.
        if old_length == 1 {
            unsafe {
                // Note: Small hack to appease the borrow checker. Can't mutably borrow root.count
                let mut temp = (*search_stack.map).root.count;
                let (value_ref, _) = insert(
                    &mut temp,
                    search_stack.get_ref(0),
                    search_stack.key,
                    value,
                    1,
                );
                (*search_stack.map).root.count = temp;
                value_ref
            }
        }
        // Otherwise, find the predecessor of the last stack node, and insert as normal.
        else {
            match *search_stack.get_ref(old_length - 2) {
                Internal(ref mut parent) => {
                    let parent = &mut **parent;
                    let (value_ref, _) = insert(
                        &mut parent.count,
                        &mut parent.children[search_stack.key.chunk(old_length - 1)],
                        search_stack.key,
                        value,
                        old_length,
                    );
                    value_ref
                }
                // Invalid SearchStack, non-internal ancestor node.
                _ => unreachable!(),
            }
        }
    }
}

/// A forward iterator over a map.
pub struct Iter<'a, K: 'a, V: 'a> {
    stack: [slice::Iter<'a, TrieNode<K, V>>; MAX_DEPTH],
    length: usize,
    remaining: usize,
}

impl<'a, K, V> Clone for Iter<'a, K, V> {
    #[cfg(target_pointer_width = "32")]
    fn clone(&self) -> Iter<'a, T> {
        Iter {
            stack: [
                self.stack[0].clone(),
                self.stack[1].clone(),
                self.stack[2].clone(),
                self.stack[3].clone(),
                self.stack[4].clone(),
                self.stack[5].clone(),
                self.stack[6].clone(),
                self.stack[7].clone(),
            ],
            ..*self
        }
    }

    #[cfg(target_pointer_width = "64")]
    fn clone(&self) -> Iter<'a, K, V> {
        Iter {
            stack: [
                self.stack[0].clone(),
                self.stack[1].clone(),
                self.stack[2].clone(),
                self.stack[3].clone(),
                self.stack[4].clone(),
                self.stack[5].clone(),
                self.stack[6].clone(),
                self.stack[7].clone(),
                self.stack[8].clone(),
                self.stack[9].clone(),
                self.stack[10].clone(),
                self.stack[11].clone(),
                self.stack[12].clone(),
                self.stack[13].clone(),
                self.stack[14].clone(),
                self.stack[15].clone(),
            ],
            ..*self
        }
    }
}

/// A forward iterator over the key-value pairs of a map, with the
/// values being mutable.
pub struct IterMut<'a, K: 'a, V: 'a> {
    stack: [slice::IterMut<'a, TrieNode<K, V>>; MAX_DEPTH],
    length: usize,
    remaining: usize,
}

/// A forward iterator over the keys of a map.
pub struct Keys<'a, K: 'a, V: 'a>(Iter<'a, K, V>);

impl<'a, K, V> Clone for Keys<'a, K, V> {
    fn clone(&self) -> Keys<'a, K, V> {
        Keys(self.0.clone())
    }
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|e| e.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {}

/// A forward iterator over the values of a map.
pub struct Values<'a, K: 'a, V: 'a>(Iter<'a, K, V>);

impl<'a, K, V> Clone for Values<'a, K, V> {
    fn clone(&self) -> Values<'a, K, V> {
        Values(self.0.clone())
    }
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|e| e.1)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {}

macro_rules! iterator_impl {
    ($name:ident,
     iter = $iter:ident,
     mutability = ($($mut_:tt)*)) => {
        impl<'a, K, V> $name<'a, K, V> {
            // Create new zero'd iterator. We have a thin gilding of safety by
            // using init rather than uninit, so that the worst that can happen
            // from failing to initialise correctly after calling these is a
            // segfault.
            #[cfg(target_pointer_width="32")]
            unsafe fn new() -> Self {
                $name {
                    remaining: 0,
                    length: 0,
                    // ick :( ... at least the compiler will tell us if we screwed up.
                    stack: [IntoIterator::into_iter((& $($mut_)*[])),
                            IntoIterator::into_iter((& $($mut_)*[])),
                            IntoIterator::into_iter((& $($mut_)*[])),
                            IntoIterator::into_iter((& $($mut_)*[])),

                            IntoIterator::into_iter((& $($mut_)*[])),
                            IntoIterator::into_iter((& $($mut_)*[])),
                            IntoIterator::into_iter((& $($mut_)*[])),
                            IntoIterator::into_iter((& $($mut_)*[])),
                            ]
                }
            }

            #[cfg(target_pointer_width="64")]
            unsafe fn new() -> Self {
                $name {
                    remaining: 0,
                    length: 0,
                    stack: [IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),

                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),

                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),

                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),
                            IntoIterator::into_iter(& $($mut_)*[]),
                            ]
                }
            }
        }

        impl<'a, K, V> Iterator for $name<'a, K, V> {
            type Item = (&'a K, &'a $($mut_)* V);
            // you might wonder why we're not even trying to act within the
            // rules, and are just manipulating raw pointers like there's no
            // such thing as invalid pointers and memory unsafety. The
            // reason is performance, without doing this we can get the
            // (now replaced) bench_iter_large microbenchmark down to about
            // 30000 ns/iter (using .unsafe_get to index self.stack directly, 38000
            // ns/iter with [] checked indexing), but this smashes that down
            // to 13500 ns/iter.
            //
            // Fortunately, it's still safe...
            //
            // We have an invariant that every Internal node
            // corresponds to one push to self.stack, and one pop,
            // nested appropriately. self.stack has enough storage
            // to store the maximum depth of Internal nodes in the
            // trie (8 on 32-bit platforms, 16 on 64-bit).
            fn next(&mut self) -> Option<Self::Item> {
                let start_ptr = self.stack.as_mut_ptr();

                unsafe {
                    // write_ptr is the next place to write to the stack.
                    // invariant: start_ptr <= write_ptr < end of the
                    // vector.
                    let mut write_ptr = start_ptr.offset(self.length as isize);
                    while write_ptr != start_ptr {
                        // indexing back one is safe, since write_ptr >
                        // start_ptr now.
                        match (*write_ptr.offset(-1)).next() {
                            // exhausted this iterator (i.e. finished this
                            // Internal node), so pop from the stack.
                            //
                            // don't bother clearing the memory, because the
                            // next time we use it we'll've written to it
                            // first.
                            None => write_ptr = write_ptr.offset(-1),
                            Some(child) => {
                                match *child {
                                    Internal(ref $($mut_)* node) => {
                                        // going down a level, so push
                                        // to the stack (this is the
                                        // write referenced above)
                                        *write_ptr = node.children.$iter();
                                        write_ptr = write_ptr.offset(1);
                                    }
                                    External(ref key, ref $($mut_)* value) => {
                                        self.remaining -= 1;
                                        // store the new length of the
                                        // stack, based on our current
                                        // position.
                                        self.length = (write_ptr as usize
                                                        - start_ptr as usize) /
                                            mem::size_of::<slice::Iter<'_, TrieNode<K, V>>>();

                                        return Some((key, value));
                                    }
                                    Nothing => {}
                                }
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

        impl<'a, K, V> ExactSizeIterator for $name<'a, K, V> {
            fn len(&self) -> usize { self.remaining }
        }
    }
}

iterator_impl! { Iter, iter = iter, mutability = () }
iterator_impl! { IterMut, iter = iter_mut, mutability = (mut) }

/// A bounded forward iterator over a map.
pub struct Range<'a, K: 'a, V: 'a>(Iter<'a, K, V>);

impl<'a, K, V> Clone for Range<'a, K, V> {
    fn clone(&self) -> Range<'a, K, V> {
        Range(self.0.clone())
    }
}

impl<'a, K, V> Iterator for Range<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.0.remaining))
    }
}

/// A bounded forward iterator over the key-value pairs of a map, with the
/// values being mutable.
pub struct RangeMut<'a, K: 'a, V: 'a>(IterMut<'a, K, V>);

impl<'a, K, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.0.remaining))
    }
}

impl<'a, K: Chunk, V> IntoIterator for &'a Map<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K: Chunk, V> IntoIterator for &'a mut Map<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

#[cfg(test)]
mod test {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::hint::black_box;

    use super::Entry::*;
    use super::TrieNode::*;
    use super::{InternalNode, Map};

    fn check_integrity<K, V>(trie: &InternalNode<K, V>) {
        assert!(trie.count != 0);

        let mut sum = 0;

        for x in trie.children.iter() {
            match *x {
                Nothing => (),
                Internal(ref y) => {
                    check_integrity(&**y);
                    sum += 1
                }
                External(_, _) => sum += 1,
            }
        }

        assert_eq!(sum, trie.count);
    }

    #[test]
    fn test_find_mut() {
        let mut m = Map::new();
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
        let mut m = Map::new();
        assert!(m.get_mut(&0).is_none());
        assert!(m.insert(1, 12).is_none());
        assert!(m.get_mut(&0).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.get_mut(&0).is_none());
    }

    #[test]
    fn test_step() {
        let mut trie = Map::new();
        let n = 300;

        for x in (1..n).step_by(2) {
            assert!(trie.insert(x, x + 1).is_none());
            assert!(trie.contains_key(&x));
            check_integrity(&trie.root);
        }

        for x in (0..n).step_by(2) {
            assert!(!trie.contains_key(&x));
            assert!(trie.insert(x, x + 1).is_none());
            check_integrity(&trie.root);
        }

        for x in 0..n {
            assert!(trie.contains_key(&x));
            assert!(trie.insert(x, x + 1).is_some());
            check_integrity(&trie.root);
        }

        for x in (1..n).step_by(2) {
            assert!(trie.remove(&x).is_some());
            assert!(!trie.contains_key(&x));
            check_integrity(&trie.root);
        }

        for x in (0..n).step_by(2) {
            assert!(trie.contains_key(&x));
            assert!(trie.insert(x, x + 1).is_some());
            check_integrity(&trie.root);
        }
    }

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
        let mut m = Map::new();
        assert_eq!(m.insert(1, 2), None);
        assert_eq!(m.insert(1, 3), Some(2));
        assert_eq!(m.insert(1, 4), Some(3));
    }

    #[test]
    fn test_remove() {
        let mut m = Map::new();
        m.insert(1, 2);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: Map<usize, i32> = xs.iter().cloned().collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

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

    #[test]
    fn test_bound() {
        let empty_map: Map<usize, usize> = Map::new();
        assert_eq!(empty_map.lower_bound(&0).next(), None);
        assert_eq!(empty_map.upper_bound(&0).next(), None);

        let last = 999;
        let step = 3;
        let value = 42;

        let mut map: Map<usize, usize> = Map::new();
        for x in (0..last).step_by(step) {
            assert!(x % step == 0);
            map.insert(x, value);
        }

        for i in 0..last - step {
            let mut lb = map.lower_bound(&i);
            let mut ub = map.upper_bound(&i);
            let next_key = i - i % step + step;
            let next_pair = (&next_key, &value);
            if i % step == 0 {
                assert_eq!(lb.next(), Some((&i, &value)));
            } else {
                assert_eq!(lb.next(), Some(next_pair));
            }
            assert_eq!(ub.next(), Some(next_pair));
        }

        let mut lb = map.lower_bound(&(last - step));
        assert_eq!(lb.next(), Some((&(last - step), &value)));
        let mut ub = map.upper_bound(&(last - step));
        assert_eq!(ub.next(), None);

        for i in last - step + 1..last {
            let mut lb = map.lower_bound(&i);
            assert_eq!(lb.next(), None);
            let mut ub = map.upper_bound(&i);
            assert_eq!(ub.next(), None);
        }
    }

    #[test]
    fn test_mut_bound() {
        let empty_map: Map<usize, usize> = Map::new();
        assert_eq!(empty_map.lower_bound(&0).next(), None);
        assert_eq!(empty_map.upper_bound(&0).next(), None);

        let mut m_lower = Map::new();
        let mut m_upper = Map::new();
        for i in 0..100 {
            m_lower.insert(2 * i, 4 * i);
            m_upper.insert(2 * i, 4 * i);
        }

        for i in 0..199 {
            let mut lb_it = m_lower.lower_bound_mut(&i);
            let (&k, v) = lb_it.next().unwrap();
            let lb = i + i % 2;
            assert_eq!(lb, k);
            *v -= k;
        }

        for i in 0..198 {
            let mut ub_it = m_upper.upper_bound_mut(&i);
            let (&k, v) = ub_it.next().unwrap();
            let ub = i + 2 - i % 2;
            assert_eq!(ub, k);
            *v -= k;
        }

        assert!(m_lower.lower_bound_mut(&199).next().is_none());
        assert!(m_upper.upper_bound_mut(&198).next().is_none());

        assert!(m_lower.iter().all(|(_, &x)| x == 0));
        assert!(m_upper.iter().all(|(_, &x)| x == 0));
    }

    #[test]
    fn test_clone() {
        let mut a = Map::new();

        a.insert(1, 'a');
        a.insert(2, 'b');
        a.insert(3, 'c');

        assert!(a.clone() == a);
    }

    #[test]
    fn test_eq() {
        let mut a = Map::new();
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

    #[test]
    fn test_debug() {
        let mut map = Map::new();
        let empty: Map<usize, char> = Map::new();

        map.insert(1, 'a');
        map.insert(2, 'b');

        assert_eq!(format!("{:?}", map), "{1: 'a', 2: 'b'}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[test]
    fn test_index() {
        let mut map = Map::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[&2], 1);
    }

    #[test]
    #[should_panic]
    fn test_index_nonexistent() {
        let mut map = Map::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        black_box(map[&4]);
    }

    // Number of items to insert into the map during entry tests.
    // The tests rely on it being even.
    const SQUARES_UPPER_LIM: usize = 128;

    /// Make a map storing i^2 for i in [0, 128)
    fn squares_map() -> Map<usize, usize> {
        let mut map = Map::new();
        for i in 0..SQUARES_UPPER_LIM {
            map.insert(i, i * i);
        }
        map
    }

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

    #[test]
    fn test_entry_get_mut() {
        let mut map = squares_map();

        // Change the entries to cubes.
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

    #[test]
    fn test_entry_take() {
        let mut map = squares_map();
        assert_eq!(map.len(), SQUARES_UPPER_LIM);

        // Remove every odd key, checking that the correct value is returned.
        for i in (1..SQUARES_UPPER_LIM).step_by(2) {
            match map.entry(i) {
                Occupied(e) => assert_eq!(e.remove(), i * i),
                Vacant(_) => panic!("Key not found."),
            }
        }

        check_integrity(&map.root);

        // Check that the values for even keys remain unmodified.
        for i in (0..SQUARES_UPPER_LIM).step_by(2) {
            assert_eq!(map.get(&i).unwrap(), &(i * i));
        }

        assert_eq!(map.len(), SQUARES_UPPER_LIM / 2);
    }

    #[test]
    fn test_occupied_entry_set() {
        let mut map = squares_map();

        // Change all the entries to cubes.
        for i in 0..SQUARES_UPPER_LIM {
            match map.entry(i) {
                Occupied(mut e) => assert_eq!(e.insert(i * i * i), i * i),
                Vacant(_) => panic!("Key not found."),
            }
            assert_eq!(map.get(&i).unwrap(), &(i * i * i));
        }
        check_integrity(&map.root);
    }

    #[test]
    fn test_vacant_entry_set() {
        let mut map = Map::new();

        for i in 0..SQUARES_UPPER_LIM {
            match map.entry(i) {
                Vacant(e) => {
                    // Insert i^2.
                    let inserted_val = e.insert(i * i);
                    assert_eq!(*inserted_val, i * i);

                    // Update it to i^3 using the returned mutable reference.
                    *inserted_val = i * i * i;
                }
                _ => panic!("Non-existent key found."),
            }
            assert_eq!(map.get(&i).unwrap(), &(i * i * i));
        }

        check_integrity(&map.root);
        assert_eq!(map.len(), SQUARES_UPPER_LIM);
    }

    #[test]
    fn test_single_key() {
        let mut map = Map::new();
        map.insert(1, 2);

        if let Occupied(e) = map.entry(1) {
            e.remove();
        }
    }
}
