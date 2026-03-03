// Copyright 2014-2026 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(conventions): implement bounded iterators
// FIXME(conventions): replace each_reverse by making iter DoubleEnded
// FIXME(conventions): implement iter_mut and into_iter

//! An ordered set based on a trie.

use std::cmp::Ordering::{self, Equal, Greater, Less};
use std::fmt::{self, Debug};
use std::iter::{self, Peekable};
use std::ops;

use crate::Chunk;

use super::map::{self, Map};

/// A set implemented as a radix trie.
///
/// # Examples
///
/// ```
/// let mut set = trie::Set::new();
/// set.insert(6);
/// set.insert(28);
/// set.insert(6);
///
/// assert_eq!(set.len(), 2);
///
/// if !set.contains(&3) {
///     println!("3 is not in the set");
/// }
///
/// // Print contents in order
/// for x in set.iter() {
///     println!("{}", x);
/// }
///
/// set.remove(&6);
/// assert_eq!(set.len(), 1);
///
/// set.clear();
/// assert!(set.is_empty());
/// ```
#[derive(Clone, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Set<T: Chunk> {
    map: Map<T, ()>,
}

impl<T: Chunk + Debug + PartialOrd> Debug for Set<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T: Chunk> Set<T> {
    /// Creates an empty set.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut set: trie::Set<usize> = trie::Set::new();
    /// ```
    #[inline]
    pub fn new() -> Set<T> {
        Set { map: Map::new() }
    }

    /// Visits all values in reverse order. Aborts traversal when `f` returns `false`.
    /// Returns `true` if `f` returns `true` for all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let set: trie::Set<usize> = [1, 2, 3, 4, 5].iter().cloned().collect();
    ///
    /// let mut vec = vec![];
    /// assert_eq!(true, set.each_reverse(|&x| { vec.push(x); true }));
    /// assert_eq!(vec, [5, 4, 3, 2, 1]);
    ///
    /// // Stop when we reach 3
    /// let mut vec = vec![];
    /// assert_eq!(false, set.each_reverse(|&x| { vec.push(x); x != 3 }));
    /// assert_eq!(vec, [5, 4, 3]);
    /// ```
    #[inline]
    pub fn each_reverse<F>(&self, mut f: F) -> bool
    where
        F: FnMut(&T) -> bool,
    {
        self.map.each_reverse(|k, _| f(k))
    }

    /// Gets an iterator over the values in the set, in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut set = trie::Set::new();
    /// set.insert(3);
    /// set.insert(2);
    /// set.insert(1);
    /// set.insert(2);
    ///
    /// // Print 1, 2, 3
    /// for x in set.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            iter: self.map.iter(),
        }
    }
}

impl<T: Chunk + PartialOrd> Set<T> {
    /// Gets an iterator pointing to the first value that is not less than `val`.
    /// If all values in the set are less than `val` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let set: trie::Set<usize> = [2, 4, 6, 8].iter().cloned().collect();
    /// assert_eq!(set.lower_bound(&4).next(), Some(&4));
    /// assert_eq!(set.lower_bound(&5).next(), Some(&6));
    /// assert_eq!(set.lower_bound(&10).next(), None);
    /// ```
    pub fn lower_bound(&self, val: &T) -> Range<'_, T> {
        Range {
            iter: self.map.lower_bound(val),
        }
    }

    /// Gets an iterator pointing to the first value that key is greater than `val`.
    /// If all values in the set are less than or equal to `val` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let set: trie::Set<usize> = [2, 4, 6, 8].iter().cloned().collect();
    /// assert_eq!(set.upper_bound(&4).next(), Some(&6));
    /// assert_eq!(set.upper_bound(&5).next(), Some(&6));
    /// assert_eq!(set.upper_bound(&10).next(), None);
    /// ```
    pub fn upper_bound(&self, val: &T) -> Range<'_, T> {
        Range {
            iter: self.map.upper_bound(val),
        }
    }

    /// Visits the values representing the difference, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// let a: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// let b: trie::Set<usize> = [3, 4, 5].iter().cloned().collect();
    ///
    /// // Can be seen as `a - b`.
    /// for x in a.difference(&b) {
    ///     println!("{}", x); // Print 1 then 2
    /// }
    ///
    /// let diff1: trie::Set<usize> = a.difference(&b).copied().collect();
    /// assert_eq!(diff1, [1, 2].iter().cloned().collect());
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else:
    /// let diff2: trie::Set<usize> = b.difference(&a).copied().collect();
    /// assert_eq!(diff2, [4, 5].iter().cloned().collect());
    /// ```
    pub fn difference<'a>(&'a self, other: &'a Set<T>) -> Difference<'a, T> {
        Difference {
            a: self.iter().peekable(),
            b: other.iter().peekable(),
        }
    }

    /// Visits the values representing the symmetric difference, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// let a: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// let b: trie::Set<usize> = [3, 4, 5].iter().cloned().collect();
    ///
    /// // Print 1, 2, 4, 5 in ascending order.
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff1: trie::Set<usize> = a.symmetric_difference(&b).copied().collect();
    /// let diff2: trie::Set<usize> = b.symmetric_difference(&a).copied().collect();
    ///
    /// assert_eq!(diff1, diff2);
    /// assert_eq!(diff1, [1, 2, 4, 5].iter().cloned().collect());
    /// ```
    pub fn symmetric_difference<'a>(&'a self, other: &'a Set<T>) -> SymmetricDifference<'a, T> {
        SymmetricDifference {
            a: self.iter().peekable(),
            b: other.iter().peekable(),
        }
    }

    /// Visits the values representing the intersection, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// let a: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// let b: trie::Set<usize> = [2, 3, 4].iter().cloned().collect();
    ///
    /// // Print 2, 3 in ascending order.
    /// for x in a.intersection(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff: trie::Set<usize> = a.intersection(&b).copied().collect();
    /// assert_eq!(diff, [2, 3].iter().cloned().collect());
    /// ```
    pub fn intersection<'a>(&'a self, other: &'a Set<T>) -> Intersection<'a, T> {
        Intersection {
            a: self.iter().peekable(),
            b: other.iter().peekable(),
        }
    }

    /// Visits the values representing the union, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// let a: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// let b: trie::Set<usize> = [3, 4, 5].iter().cloned().collect();
    ///
    /// // Print 1, 2, 3, 4, 5 in ascending order.
    /// for x in a.union(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff: trie::Set<usize> = a.union(&b).copied().collect();
    /// assert_eq!(diff, [1, 2, 3, 4, 5].iter().cloned().collect());
    /// ```
    pub fn union<'a>(&'a self, other: &'a Set<T>) -> Union<'a, T> {
        Union {
            a: self.iter().peekable(),
            b: other.iter().peekable(),
        }
    }
}

impl<T: Chunk> Set<T> {
    /// Return the number of elements in the set
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = trie::Set::new();
    /// assert_eq!(v.len(), 0);
    /// v.insert(1);
    /// assert_eq!(v.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if the set contains no elements
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = trie::Set::new();
    /// assert!(v.is_empty());
    /// v.insert(1);
    /// assert!(!v.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clears the set, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = trie::Set::new();
    /// v.insert(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.map.clear()
    }

    /// Returns `true` if the set contains a value.
    ///
    /// # Examples
    ///
    /// ```
    /// let set: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.map.contains_key(value)
    }

    /// Returns `true` if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// let a: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// let mut b = trie::Set::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    #[inline]
    pub fn is_disjoint(&self, other: &Set<T>) -> bool {
        self.iter().all(|v| !other.contains(v))
    }

    /// Returns `true` if the set is a subset of another.
    ///
    /// # Examples
    ///
    /// ```
    /// let sup: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// let mut set = trie::Set::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    #[inline]
    pub fn is_subset(&self, other: &Set<T>) -> bool {
        self.iter().all(|v| other.contains(v))
    }

    /// Returns `true` if the set is a superset of another.
    ///
    /// # Examples
    ///
    /// ```
    /// let sub: trie::Set<usize> = [1, 2].iter().cloned().collect();
    /// let mut set = trie::Set::new();
    ///
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(0);
    /// set.insert(1);
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(2);
    /// assert_eq!(set.is_superset(&sub), true);
    /// ```
    #[inline]
    pub fn is_superset(&self, other: &Set<T>) -> bool {
        other.is_subset(self)
    }

    /// Adds a value to the set. Returns `true` if the value was not already
    /// present in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut set = trie::Set::new();
    ///
    /// assert_eq!(set.insert(2), true);
    /// assert_eq!(set.insert(2), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        self.map.insert(value, ()).is_none()
    }

    /// Removes a value from the set. Returns `true` if the value was
    /// present in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut set = trie::Set::new();
    ///
    /// set.insert(2);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
    #[inline]
    pub fn remove(&mut self, value: &T) -> bool {
        self.map.remove(value).is_some()
    }
}

impl<T: Chunk> iter::FromIterator<T> for Set<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Set<T> {
        let mut set = Set::new();
        set.extend(iter);
        set
    }
}

impl<T: Chunk> Extend<T> for Set<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for elem in iter {
            self.insert(elem);
        }
    }
}

impl<T: Chunk + Clone + Ord> ops::BitOr<&Set<T>> for &Set<T> {
    type Output = Set<T>;

    /// Returns the union of `self` and `rhs` as a new set.
    ///
    /// # Example
    ///
    /// ```
    /// let a: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// let b: trie::Set<usize> = [3, 4, 5].iter().cloned().collect();
    ///
    /// let set: trie::Set<usize> = &a | &b;
    /// let v: Vec<usize> = set.iter().copied().collect();
    /// assert_eq!(v, [1, 2, 3, 4, 5]);
    /// ```
    fn bitor(self, rhs: &Set<T>) -> Set<T> {
        self.union(rhs).cloned().collect()
    }
}

impl<T: Chunk + Clone + Ord> ops::BitAnd<&Set<T>> for &Set<T> {
    type Output = Set<T>;

    /// Returns the intersection of `self` and `rhs` as a new set.
    ///
    /// # Example
    ///
    /// ```
    /// let a: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// let b: trie::Set<usize> = [2, 3, 4].iter().cloned().collect();
    ///
    /// let set: trie::Set<usize> = &a & &b;
    /// let v: Vec<usize> = set.iter().copied().collect();
    /// assert_eq!(v, [2, 3]);
    /// ```
    fn bitand(self, rhs: &Set<T>) -> Set<T> {
        self.intersection(rhs).cloned().collect()
    }
}

impl<T: Chunk + Clone + Ord> ops::BitXor<&Set<T>> for &Set<T> {
    type Output = Set<T>;

    /// Returns the symmetric difference of `self` and `rhs` as a new set.
    ///
    /// # Example
    ///
    /// ```
    /// let a: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// let b: trie::Set<usize> = [3, 4, 5].iter().cloned().collect();
    ///
    /// let set: trie::Set<usize> = &a ^ &b;
    /// let v: Vec<usize> = set.iter().copied().collect();
    /// assert_eq!(v, [1, 2, 4, 5]);
    /// ```
    fn bitxor(self, rhs: &Set<T>) -> Set<T> {
        self.symmetric_difference(rhs).cloned().collect()
    }
}

impl<T: Chunk + Clone + Ord> ops::Sub<&Set<T>> for &Set<T> {
    type Output = Set<T>;

    /// Returns the difference of `self` and `rhs` as a new set.
    ///
    /// # Example
    ///
    /// ```
    /// let a: trie::Set<usize> = [1, 2, 3].iter().cloned().collect();
    /// let b: trie::Set<usize> = [3, 4, 5].iter().cloned().collect();
    ///
    /// let set: trie::Set<usize> = &a - &b;
    /// let v: Vec<usize> = set.iter().copied().collect();
    /// assert_eq!(v, [1, 2]);
    /// ```
    fn sub(self, rhs: &Set<T>) -> Set<T> {
        self.difference(rhs).cloned().collect()
    }
}

/// A forward iterator over a set.
#[derive(Clone)]
pub struct Iter<'a, T: 'a> {
    iter: map::Iter<'a, T, ()>,
}

/// A bounded forward iterator over a set.
#[derive(Clone)]
pub struct Range<'a, T: 'a> {
    iter: map::Range<'a, T, ()>,
}

/// An iterator producing elements in the set difference (in-order).
#[derive(Clone)]
pub struct Difference<'a, T: 'a> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

/// An iterator producing elements in the set symmetric difference (in-order).
#[derive(Clone)]
pub struct SymmetricDifference<'a, T: 'a> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

/// An iterator producing elements in the set intersection (in-order).
#[derive(Clone)]
pub struct Intersection<'a, T: 'a> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

/// An iterator producing elements in the set union (in-order).
#[derive(Clone)]
pub struct Union<'a, T: 'a> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

/// Compare `x` and `y`, but return `short` if x is None and `long` if y is None
fn cmp_opt<T: Ord>(x: Option<&T>, y: Option<&T>, short: Ordering, long: Ordering) -> Ordering {
    match (x, y) {
        (None, _) => short,
        (_, None) => long,
        (Some(x1), Some(y1)) => x1.cmp(y1),
    }
}

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(key, _)| key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: 'a> ExactSizeIterator for Iter<'a, T> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, T: 'a> Iterator for Range<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(key, _)| key)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: 'a + Ord> Iterator for Difference<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match cmp_opt(self.a.peek().copied(), self.b.peek().copied(), Less, Less) {
                Less => return self.a.next(),
                Equal => {
                    self.a.next();
                    self.b.next();
                }
                Greater => {
                    self.b.next();
                }
            }
        }
    }
}

impl<'a, T: Ord + 'a> Iterator for SymmetricDifference<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match cmp_opt(
                self.a.peek().copied(),
                self.b.peek().copied(),
                Greater,
                Less,
            ) {
                Less => return self.a.next(),
                Equal => {
                    self.a.next();
                    self.b.next();
                }
                Greater => return self.b.next(),
            }
        }
    }
}

impl<'a, T: Ord + 'a> Iterator for Intersection<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let o_cmp = match (self.a.peek(), self.b.peek()) {
                (None, _) => None,
                (_, None) => None,
                (Some(a1), Some(b1)) => Some(a1.cmp(b1)),
            };
            match o_cmp {
                None => return None,
                Some(Less) => {
                    self.a.next();
                }
                Some(Equal) => {
                    self.b.next();
                    return self.a.next();
                }
                Some(Greater) => {
                    self.b.next();
                }
            }
        }
    }
}

impl<'a, T: Ord + 'a> Iterator for Union<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        match cmp_opt(
            self.a.peek().copied(),
            self.b.peek().copied(),
            Greater,
            Less,
        ) {
            Less => self.a.next(),
            Equal => {
                self.b.next();
                self.a.next()
            }
            Greater => self.b.next(),
        }
    }
}

impl<'a, T: Chunk + 'a> IntoIterator for &'a Set<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[cfg(test)]
mod test {
    use super::Set;
    use crate::chunk::USIZE_BITS;

    #[test]
    fn test_sane_chunk() {
        let x = 1;
        let y = 1 << (USIZE_BITS - 1);

        let mut trie = Set::new();

        assert!(trie.insert(x));
        assert!(trie.insert(y));

        assert_eq!(trie.len(), 2);

        let expected = [x, y];

        for (i, &x) in trie.iter().enumerate() {
            assert_eq!(expected[i], x);
        }
    }

    #[test]
    fn test_from_iter() {
        let xs = [9, 8, 7, 6, 5, 4, 3, 2, 1];

        let set: Set<usize> = xs.iter().cloned().collect();

        for x in xs.iter() {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_debug() {
        let mut set = Set::new();
        let empty: Set<usize> = Set::new();

        set.insert(1);
        set.insert(2);

        assert_eq!(format!("{:?}", set), "{1, 2}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[test]
    fn test_clone() {
        let mut a = Set::new();

        a.insert(1);
        a.insert(2);
        a.insert(3);

        assert!(a.clone() == a);
    }

    #[test]
    fn test_lt() {
        let mut a = Set::new();
        let mut b = Set::new();

        assert!((a >= b) && (b >= a));
        assert!(b.insert(2));
        assert!(a < b);
        assert!(a.insert(3));
        assert!((a >= b) && b < a);
        assert!(b.insert(1));
        assert!(b < a);
        assert!(a.insert(0));
        assert!(a < b);
        assert!(a.insert(6));
        assert!(a < b && (b >= a));
    }

    #[test]
    fn test_ord() {
        let mut a = Set::new();
        let mut b = Set::new();

        assert!(a == b);
        assert!(a.insert(1));
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2));
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    fn check<F>(a: &[usize], b: &[usize], expected: &[usize], f: F)
    where
        F: FnOnce(
            &Set<usize>,
            &Set<usize>,
            &mut usize,
            &[usize],
            Box<dyn FnMut(&usize, &mut usize, &[usize]) -> bool>,
        ) -> bool,
    {
        let mut set_a = Set::new();
        let mut set_b = Set::new();

        for x in a.iter() {
            assert!(set_a.insert(*x))
        }
        for y in b.iter() {
            assert!(set_b.insert(*y))
        }

        let mut i = 0;
        f(
            &set_a,
            &set_b,
            &mut i,
            expected,
            Box::new(|&x: &usize, i: &mut usize, expected: &[usize]| {
                assert_eq!(x, expected[*i]);
                *i += 1;
                true
            }),
        );
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_intersection() {
        fn check_intersection(a: &[usize], b: &[usize], expected: &[usize]) {
            check(a, b, expected, |x, y, i, expected, mut f| {
                x.intersection(y).all(|elem| f(elem, i, expected))
            })
        }

        check_intersection(&[], &[], &[]);
        check_intersection(&[1, 2, 3], &[], &[]);
        check_intersection(&[], &[1, 2, 3], &[]);
        check_intersection(&[2], &[1, 2, 3], &[2]);
        check_intersection(&[1, 2, 3], &[2], &[2]);
        check_intersection(&[11, 1, 3, 77, 103, 5], &[2, 11, 77, 5, 3], &[3, 5, 11, 77]);
    }

    #[test]
    fn test_difference() {
        fn check_difference(a: &[usize], b: &[usize], expected: &[usize]) {
            check(a, b, expected, |x, y, i, e, mut f| {
                x.difference(y).all(|elem| f(elem, i, e))
            })
        }

        check_difference(&[], &[], &[]);
        check_difference(&[1, 12], &[], &[1, 12]);
        check_difference(&[], &[1, 2, 3, 9], &[]);
        check_difference(&[1, 3, 5, 9, 11], &[3, 9], &[1, 5, 11]);
        check_difference(
            &[11, 22, 33, 40, 42],
            &[14, 23, 34, 38, 39, 50],
            &[11, 22, 33, 40, 42],
        );
    }

    #[test]
    fn test_symmetric_difference() {
        fn check_symmetric_difference(a: &[usize], b: &[usize], expected: &[usize]) {
            check(a, b, expected, |x, y, i, e, mut f| {
                x.symmetric_difference(y).all(|elem| f(elem, i, e))
            })
        }

        check_symmetric_difference(&[], &[], &[]);
        check_symmetric_difference(&[1, 2, 3], &[2], &[1, 3]);
        check_symmetric_difference(&[2], &[1, 2, 3], &[1, 3]);
        check_symmetric_difference(&[1, 3, 5, 9, 11], &[3, 9, 14, 22], &[1, 5, 11, 14, 22]);
    }

    #[test]
    fn test_union() {
        fn check_union(a: &[usize], b: &[usize], expected: &[usize]) {
            check(a, b, expected, |x, y, i, e, mut f| {
                x.union(y).all(|elem| f(elem, i, e))
            })
        }

        check_union(&[], &[], &[]);
        check_union(&[1, 2, 3], &[2], &[1, 2, 3]);
        check_union(&[2], &[1, 2, 3], &[1, 2, 3]);
        check_union(
            &[1, 3, 5, 9, 11, 16, 19, 24],
            &[1, 5, 9, 13, 19],
            &[1, 3, 5, 9, 11, 13, 16, 19, 24],
        );
    }

    #[test]
    fn test_bit_or() {
        let a: Set<usize> = [1, 2, 3].iter().cloned().collect();
        let b: Set<usize> = [3, 4, 5].iter().cloned().collect();

        let set: Set<usize> = &a | &b;
        let v: Vec<usize> = set.iter().copied().collect();
        assert_eq!(v, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_bit_and() {
        let a: Set<usize> = [1, 2, 3].iter().cloned().collect();
        let b: Set<usize> = [2, 3, 4].iter().cloned().collect();

        let set: Set<usize> = &a & &b;
        let v: Vec<usize> = set.iter().copied().collect();
        assert_eq!(v, [2, 3]);
    }

    #[test]
    fn test_bit_xor() {
        let a: Set<usize> = [1, 2, 3].iter().cloned().collect();
        let b: Set<usize> = [3, 4, 5].iter().cloned().collect();

        let set: Set<usize> = &a ^ &b;
        let v: Vec<usize> = set.iter().copied().collect();
        assert_eq!(v, [1, 2, 4, 5]);
    }

    #[test]
    fn test_sub() {
        let a: Set<usize> = [1, 2, 3].iter().cloned().collect();
        let b: Set<usize> = [3, 4, 5].iter().cloned().collect();

        let set: Set<usize> = &a - &b;
        let v: Vec<usize> = set.iter().copied().collect();
        assert_eq!(v, [1, 2]);
    }
}
