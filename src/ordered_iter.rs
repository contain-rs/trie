extern crate ordered_iter;

use self::ordered_iter::{OrderedMapIterator, OrderedSetIterator};
use super::{map, set};

impl<'a, T> OrderedMapIterator for map::Iter<'a, T> {
    type Key = usize;
    type Val = &'a T;
}

impl<'a, T> OrderedMapIterator for map::IterMut<'a, T> {
    type Key = usize;
    type Val = &'a mut T;
}

impl<'a, T> OrderedMapIterator for map::Range<'a, T> {
    type Key = usize;
    type Val = &'a T;
}

impl<'a, T> OrderedMapIterator for map::RangeMut<'a, T> {
    type Key = usize;
    type Val = &'a mut T;
}

impl<'a, T> OrderedSetIterator for map::Keys<'a, T> {}

impl<'a> OrderedSetIterator for set::Iter<'a> {}

impl<'a> OrderedSetIterator for set::Range<'a> {}

impl<'a> OrderedSetIterator for set::Difference<'a> {}

impl<'a> OrderedSetIterator for set::Intersection<'a> {}

impl<'a> OrderedSetIterator for set::SymmetricDifference<'a> {}

impl<'a> OrderedSetIterator for set::Union<'a> {}
