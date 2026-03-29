use core::hash;
use std::collections::{BTreeMap, btree_map, hash_map};
use std::hash::Hash;
use std::marker::PhantomData;
use std::{borrow::Borrow, collections::HashMap};
use std::{iter, mem, ptr};

use crate::Chunk;
use crate::map_trait::Unit;

pub(crate) enum Entry<Occupied, Vacant> {
    Occupied(Occupied),
    Vacant(Vacant),
}

pub trait Occupied<'a, K, V> {
    fn into_mut(self) -> &'a mut V;
    fn insert(&mut self, value: V) -> V;
    fn into_key(self) -> K;
}

pub trait Vacant<'a, K, V> {
    fn len(&self) -> usize;
    fn should_split(&self) -> bool;
    fn insert(self, value: V) -> &'a mut V;
    fn into_key(self) -> K;
}

pub trait Inner<K, V>: IntoIterator<Item = (K, V)> {
    type Occupied<'a>: Occupied<'a, K, V>
    where
        Self: 'a;
    type Vacant<'a>: Vacant<'a, K, V>
    where
        Self: 'a;
    type Iter<'a>: Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
        Self: 'a;
    type IterMut<'a>: Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a;
    fn new(key: K, value: V) -> Self;
    fn get<Q: Chunk>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>;
    fn get_mut<Q: Chunk>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>;
    fn values_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut V>
    where
        V: 'a;
    fn should_remove(&self) -> bool;
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Chunk;
    fn insert(&mut self, key: K, value: V) -> Option<V>;
    fn entry<'a>(&'a mut self, key: K) -> Entry<Self::Occupied<'a>, Self::Vacant<'a>>
    where
        K: PartialEq;
    fn iter(&self) -> Self::Iter<'_>;
    fn len(&self) -> usize;
}

pub struct WithLen<T> {
    field: T,
    len: usize,
}

impl<'a, K, V> Occupied<'a, K, V> for WithLen<hash_map::OccupiedEntry<'a, K, V>> {
    fn into_mut(mut self) -> &'a mut V {
        self.field.into_mut()
    }

    fn insert(&mut self, value: V) -> V {
        self.field.insert(value)
    }

    fn into_key(self) -> K {
        unsafe {
            let key = ptr::read(self.field.key());
            mem::forget(self);
            key
        }
    }
}

impl<'a, K, V> Vacant<'a, K, V> for WithLen<hash_map::VacantEntry<'a, K, V>> {
    fn insert(self, value: V) -> &'a mut V {
        self.field.insert(value)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn should_split(&self) -> bool {
        self.len() > 128
    }

    fn into_key(self) -> K {
        self.field.into_key()
    }
}

type HashMapVacant<'a, K, V> = WithLen<hash_map::VacantEntry<'a, K, V>>;
type HashMapOccupied<'a, K, V> = WithLen<hash_map::OccupiedEntry<'a, K, V>>;
type HashMapEntry<'a, K, V> = Entry<HashMapOccupied<'a, K, V>, HashMapVacant<'a, K, V>>;

impl<K, V> Inner<K, V> for HashMap<K, V>
where
    K: Eq + Hash,
{
    type Occupied<'a>
        = WithLen<hash_map::OccupiedEntry<'a, K, V>>
    where
        Self: 'a;
    type Vacant<'a>
        = WithLen<hash_map::VacantEntry<'a, K, V>>
    where
        Self: 'a;
    type Iter<'a>
        = hash_map::Iter<'a, K, V>
    where
        K: 'a,
        V: 'a;
    type IterMut<'a>
        = hash_map::IterMut<'a, K, V>
    where
        K: 'a,
        V: 'a;

    fn new(key: K, value: V) -> Self {
        iter::once((key, value)).collect()
    }

    fn get<Q: Chunk>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        HashMap::get(self, key)
    }

    fn get_mut<Q: Chunk>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        HashMap::get_mut(self, key)
    }

    fn values_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut V>
    where
        V: 'a,
    {
        self.values_mut()
    }

    fn should_remove(&self) -> bool {
        self.len() <= 1
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert(key, value)
    }

    fn entry(&mut self, key: K) -> HashMapEntry<'_, K, V> {
        let len = self.len();
        match self.entry(key) {
            hash_map::Entry::Occupied(occupied) => Entry::Occupied(WithLen {
                field: occupied,
                len,
            }),
            hash_map::Entry::Vacant(vacant) => Entry::Vacant(WithLen { field: vacant, len }),
        }
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Chunk,
    {
        self.remove(key)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

type BTreeMapVacant<'a, K, V> = WithLen<btree_map::VacantEntry<'a, K, V>>;
type BTreeMapOccupied<'a, K, V> = WithLen<btree_map::OccupiedEntry<'a, K, V>>;
type BTreeMapEntry<'a, K, V> = Entry<BTreeMapOccupied<'a, K, V>, BTreeMapVacant<'a, K, V>>;

impl<'a, K: Ord, V> Occupied<'a, K, V> for WithLen<btree_map::OccupiedEntry<'a, K, V>> {
    fn into_mut(self) -> &'a mut V {
        self.field.into_mut()
    }

    fn insert(&mut self, value: V) -> V {
        self.field.insert(value)
    }

    fn into_key(self) -> K {
        unsafe {
            let key = ptr::read(self.field.key());
            mem::forget(self);
            key
        }
    }
}

impl<'a, K: Ord, V> Vacant<'a, K, V> for WithLen<btree_map::VacantEntry<'a, K, V>> {
    fn insert(self, value: V) -> &'a mut V {
        self.field.insert(value)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn should_split(&self) -> bool {
        self.len() > 128
    }

    fn into_key(self) -> K {
        self.field.into_key()
    }
}

impl<K, V> Inner<K, V> for BTreeMap<K, V>
where
    K: Ord,
{
    type Occupied<'a>
        = BTreeMapOccupied<'a, K, V>
    where
        Self: 'a;
    type Vacant<'a>
        = BTreeMapVacant<'a, K, V>
    where
        Self: 'a;
    type Iter<'a>
        = btree_map::Iter<'a, K, V>
    where
        K: 'a,
        V: 'a;
    type IterMut<'a>
        = btree_map::IterMut<'a, K, V>
    where
        K: 'a,
        V: 'a;

    fn new(key: K, value: V) -> Self {
        iter::once((key, value)).collect()
    }

    fn get<Q: Ord>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        BTreeMap::get(self, key)
    }

    fn get_mut<Q: Ord>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        BTreeMap::get_mut(self, key)
    }

    fn values_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut V>
    where
        V: 'a,
    {
        self.values_mut()
    }

    fn entry<'a>(&'a mut self, key: K) -> Entry<Self::Occupied<'a>, Self::Vacant<'a>> {
        let len = self.len();
        match self.entry(key) {
            btree_map::Entry::Occupied(occupied) => Entry::Occupied(WithLen {
                field: occupied,
                len,
            }),
            btree_map::Entry::Vacant(vacant) => Entry::Vacant(WithLen { field: vacant, len }),
        }
    }

    fn should_remove(&self) -> bool {
        self.len() <= 1
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert(key, value)
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Chunk,
    {
        self.remove(key)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

pub struct UnitOccupied<'a, K, V> {
    our_key: K,
    key: &'a mut K,
    value: &'a mut V,
}

pub struct UnitVacant<'a, K, V> {
    key: K,
    phantom: PhantomData<&'a V>,
}

impl<'a, K, V> Occupied<'a, K, V> for UnitOccupied<'a, K, V> {
    fn into_mut(self) -> &'a mut V {
        self.value
    }

    fn insert(&mut self, value: V) -> V {
        mem::replace(&mut self.value, value)
    }

    fn into_key(self) -> K {
        self.our_key
    }
}

impl<'a, K, V> Vacant<'a, K, V> for UnitVacant<'a, K, V> {
    fn insert(self, value: V) -> &'a mut V {
        unreachable!()
    }

    fn len(&self) -> usize {
        1
    }

    fn should_split(&self) -> bool {
        true
    }

    fn into_key(self) -> K {
        self.key
    }
}

impl<K, V> IntoIterator for Unit<K, V> {
    type IntoIter = iter::Once<(K, V)>;
    type Item = (K, V);

    fn into_iter(self) -> Self::IntoIter {
        iter::once((self.key, self.value))
    }
}

impl<'a, K, V> IntoIterator for &'a Unit<K, V> {
    type IntoIter = iter::Once<(&'a K, &'a V)>;
    type Item = (&'a K, &'a V);

    fn into_iter(self) -> Self::IntoIter {
        iter::once((&self.key, &self.value))
    }
}

impl<'a, K, V> IntoIterator for &'a mut Unit<K, V> {
    type IntoIter = iter::Once<(&'a K, &'a mut V)>;
    type Item = (&'a K, &'a mut V);

    fn into_iter(self) -> Self::IntoIter {
        iter::once((&self.key, &mut self.value))
    }
}

impl<K, V> Clone for Unit<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        Unit {
            key: self.key.clone(),
            value: self.value.clone(),
        }
    }
}

impl<K, V> Inner<K, V> for Unit<K, V> {
    type Occupied<'a>
        = UnitOccupied<'a, K, V>
    where
        Self: 'a;
    type Vacant<'a>
        = UnitVacant<'a, K, V>
    where
        Self: 'a;
    type Iter<'a>
        = iter::Once<(&'a K, &'a V)>
    where
        K: 'a,
        V: 'a;
    type IterMut<'a>
        = iter::Once<(&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a;

    fn new(key: K, value: V) -> Self {
        Self { key, value }
    }

    fn get<Q: Eq + Hash>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        if key == self.key.borrow() {
            Some(&self.value)
        } else {
            None
        }
    }

    fn get_mut<Q: Eq + Hash>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        if key == self.key.borrow() {
            Some(&mut self.value)
        } else {
            None
        }
    }

    fn values_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut V>
    where
        V: 'a,
    {
        iter::once(&mut self.value)
    }

    fn entry<'a>(&'a mut self, key: K) -> Entry<Self::Occupied<'a>, Self::Vacant<'a>>
    where
        K: PartialEq,
    {
        if key == self.key {
            Entry::Occupied(UnitOccupied {
                our_key: key,
                key: &mut self.key,
                value: &mut self.value,
            })
        } else {
            Entry::Vacant(UnitVacant {
                key,
                phantom: PhantomData,
            })
        }
    }

    fn should_remove(&self) -> bool {
        true
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        Some(mem::replace(&mut self.value, value))
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Chunk,
    {
        // if key == self.key.borrow() {
        //     Some(self.value)
        // } else {
        //     None
        // }
        unreachable!()
    }

    fn iter(&self) -> Self::Iter<'_> {
        iter::once((&self.key, &self.value))
    }

    fn len(&self) -> usize {
        1
    }
}
