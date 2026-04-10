#[generic_tests::define]
mod instantiate {
    use std::hint::black_box;

    #[cfg(feature = "extra")]
    use crate::Entry::*;
    use crate::Map;
    use crate::inner::Inner;
    use crate::map_trait::{BTrieHint, BasicTrieHint, Children, HatTrieHint, MapTrait, MaybeValue, NullCount};
    use crate::node::TrieNode::{self, *};
    use crate::root_node::{AnyNodeRef, RootNode};

    /// check_integrity now accepts a TrieNode instead of an InternalNode,
    /// because the root is a TrieNode enum.
    fn check_integrity<M: MapTrait>(node: &M::Root) {
        match node.into_any_ref() {
            AnyNodeRef::Branch { children, value, count, .. } => check_integrity_internal(count, value, children),
            // A bare External or Nothing root is valid (0 or 1 elements).
            AnyNodeRef::External(..) | AnyNodeRef::Nothing => {}
        }
    }

    fn check_integrity_internal<M: MapTrait>(
        count: M::Count,
        value: Option<&M::Value>,
        children: &[TrieNode<M>],
    ) {
        assert!(count != 0);

        let mut sum: M::Count = Default::default();
        for x in children.iter() {
            match *x {
                Nothing => (),
                Internal(ref y) => {
                    check_integrity_internal(y.count, y.value.as_ref(), y.children.as_slice());
                    sum += 1;
                }
                External(ref maybe_inner) => sum += maybe_inner.len(),
            }
        }
        assert_eq!(sum, count);
    }

    #[test]
    fn test_insert_from_bench<M: MapTrait<Key = u32, Value = u32>>() {
        let mut m: Map<M> = Map::new();
        m.insert(0xad6b27f7, 0);
        m.insert(0xad6f27f7, 1);
        black_box(&m);
    }

    #[test]
    fn test_find_mut<M: MapTrait<Key = u32, Value = u32>>() where M::Value: PartialEq {
        let mut m: Map<M> = Map::new();
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
    fn test_find_mut_missing<M: MapTrait<Key = u32, Value = u32>>() {
        let mut m: Map<M> = Map::new();
        assert!(m.get_mut(&0).is_none());
        assert!(m.insert(1, 12).is_none());
        assert!(m.get_mut(&0).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.get_mut(&0).is_none());
    }

    #[test]
    fn test_step<M: MapTrait<Key = u32, Value = u32>>() {
        let mut trie: Map<M> = Map::new();
        let n = 300;

        for x in (1..n).step_by(2) {
            assert!(trie.insert(x, x + 1).is_none(), "{}", x);
            assert!(trie.contains_key(&x), "{}", x);
            check_integrity::<M>(&trie.root);
        }

        for x in (0..n).step_by(2) {
            assert!(!trie.contains_key(&x));
            assert!(trie.insert(x, x + 1).is_none());
            check_integrity::<M>(&trie.root);
        }

        for x in 0..n {
            assert!(trie.contains_key(&x));
            assert!(trie.insert(x, x + 1).is_some());
            check_integrity::<M>(&trie.root);
        }

        for x in (1..n).step_by(2) {
            assert!(trie.remove(&x).is_some());
            assert!(!trie.contains_key(&x));
            check_integrity::<M>(&trie.root);
        }

        for x in (0..n).step_by(2) {
            assert!(trie.contains_key(&x));
            assert!(trie.insert(x, x + 1).is_some());
            check_integrity::<M>(&trie.root);
        }
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_each_reverse<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_each_reverse_break<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_insert<M: MapTrait<Key = u32, Value = u32>>() {
        let mut m: Map<M> = Map::new();
        assert_eq!(m.insert(1, 2), None);
        assert_eq!(m.insert(1, 3), Some(2));
        assert_eq!(m.insert(1, 4), Some(3));
    }

    #[test]
    fn test_remove<M: MapTrait<Key = u32, Value = u32>>() {
        let mut m: Map<M> = Map::new();
        m.insert(1, 2);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_from_iter<M: MapTrait<Key = u32, Value = u32>>() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: TrieMap<usize, i32> = xs.iter().cloned().collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_keys<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_values<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_iteration<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_mut_iter<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_bound<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_mut_bound<M: MapTrait<Key = u32, Value = u32>>() {
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

        assert!(
            m_lower
                .lower_bound_mut(Bound::Excluded(&199))
                .next()
                .is_none()
        );
        assert!(
            m_upper
                .upper_bound_mut(Bound::Excluded(&198))
                .next()
                .is_none()
        );

        assert!(m_lower.iter().all(|(_, &x)| x == 0));
        assert!(m_upper.iter().all(|(_, &x)| x == 0));
    }

    #[test]
    fn test_clone<M: MapTrait<Key = u32, Value = u32>>() {
        let mut a: Map<M> = Map::new();

        a.insert(1, 123);
        a.insert(2, 345);
        a.insert(3, 567);

        assert!(a.clone() == a);
    }

    #[test]
    fn test_eq<M: MapTrait<Key = u32, Value = u32>>() {
        let mut a: Map<M> = Map::new();
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
    fn test_lt<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_ord<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_hash<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_debug<M: MapTrait<Key = u32, Value = u32>>() {
        let mut map = Map::new();
        let empty: Map<usize, char> = Map::new();

        map.insert(1, 'a');
        map.insert(2, 'b');

        assert_eq!(format!("{:?}", map), "{1: 'a', 2: 'b'}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_index<M: MapTrait<Key = u32, Value = u32>>() {
        let mut map = Map::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[&2], 1);
    }

    #[cfg(feature = "extra")]
    #[test]
    #[should_panic]
    fn test_index_nonexistent<M: MapTrait<Key = u32, Value = u32>>() {
        let mut map: Map<M> = Map::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        black_box(map[&4]);
    }

    const SQUARES_UPPER_LIM: usize = 128;

    #[cfg(feature = "extra")]
    fn squares_map<M: MapTrait>() -> Map<M> {
        let mut map = Map::new();
        for i in 0..SQUARES_UPPER_LIM {
            map.insert(i, i * i);
        }
        map
    }

    #[cfg(feature = "extra")]
    #[test]
    fn test_entry_get<M: MapTrait<Key = u32, Value = u32>>() {
        let mut map: Map<M> = squares_map::<M>();

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
    fn test_entry_get_mut<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_entry_into_mut<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_entry_take<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_occupied_entry_set<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_vacant_entry_set<M: MapTrait<Key = u32, Value = u32>>() {
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
    fn test_single_key<M: MapTrait<Key = u32, Value = u32>>() {
        let mut map = Map::new();
        map.insert(1, 2);

        if let Occupied(e) = map.entry(1) {
            e.remove();
        }
    }

    #[instantiate_tests(<BasicTrieHint<u32, u32, NullCount>>)]
    mod with_null_count {}

    #[instantiate_tests(<BasicTrieHint<u32, u32, usize>>)]
    mod with_usize_count {}

    #[instantiate_tests(<HatTrieHint<256, u32, u32, usize>>)]
    mod with_hat_trie {}

    #[instantiate_tests(<HatTrieHint<32, u32, u32, usize>>)]
    mod with_small_hat_trie {}

    #[instantiate_tests(<BTrieHint<256, u32, u32, usize>>)]
    mod with_btree {}

    #[instantiate_tests(<BTrieHint<32, u32, u32, usize>>)]
    mod with_small_btree {}
}
