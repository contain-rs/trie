var searchIndex = {};
searchIndex['trie'] = {"items":[[0,"","trie","An ordered map and set based on a trie."],[0,"map","","An ordered map based on a trie."],[3,"TrieMap","trie::map","A map implemented as a radix trie."],[3,"OccupiedEntry","","A view into an occupied entry in a TrieMap."],[3,"VacantEntry","","A view into a vacant entry in a TrieMap."],[3,"Iter","","A forward iterator over a map."],[3,"IterMut","","A forward iterator over the key-value pairs of a map, with the\nvalues being mutable."],[4,"Entry","","A view into a single entry in a TrieMap, which may be vacant or occupied."],[13,"Occupied","","An occupied entry.",0],[13,"Vacant","","A vacant entry.",0],[6,"Keys","","A forward iterator over the keys of a map."],[6,"Values","","A forward iterator over the values of a map."],[11,"clone","","",1],[11,"eq","","",1],[11,"partial_cmp","","",1],[11,"cmp","","",1],[11,"fmt","","",1],[11,"default","","",1],[11,"new","","Creates an empty `TrieMap`.",1],[11,"each_reverse","","Visits all key-value pairs in reverse order. Aborts traversal when `f` returns `false`.\nReturns `true` if `f` returns `true` for all elements.",1],[11,"keys","","Gets an iterator visiting all keys in ascending order by the keys.\nThe iterator's element type is `usize`.",1],[11,"values","","Gets an iterator visiting all values in ascending order by the keys.\nThe iterator's element type is `&'r T`.",1],[11,"iter","","Gets an iterator over the key-value pairs in the map, ordered by keys.",1],[11,"iter_mut","","Gets an iterator over the key-value pairs in the map, with the\nability to mutate the values.",1],[11,"len","","Return the number of elements in the map.",1],[11,"is_empty","","Return true if the map contains no elements.",1],[11,"clear","","Clears the map, removing all values.",1],[11,"get","","Returns a reference to the value corresponding to the key.",1],[11,"contains_key","","Returns true if the map contains a value for the specified key.",1],[11,"get_mut","","Returns a mutable reference to the value corresponding to the key.",1],[11,"insert","","Inserts a key-value pair from the map. If the key already had a value\npresent in the map, that value is returned. Otherwise, `None` is returned.",1],[11,"remove","","Removes a key from the map, returning the value at the key if the key\nwas previously in the map.",1],[11,"lower_bound","","Gets an iterator pointing to the first key-value pair whose key is not less than `key`.\nIf all keys in the map are less than `key` an empty iterator is returned.",1],[11,"upper_bound","","Gets an iterator pointing to the first key-value pair whose key is greater than `key`.\nIf all keys in the map are not greater than `key` an empty iterator is returned.",1],[11,"lower_bound_mut","","Gets an iterator pointing to the first key-value pair whose key is not less than `key`.\nIf all keys in the map are less than `key` an empty iterator is returned.",1],[11,"upper_bound_mut","","Gets an iterator pointing to the first key-value pair whose key is greater than `key`.\nIf all keys in the map are not greater than `key` an empty iterator is returned.",1],[11,"from_iter","","",1],[11,"extend","","",1],[11,"hash","","",1],[6,"Output","",""],[11,"index","","",1],[11,"index_mut","","",1],[11,"get","","Returns a mutable reference to the value if occupied, or the `VacantEntry` if\nvacant.",0],[11,"entry","","Gets the given key's corresponding entry in the map for in-place manipulation.",1],[11,"get","","Gets a reference to the value in the entry.",2],[11,"get_mut","","Gets a mutable reference to the value in the entry.",2],[11,"into_mut","","Converts the OccupiedEntry into a mutable reference to the value in the entry,\nwith a lifetime bound to the map itself.",2],[11,"insert","","Sets the value of the entry, and returns the entry's old value.",2],[11,"remove","","Takes the value out of the entry, and returns it.",2],[11,"insert","","Set the vacant entry to the given value.",3],[6,"Item","",""],[11,"next","","",4],[11,"size_hint","","",4],[6,"Item","",""],[11,"next","","",5],[11,"size_hint","","",5],[6,"Item","",""],[6,"IntoIter","",""],[6,"Item","",""],[6,"IntoIter","",""],[0,"set","trie","An ordered set based on a trie."],[3,"TrieSet","trie::set","A set implemented as a radix trie."],[3,"Iter","","A forward iterator over a set."],[3,"Difference","","An iterator producing elements in the set difference (in-order)."],[3,"SymmetricDifference","","An iterator producing elements in the set symmetric difference (in-order)."],[3,"Intersection","","An iterator producing elements in the set intersection (in-order)."],[3,"Union","","An iterator producing elements in the set union (in-order)."],[11,"cmp","","",6],[11,"partial_cmp","","",6],[11,"lt","","",6],[11,"le","","",6],[11,"gt","","",6],[11,"ge","","",6],[11,"eq","","",6],[11,"ne","","",6],[11,"hash","","",6],[11,"default","","",6],[11,"clone","","",6],[11,"fmt","","",6],[11,"new","","Creates an empty TrieSet.",6],[11,"each_reverse","","Visits all values in reverse order. Aborts traversal when `f` returns `false`.\nReturns `true` if `f` returns `true` for all elements.",6],[11,"iter","","Gets an iterator over the values in the set, in sorted order.",6],[11,"lower_bound","","Gets an iterator pointing to the first value that is not less than `val`.\nIf all values in the set are less than `val` an empty iterator is returned.",6],[11,"upper_bound","","Gets an iterator pointing to the first value that key is greater than `val`.\nIf all values in the set are less than or equal to `val` an empty iterator is returned.",6],[11,"difference","","Visits the values representing the difference, in ascending order.",6],[11,"symmetric_difference","","Visits the values representing the symmetric difference, in ascending order.",6],[11,"intersection","","Visits the values representing the intersection, in ascending order.",6],[11,"union","","Visits the values representing the union, in ascending order.",6],[11,"len","","Return the number of elements in the set",6],[11,"is_empty","","Returns true if the set contains no elements",6],[11,"clear","","Clears the set, removing all values.",6],[11,"contains","","Returns `true` if the set contains a value.",6],[11,"is_disjoint","","Returns `true` if the set has no elements in common with `other`.\nThis is equivalent to checking for an empty intersection.",6],[11,"is_subset","","Returns `true` if the set is a subset of another.",6],[11,"is_superset","","Returns `true` if the set is a superset of another.",6],[11,"insert","","Adds a value to the set. Returns `true` if the value was not already\npresent in the set.",6],[11,"remove","","Removes a value from the set. Returns `true` if the value was\npresent in the set.",6],[11,"from_iter","","",6],[11,"extend","","",6],[6,"Output","",""],[6,"Output","",""],[6,"Output","",""],[6,"Output","",""],[6,"Item","",""],[11,"next","","",7],[11,"size_hint","","",7],[6,"Item","",""],[11,"next","","",8],[6,"Item","",""],[11,"next","","",9],[6,"Item","",""],[11,"next","","",10],[6,"Item","",""],[11,"next","","",11],[6,"Item","",""],[6,"IntoIter","",""]],"paths":[[4,"Entry"],[3,"TrieMap"],[3,"OccupiedEntry"],[3,"VacantEntry"],[3,"Iter"],[3,"IterMut"],[3,"TrieSet"],[3,"Iter"],[3,"Difference"],[3,"SymmetricDifference"],[3,"Intersection"],[3,"Union"]]};
initSearch(searchIndex);
