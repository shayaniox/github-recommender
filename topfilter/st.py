from typing import TypeVar, Generic, Optional, Iterator, List
from collections import OrderedDict
import bisect

K = TypeVar('K', bound=Comparable)
V = TypeVar('V')

class ST(Generic[K, V]):
    """
    A sorted symbol table implementation using Python's OrderedDict.
    Does not allow duplicate keys or None values.
    """
    
    def __init__(self):
        self._st = OrderedDict()
    
    def get(self, key: K) -> Optional[V]:
        """
        Returns the value associated with the given key.
        
        Args:
            key: The key to look up
            
        Returns:
            The associated value or None if key not found
        """
        if key is None:
            raise ValueError("called get() with None key")
        return self._st.get(key)
    
    def put(self, key: K, val: V) -> None:
        """
        Inserts the key-value pair into the symbol table.
        
        Args:
            key: The key
            val: The value (cannot be None)
        """
        if key is None:
            raise ValueError("called put() with None key")
        if val is None:
            self.delete(key)
        else:
            self._st[key] = val
    
    def delete(self, key: K) -> None:
        """
        Removes the key and its associated value from the symbol table.
        
        Args:
            key: The key to remove
        """
        if key is None:
            raise ValueError("called delete() with None key")
        self._st.pop(key, None)
    
    def contains(self, key: K) -> bool:
        """
        Checks if the symbol table contains the given key.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if key is None:
            raise ValueError("called contains() with None key")
        return key in self._st
    
    def size(self) -> int:
        """Returns the number of key-value pairs in the symbol table."""
        return len(self._st)
    
    def is_empty(self) -> bool:
        """Returns True if the symbol table is empty."""
        return self.size() == 0
    
    def keys(self) -> List[K]:
        """Returns all keys in the symbol table as a sorted list."""
        return sorted(self._st.keys())
    
    def __iter__(self) -> Iterator[K]:
        """Returns an iterator over the keys in sorted order."""
        return iter(self.keys())
    
    def min(self) -> K:
        """Returns the smallest key in the symbol table."""
        if self.is_empty():
            raise ValueError("called min() with empty symbol table")
        return min(self._st.keys())
    
    def max(self) -> K:
        """Returns the largest key in the symbol table."""
        if self.is_empty():
            raise ValueError("called max() with empty symbol table")
        return max(self._st.keys())
    
    def ceil(self, key: K) -> K:
        """
        Returns the smallest key in the symbol table greater than or equal to key.
        
        Args:
            key: The key to find ceiling for
            
        Returns:
            The ceiling key
        """
        if key is None:
            raise ValueError("called ceil() with None key")
        
        keys = self.keys()
        i = bisect.bisect_left(keys, key)
        if i == len(keys):
            raise ValueError("no ceiling key found")
        return keys[i]
    
    def floor(self, key: K) -> K:
        """
        Returns the largest key in the symbol table less than or equal to key.
        
        Args:
            key: The key to find floor for
            
        Returns:
            The floor key
        """
        if key is None:
            raise ValueError("called floor() with None key")
        
        keys = self.keys()
        i = bisect.bisect_right(keys, key) - 1
        if i < 0:
            raise ValueError("no floor key found")
        return keys[i]

    def __str__(self) -> str:
        """Returns string representation of the symbol table."""
        return "\n".join(f"{k}: {v}" for k, v in sorted(self._st.items()))


# Example usage
if __name__ == "__main__":
    st = ST[str, str]()
    
    # Insert some key-value pairs
    st.put("www.cs.princeton.edu", "128.112.136.11")
    st.put("www.cs.princeton.edu", "128.112.136.35")  # overwrite
    st.put("www.princeton.edu", "128.112.130.211")
    st.put("www.math.princeton.edu", "128.112.18.11")
    st.put("www.yale.edu", "130.132.51.8")
    
    print("Size:", st.size())
    print("Min:", st.min())
    print("Max:", st.max())
    print("Floor of www.simpson.com:", st.floor("www.simpson.com"))
    print("Ceiling of www.simpson.com:", st.ceil("www.simpson.com"))