#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};

struct MinStack {
    stack: Vec<i32>,
    min_stack: Vec<i32>,
}

impl MinStack {
    fn new() -> Self {
        MinStack {
            stack: Vec::new(),
            min_stack: Vec::new(),
        }
    }

    fn push(&mut self, x: i32) {
        self.stack.push(x);
        if self.min_stack.is_empty() || x <= *self.min_stack.last().unwrap() {
            self.min_stack.push(x);
        }
    }

    fn pop(&mut self) {
        let x = self.stack.pop().unwrap();
        if x == *self.min_stack.last().unwrap() {
            self.min_stack.pop();
        }
    }

    fn top(&self) -> i32 {
        *self.stack.last().unwrap()
    }

    fn get_min(&self) -> i32 {
        *self.min_stack.last().unwrap()
    }
}

struct Allocator {
    free_memory: BTreeMap<i32, i32>,
    allocated: HashMap<i32, Vec<(i32, i32)>>,
}

impl Allocator {
    fn new(size: usize) -> Self {
        Self {
            free_memory: BTreeMap::from([(0, size as i32)]),
            allocated: HashMap::new(),
        }
    }

    pub fn allocate(&mut self, size: usize, m_id: i32) -> Option<usize> {
        if let Some((&index, &ptr_size)) = self.free_memory.iter().find(|(_, &v)| v >= size as i32)
        {
            self.free_memory.remove(&index);
            let ptr = if ptr_size == size as i32 {
                (index, ptr_size)
            } else {
                self.free_memory
                    .insert(index + size as i32, ptr_size - size as i32);
                (index, size as i32)
            };
            self.allocated.entry(m_id).or_insert(Vec::new()).push(ptr);
            return Some(ptr.0 as usize);
        }
        None
    }

    fn deallocate(&mut self, (index, size): (i32, i32)) {
        let mut size = size;
        let mut index = index;
        let right_index = index + size;
        if self.free_memory.contains_key(&right_index) {
            let adjacent_ptr = self.free_memory.remove(&right_index).unwrap();
            size += adjacent_ptr;
        }
        if let Some((&left_index, &left_size)) = self.free_memory.range(0..index).rev().next() {
            if left_index + left_size == index {
                index = left_index;
                size += left_size;
            }
        }
        self.free_memory.insert(index, size);
    }

    fn free(&mut self, m_id: i32) -> i32 {
        if let Some(ptrs) = self.allocated.remove(&m_id) {
            let mut mem = 0;
            for ptr in ptrs.into_iter() {
                self.deallocate(ptr);
                mem += ptr.1;
            }
            return mem;
        }
        0
    }
}

struct DataStream {
    value: i32,
    k: i32,
    cnt: i32,
}

impl DataStream {
    pub fn new(value: i32, k: i32) -> Self {
        Self { value, k, cnt: 0 }
    }

    pub fn consec(&mut self, num: i32) -> bool {
        if num == self.value {
            self.cnt += 1;
        } else {
            self.cnt = 0;
        }
        self.cnt == self.k
    }
}

struct FrequencyTracker {
    nums: HashMap<i32, i32>,
    nums_by_freq: HashMap<i32, HashSet<i32>>,
}

impl FrequencyTracker {
    pub fn new() -> Self {
        Self {
            nums: HashMap::new(),
            nums_by_freq: HashMap::new(),
        }
    }

    pub fn add(&mut self, num: i32) {
        let freq = self.nums.entry(num).or_insert(0);
        if *freq > 0 {
            self.nums_by_freq.entry(*freq).or_default().remove(&num);
            if self.nums_by_freq[&freq].is_empty() {
                self.nums_by_freq.remove(&freq);
            }
        }
        *freq += 1;
        self.nums_by_freq.entry(*freq).or_default().insert(num);
    }

    pub fn delete_one(&mut self, num: i32) {
        if let Some(v) = self.nums.get_mut(&num) {
            self.nums_by_freq.entry(*v).or_default().remove(&num);
            if self.nums_by_freq[&v].is_empty() {
                self.nums_by_freq.remove(&v);
            }
            *v -= 1;
            if *v == 0 {
                self.nums.remove(&num);
            } else {
                self.nums_by_freq.entry(*v).or_default().insert(num);
            }
        }
    }

    pub fn has_frequency(&self, freq: i32) -> bool {
        self.nums_by_freq.contains_key(&freq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_stack() {
        let mut obj = MinStack::new();
        obj.push(-2);
        obj.push(0);
        obj.push(-3);
        assert_eq!(obj.get_min(), -3);
        obj.pop();
        assert_eq!(obj.top(), 0);
        assert_eq!(obj.get_min(), -2);
    }

    #[test]
    fn test_data_stream() {
        let mut obj = DataStream::new(1, 3);
        assert!(!obj.consec(1));
        assert!(!obj.consec(1));
        assert!(obj.consec(1));
        assert!(!obj.consec(2));
        assert!(!obj.consec(2));
        assert!(!obj.consec(3));
        assert!(!obj.consec(3));
        assert!(!obj.consec(3));
        assert!(!obj.consec(1));
        assert!(!obj.consec(1));
        assert!(obj.consec(1));
    }

    #[test]
    fn test_frequency_tracker() {
        let mut obj = FrequencyTracker::new();
        obj.add(3);
        obj.add(3);
        assert!(obj.has_frequency(2));

        obj.add(7);
        assert!(obj.has_frequency(1));
        obj.delete_one(7);
        assert!(!obj.has_frequency(1));
    }
}
