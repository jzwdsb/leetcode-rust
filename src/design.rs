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
    free_memory: std::collections::BTreeMap<i32, i32>, //
    allocated: std::collections::HashMap<i32, Vec<(i32, i32)>>,
}

impl Allocator {
    fn new(size: usize) -> Self {
        Self {
            free_memory: std::collections::BTreeMap::from([(0, size as i32)]),
            allocated: std::collections::HashMap::new(),
        }
    }

    pub fn allocate(&mut self, size: usize, m_id: i32) -> Option<usize> {
        if let Some((&index, &ptr_size)) = self.free_memory.iter().find(|(k, &v)| v >= size as i32)
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
}
