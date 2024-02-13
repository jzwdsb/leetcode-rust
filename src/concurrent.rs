//! Concurrent programming
#![allow(dead_code)]

use std::sync::{Condvar, Mutex};

struct PrintInOrder {
    order: Mutex<u32>,
    cond: Condvar,
}

impl PrintInOrder {
    pub fn new() -> Self {
        Self {
            order: Mutex::new(1),
            cond: Condvar::new(),
        }
    }

    pub fn first(&self, f: Box<dyn Fn()>) {
        let mut order = self.order.lock().unwrap();

        // do first
        f();

        // notify second
        *order = 2;
        self.cond.notify_all();
    }

    // wait first
    pub fn second(&self, f: Box<dyn Fn()>) {
        // wait first
        let mut order = self.order.lock().unwrap();
        while *order != 2 {
            order = self.cond.wait(order).unwrap();
        }

        // do second
        f();

        *order = 3;
        self.cond.notify_all();
    }

    // wait second
    pub fn thrid(&self, f: Box<dyn Fn()>) {
        // wait second
        let mut order = self.order.lock().unwrap();
        while *order != 3 {
            order = self.cond.wait(order).unwrap();
        }

        // do thrid
        f();
    }
} // impl Foo

#[cfg(test)]
mod test {
    use super::*;
    use std::{
        sync::{Arc, Mutex},
        thread::spawn,
    };

    #[test]
    fn test_print_in_order() {
        // retried 10 times to make sure the result is consistent
        for _ in 0..10 {
            let result = Arc::new(Mutex::new(String::new()));
            let res_a = result.clone();
            let res_b = result.clone();
            let res_c = result.clone();
            let first = Box::new(move || res_a.lock().unwrap().push_str("first"));
            let second = Box::new(move || res_b.lock().unwrap().push_str("second"));
            let thrid = Box::new(move || res_c.lock().unwrap().push_str("thrid"));
            let foo = Arc::new(PrintInOrder::new());
            let one_obj = foo.clone();
            let two_obj = foo.clone();
            let three_obj = foo.clone();
            // dead locks
            let one = spawn(move || one_obj.first(first));
            let two = spawn(move || two_obj.second(second));
            let three = spawn(move || three_obj.thrid(thrid));
            one.join().unwrap();
            two.join().unwrap();
            three.join().unwrap();
            assert_eq!(result.lock().unwrap().as_str(), "firstsecondthrid");
        }
    }
}
