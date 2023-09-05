// TODO: refactor the list node to use Rc<RefCell<ListNode<T>>> to avoid the ownership problem
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode<T> {
    pub val: T,
    pub next: Option<Box<ListNode<T>>>,
}

impl<T> ListNode<T> {
    #[inline]
    #[allow(dead_code)]
    fn new(val: T) -> Self {
        ListNode {
            next: None,
            val: val,
        }
    }
    #[inline]
    #[allow(dead_code)]
    fn from_vec(v: Vec<T>) -> Option<Box<ListNode<T>>> {
        let mut head = None;
        for i in v.into_iter().rev() {
            let mut node = ListNode::new(i);
            node.next = head;
            head = Some(Box::new(node));
        }
        head
    }
}

impl<T> Iterator for ListNode<T> {
    type Item = T;
    fn next<'a>(&'a mut self) -> Option<Self::Item> {
        let node = self;
        let mut res = None;
        std::mem::swap(&mut node.next, &mut res);
        res.map(|n| {
            node.next = n.next;
            n.val
        })
    }
}

pub struct ListSolution {}

impl ListSolution {
    /*
    leetcode link: https://leetcode.com/problems/remove-nth-node-from-end-of-list/
    we can use two pointers to solve this problem in one pass
     */
    pub fn remove_nth_from_end(
        head: Option<Box<ListNode<i32>>>,
        n: i32,
    ) -> Option<Box<ListNode<i32>>> {
        // calculate the length of the list by using std::iter::successors, time complexity O(n)
        let cnt = std::iter::successors(head.as_ref(), |last| last.next.as_ref()).count();
        // dummy node to handle the case when we need to remove the first node
        let mut dummy = Some(Box::new(ListNode { val: 0, next: head }));
        // find the previous node of the node we want to remove
        let mut prev =
            (0..cnt - n as usize).fold(dummy.as_mut(), |curr, _| curr.unwrap().next.as_mut());
        // remove the node
        prev.unwrap().next = prev.as_mut().unwrap().next.as_mut().unwrap().next.take();
        dummy.unwrap().next
    }

    /*
    link: https://leetcode.com/problems/add-two-numbers/
    solve this problem by recursion
    new node.Val = (l1.Val + l2.Val) % 10
    new node.Next = add_two_numbers(l1.Next, l2.Next) if l1.Val + l2.Val < 10
                    add_two_numbers(add_two_numbers(l1.Next, 1), l2.Next) if l1.Val + l2.Val >= 10
     */
    pub fn add_two_numbers(
        l1: Option<Box<ListNode<i32>>>,
        l2: Option<Box<ListNode<i32>>>,
    ) -> Option<Box<ListNode<i32>>> {
        match (l1, l2) {
            (None, None) => None,
            (Some(l1), None) => Some(l1),
            (None, Some(l2)) => Some(l2),
            (Some(l1), Some(l2)) => {
                let val = l1.val + l2.val;
                if val < 10 {
                    let mut node = ListNode::new(val);
                    node.next = Self::add_two_numbers(l1.next, l2.next);
                    Some(Box::new(node))
                } else {
                    let mut node = ListNode::new(val - 10);
                    node.next = Self::add_two_numbers(
                        Self::add_two_numbers(l1.next, Some(Box::new(ListNode::new(1)))),
                        l2.next,
                    );
                    Some(Box::new(node))
                }
            }
        }
    }

    /* link: https://leetcode.com/problems/add-two-numbers-ii/
    we can use stack by storing the value in reverse order
    and pop the value from the stack to construct the list
     */

    pub fn add_two_numbers_ii(
        l1: Option<Box<ListNode<i32>>>,
        l2: Option<Box<ListNode<i32>>>,
    ) -> Option<Box<ListNode<i32>>> {
        let mut stack1 = Vec::new();
        let mut stack2 = Vec::new();
        let mut res = None;
        let mut carry = 0;
        let mut l1 = l1;
        let mut l2 = l2;
        while l1.is_some() {
            stack1.push(l1.as_ref().unwrap().val);
            l1 = l1.unwrap().next;
        }
        while l2.is_some() {
            stack2.push(l2.as_ref().unwrap().val);
            l2 = l2.unwrap().next;
        }
        while !stack1.is_empty() || !stack2.is_empty() || carry != 0 {
            let mut val = carry;
            if !stack1.is_empty() {
                val += stack1.pop().unwrap();
            }
            if !stack2.is_empty() {
                val += stack2.pop().unwrap();
            }
            carry = val / 10;
            let mut node = ListNode::new(val % 10);
            node.next = res;
            res = Some(Box::new(node));
        }
        res
    }

    pub fn swap_pairs(head: Option<Box<ListNode<i32>>>) -> Option<Box<ListNode<i32>>> {
        let mut dummy = Some(Box::new(ListNode { val: 0, next: head }));
        let mut prev = dummy.as_mut();
        while prev.as_ref().unwrap().next.is_some()
            && prev.as_ref().unwrap().next.as_ref().unwrap().next.is_some()
        {
            let mut first = prev.as_mut().unwrap().next.take();
            let mut second = first.as_mut().unwrap().next.take();
            first.as_mut().unwrap().next = second.as_mut().unwrap().next.take();
            second.as_mut().unwrap().next = first;
            prev.as_mut().unwrap().next = second;
            prev = prev.unwrap().next.as_mut().unwrap().next.as_mut();
        }
        dummy.unwrap().next
    }

    pub fn merge_two_lists(
        list1: Option<Box<ListNode<i32>>>,
        list2: Option<Box<ListNode<i32>>>,
    ) -> Option<Box<ListNode<i32>>> {
        match (list1, list2) {
            (None, None) => None,
            (Some(list1), None) => Some(list1),
            (None, Some(list2)) => Some(list2),
            (Some(mut list1), Some(mut list2)) => {
                if list1.val < list2.val {
                    list1.next = Self::merge_two_lists(list1.next, Some(list2));
                    Some(list1)
                } else {
                    list2.next = Self::merge_two_lists(Some(list1), list2.next);
                    Some(list2)
                }
            }
        }
    }

    /*
    link: https://leetcode.com/problems/reverse-nodes-in-k-group/
     */
    pub fn reverse_k_group(head: Option<Box<ListNode<i32>>>, k: i32) -> Option<Box<ListNode<i32>>> {
        if k == 0 {
            return head;
        }
        let mut head = head;
        let mut node = &mut head;
        // check if there are k nodes left
        // find the next group's head
        for _ in 0..k {
            if let Some(n) = node {
                node = &mut n.next;
            } else {
                return head;
            }
        }
        // reserve the next group
        let mut ret = Self::reverse_k_group(node.take(), k);
        // append the current list to the head of the reversed next group
        // ret will be the new head of the current group
        while let Some(h) = head.take() {
            ret = Some(Box::new(ListNode {
                val: h.val,
                next: ret,
            }));
            head = h.next;
        }
        ret
    }

    pub fn rotate_right(head: Option<Box<ListNode<i32>>>, k: i32) -> Option<Box<ListNode<i32>>> {
        if k == 0 {
            return head;
        }
        if let None = head {
            return head;
        }

        // count the length of the list
        let mut len = 0;
        {
            let mut node = head.as_ref();
            while let Some(n) = node {
                len += 1;
                node = n.next.as_ref();
            }
        }

        if len == 0 {
            return head;
        }

        let k = k % len;
        if k == 0 {
            return head;
        }

        let mut head = head;
        let mut node = head.as_deref_mut().unwrap();

        // find the node before the kth node from the end
        for _ in 0..len - k - 1 {
            node = node.next.as_deref_mut().unwrap();
        }
        // take the kth node from the end
        let mut new_head = node.next.take().unwrap();

        // find the end the of list and link it to the head
        node = new_head.as_mut();
        while node.next.is_some() {
            node = node.next.as_mut().unwrap();
        }
        node.next = head;

        Some(new_head)
    }

    pub fn reverse_list(head: Option<Box<ListNode<i32>>>) -> Option<Box<ListNode<i32>>> {
        let mut prev = None;
        let mut node = head;
        while let Some(mut n) = node {
            node = n.next.take();
            n.next = prev;
            prev = Some(n);
        }
        prev
    }

    pub fn has_cycle(l1: Option<Box<ListNode<i32>>>) -> bool {
        let mut slow = l1.as_ref();
        let mut fast = l1.as_ref();
        while let Some(f) = fast {
            fast = f.next.as_ref();
            if let Some(f) = fast {
                fast = f.next.as_ref();
            } else {
                return false;
            }
            slow = slow.unwrap().next.as_ref();
            if slow == fast {
                return true;
            }
        }
        false
    }
    pub fn delete_duplicates(head: Option<Box<ListNode<i32>>>) -> Option<Box<ListNode<i32>>> {
        let mut dummy = Some(Box::new(ListNode { val: 0, next: head }));
        let mut prev = dummy.as_mut();
        while prev.as_ref().unwrap().next.is_some() {
            let mut curr = prev.as_mut().unwrap().next.take();
            while curr.as_ref().unwrap().next.is_some()
                && curr.as_ref().unwrap().val == curr.as_ref().unwrap().next.as_ref().unwrap().val
            {
                curr = curr.unwrap().next;
            }
            prev.as_mut().unwrap().next = curr.take();
            prev = prev.unwrap().next.as_mut();
        }
        dummy.unwrap().next
    }
}

pub fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_nth_from_end() {
        let head = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        let ans = ListNode::from_vec(vec![1, 2, 3, 5]);
        assert_eq!(ListSolution::remove_nth_from_end(head, 2), ans);
        let head = ListNode::from_vec(vec![1]);
        let ans = ListNode::from_vec(vec![]);
        assert_eq!(ListSolution::remove_nth_from_end(head, 1), ans);
        let head = ListNode::from_vec(vec![1, 2]);
        let ans = ListNode::from_vec(vec![2]);
        assert_eq!(ListSolution::remove_nth_from_end(head, 2), ans);
    }

    #[test]
    fn test_add_two_numbers() {
        let l1 = ListNode::from_vec(vec![2, 4, 3]);
        let l2 = ListNode::from_vec(vec![5, 6, 4]);
        assert_eq!(
            ListSolution::add_two_numbers(l1, l2),
            ListNode::from_vec(vec![7, 0, 8])
        );

        let l1 = ListNode::from_vec(vec![0]);
        let l2 = ListNode::from_vec(vec![0]);
        assert_eq!(
            ListSolution::add_two_numbers(l1, l2),
            ListNode::from_vec(vec![0])
        );

        let l1 = ListNode::from_vec(vec![9, 9, 9, 9, 9, 9, 9]);
        let l2 = ListNode::from_vec(vec![9, 9, 9, 9]);
        assert_eq!(
            ListSolution::add_two_numbers(l1, l2),
            ListNode::from_vec(vec![8, 9, 9, 9, 0, 0, 0, 1])
        );
    }

    #[test]
    fn test_add_two_numbers_ii() {
        let l1 = ListNode::from_vec(vec![7, 2, 4, 3]);
        let l2 = ListNode::from_vec(vec![5, 6, 4]);
        assert_eq!(
            ListSolution::add_two_numbers_ii(l1, l2),
            ListNode::from_vec(vec![7, 8, 0, 7])
        );

        let l1 = ListNode::from_vec(vec![0]);
        let l2 = ListNode::from_vec(vec![0]);
        assert_eq!(
            ListSolution::add_two_numbers_ii(l1, l2),
            ListNode::from_vec(vec![0])
        );

        let l1 = ListNode::from_vec(vec![9, 9, 9, 9, 9, 9, 9]);
        let l2 = ListNode::from_vec(vec![9, 9, 9, 9]);
        assert_eq!(
            ListSolution::add_two_numbers_ii(l1, l2),
            ListNode::from_vec(vec![1, 0, 0, 0, 9, 9, 9, 8])
        );
    }

    #[test]
    fn test_swap_pairs() {
        let head = ListNode::from_vec(vec![1, 2, 3, 4]);
        let ans = ListNode::from_vec(vec![2, 1, 4, 3]);
        assert_eq!(ListSolution::swap_pairs(head), ans);
        let head = ListNode::from_vec(vec![]);
        let ans = ListNode::from_vec(vec![]);
        assert_eq!(ListSolution::swap_pairs(head), ans);
        let head = ListNode::from_vec(vec![1]);
        let ans = ListNode::from_vec(vec![1]);
        assert_eq!(ListSolution::swap_pairs(head), ans);
        let head = ListNode::from_vec(vec![1, 2, 3]);
        let ans = ListNode::from_vec(vec![2, 1, 3]);
        assert_eq!(ListSolution::swap_pairs(head), ans);
    }

    #[test]
    fn test_merge_two_lists() {
        let l1 = ListNode::from_vec(vec![1, 2, 4]);
        let l2 = ListNode::from_vec(vec![1, 3, 4]);
        let ans = ListNode::from_vec(vec![1, 1, 2, 3, 4, 4]);
        assert_eq!(ListSolution::merge_two_lists(l1, l2), ans);
        let l1 = ListNode::from_vec(vec![]);
        let l2 = ListNode::from_vec(vec![]);
        let ans = ListNode::from_vec(vec![]);
        assert_eq!(ListSolution::merge_two_lists(l1, l2), ans);
        let l1 = ListNode::from_vec(vec![]);
        let l2 = ListNode::from_vec(vec![0]);
        let ans = ListNode::from_vec(vec![0]);
        assert_eq!(ListSolution::merge_two_lists(l1, l2), ans);
    }

    #[test]
    fn test_reverse_k_group() {
        let head = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        let ans = ListNode::from_vec(vec![2, 1, 4, 3, 5]);
        assert_eq!(ListSolution::reverse_k_group(head, 2), ans);
        let head = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        let ans = ListNode::from_vec(vec![3, 2, 1, 4, 5]);
        assert_eq!(ListSolution::reverse_k_group(head, 3), ans);
        let head = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        let ans = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(ListSolution::reverse_k_group(head, 1), ans);
        let head = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        let ans = ListNode::from_vec(vec![5, 4, 3, 2, 1]);
        assert_eq!(ListSolution::reverse_k_group(head, 5), ans);
        let head = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        let ans = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(ListSolution::reverse_k_group(head, 6), ans);
        let head = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        let ans = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(ListSolution::reverse_k_group(head, 0), ans);
        let head = ListNode::from_vec(vec![1]);
        let ans = ListNode::from_vec(vec![1]);
        assert_eq!(ListSolution::reverse_k_group(head, 1), ans);
        let head = ListNode::from_vec(vec![1, 2]);
        let ans = ListNode::from_vec(vec![2, 1]);
        assert_eq!(ListSolution::reverse_k_group(head, 2), ans);
    }

    #[test]
    fn test_rotate_right() {
        let head = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        let ans = ListNode::from_vec(vec![4, 5, 1, 2, 3]);
        assert_eq!(ListSolution::rotate_right(head, 2), ans);
        let head = ListNode::from_vec(vec![0, 1, 2]);
        let ans = ListNode::from_vec(vec![2, 0, 1]);
        assert_eq!(ListSolution::rotate_right(head, 4), ans);
        let head = ListNode::from_vec(vec![1, 2]);
        let ans = ListNode::from_vec(vec![2, 1]);
        assert_eq!(ListSolution::rotate_right(head, 1), ans);
        let head = ListNode::from_vec(vec![1, 2]);
        let ans = ListNode::from_vec(vec![1, 2]);
        assert_eq!(ListSolution::rotate_right(head, 0), ans);
        let head = ListNode::from_vec(vec![1, 2]);
        let ans = ListNode::from_vec(vec![1, 2]);
        assert_eq!(ListSolution::rotate_right(head, 2), ans);
        let head = ListNode::from_vec(vec![1]);
        let ans = ListNode::from_vec(vec![1]);
        assert_eq!(ListSolution::rotate_right(head, 1), ans);
        let head = ListNode::from_vec(vec![]);
        let ans = ListNode::from_vec(vec![]);
        assert_eq!(ListSolution::rotate_right(head, 1), ans);
    }

    #[test]
    fn test_reverse_list() {
        let head = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        let ans = ListNode::from_vec(vec![5, 4, 3, 2, 1]);
        assert_eq!(ListSolution::reverse_list(head), ans);
        let head = ListNode::from_vec(vec![1, 2]);
        let ans = ListNode::from_vec(vec![2, 1]);
        assert_eq!(ListSolution::reverse_list(head), ans);
        let head = ListNode::from_vec(vec![1]);
        let ans = ListNode::from_vec(vec![1]);
        assert_eq!(ListSolution::reverse_list(head), ans);
        let head = ListNode::from_vec(vec![]);
        let ans = ListNode::from_vec(vec![]);
        assert_eq!(ListSolution::reverse_list(head), ans);
    }

    #[test]
    fn test_delete_duplicates() {
        let head = ListNode::from_vec(vec![1, 2, 3, 3, 4, 4, 5]);
        let ans = ListNode::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(ListSolution::delete_duplicates(head), ans);
        let head = ListNode::from_vec(vec![1, 1, 1, 2, 3]);
        let ans = ListNode::from_vec(vec![1, 2, 3]);
        assert_eq!(ListSolution::delete_duplicates(head), ans);
        let head = ListNode::from_vec(vec![1, 1, 1, 1, 1]);
        let ans = ListNode::from_vec(vec![1]);
        assert_eq!(ListSolution::delete_duplicates(head), ans);
    }
}
