#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    #[allow(dead_code)]
    fn new(val: i32) -> Self {
        ListNode {
            next: None,
            val: val,
        }
    }
    #[inline]
    #[allow(dead_code)]
    fn from_vec(v: Vec<i32>) -> Option<Box<ListNode>> {
        let mut head = None;
        for i in v.into_iter().rev() {
            let mut node = ListNode::new(i);
            node.next = head;
            head = Some(Box::new(node));
        }
        head
    }
}

pub struct ListSolution {}

impl ListSolution {
    /*
    leetcode link: https://leetcode.com/problems/remove-nth-node-from-end-of-list/
    we can use two pointers to solve this problem in one pass
     */
    pub fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
        // calculate the length of the list by using std::iter::successors, time complexity O(n)
        let cnt = std::iter::successors(head.as_ref(),
        |last| last.next.as_ref()).count();
        // dummy node to handle the case when we need to remove the first node
        let mut dummy = Some(Box::new(ListNode { val: 0, next: head }));
        // find the previous node of the node we want to remove
        let mut prev =
            (0..cnt - n as usize).fold(dummy.as_mut(),
            |curr, _| curr.unwrap().next.as_mut());
        // remove the node
        prev.unwrap().next = prev.as_mut().unwrap().next.as_mut().unwrap().next.take();
        dummy.unwrap().next
    }

    pub fn add_two_numbers(l1: Option<Box<ListNode>>, l2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        match (l1, l2) {
            (None, None) => None,
            (Some(l1), None) => Some(l1),
            (None, Some(l2)) => Some(l2),
            (Some(l1), Some(l2)) => {
                let val = l1.val + l2.val;
                if val < 10 {
                    let mut node = ListNode::new(val);
                    node.next = ListSolution::add_two_numbers(l1.next, l2.next);
                    Some(Box::new(node))
                } else {
                    let mut node = ListNode::new(val - 10);
                    node.next = ListSolution::add_two_numbers(
                        ListSolution::add_two_numbers(l1.next, Some(Box::new(ListNode::new(1)))), l2.next);
                    Some(Box::new(node))
                }
            }
        } 
    }
}

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
    assert_eq!(ListSolution::add_two_numbers(l1, l2), ListNode::from_vec(vec![7, 0, 8]));

    let l1 = ListNode::from_vec(vec![0]);
    let l2 = ListNode::from_vec(vec![0]);
    assert_eq!(ListSolution::add_two_numbers(l1, l2), ListNode::from_vec(vec![0]));

    let l1 = ListNode::from_vec(vec![9, 9, 9, 9, 9, 9, 9]);
    let l2 = ListNode::from_vec(vec![9, 9, 9, 9]);
    assert_eq!(ListSolution::add_two_numbers(l1, l2), ListNode::from_vec(vec![8, 9, 9, 9, 0, 0, 0, 1]));
    
}