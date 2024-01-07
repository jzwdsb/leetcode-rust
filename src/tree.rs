#![allow(dead_code)]

use std::collections::VecDeque;
use std::{cell::RefCell, rc::Rc};

use crate::list::ListNode;

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<std::rc::Rc<std::cell::RefCell<TreeNode>>>,
    pub right: Option<std::rc::Rc<std::cell::RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}
pub type Tree = Option<Rc<RefCell<TreeNode>>>;

struct TreeSolution {}

impl TreeSolution {
    pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if root.is_none() {
            return 0;
        }

        let left = Self::max_depth(root.as_ref().unwrap().borrow().left.clone());
        let right = Self::max_depth(root.as_ref().unwrap().borrow().right.clone());

        left.max(right) + 1
    }

    pub fn min_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if root.is_none() {
            return 0;
        }
        let left = Self::min_depth(root.as_ref().unwrap().borrow().left.clone());
        let right = Self::min_depth(root.as_ref().unwrap().borrow().right.clone());
        if left == 0 || right == 0 {
            return left.max(right) + 1;
        }
        left.min(right) + 1
    }

    pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        Self::is_valid_helper(&root, std::i64::MIN, std::i64::MAX)
    }

    fn is_valid_helper(root: &Option<Rc<RefCell<TreeNode>>>, gt: i64, lt: i64) -> bool {
        match root.as_ref() {
            None => true,
            Some(node) => {
                let node = node.borrow();
                if (node.val as i64) <= gt || (node.val as i64) >= lt {
                    return false;
                }
                Self::is_valid_helper(&node.left, gt, node.val as i64)
                    && Self::is_valid_helper(&node.right, node.val as i64, lt)
            }
        }
    }

    pub fn is_same_tree(
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> bool {
        match (p, q) {
            (None, None) => true,
            (Some(p), Some(q)) => {
                let p = p.borrow();
                let q = q.borrow();
                p.val == q.val
                    && Self::is_same_tree(p.left.clone(), q.left.clone())
                    && Self::is_same_tree(p.right.clone(), q.right.clone())
            }
            _ => false,
        }
    }

    /*
    link: https://leetcode.com/problems/unique-binary-search-trees/
    return the number of unique binary search trees that can be composited with n nodes

    n = 0, res = 1
    n = 1, res = 1
    n = 2, res = 2
    n = 3, res = 5
    n = 4, res = 14

    given a sequence of numbers from 1 to n, we can choose a number as the root node,
    then the left part of the root node is the left subtree,
    the right part of the root node is the right subtree,

    if we chose i as the root node, then the left subtree has i - 1 nodes, the right subtree has n - i nodes,
    so the number of unique binary search trees that can be composited with i nodes is: f(i - 1) * f(n - i)

    so we can inference the equation that is:
    f(n) = f(0) * f(n - 1) + f(1) * f(n - 2) + ... + f(n - 1) * f(0)

    we already know that f(0) = 1, f(1) = 1, so we can build a dp array to store the result of f(i)
    from 0 to n,

    dp[i] represents the number of unique binary search trees that can be composited with i nodes
    dp[i] = dp[0] * dp[i - 1] + dp[1] * dp[i - 2] + ... + dp[i - 1] * dp[0]
     */
    pub fn num_trees(n: i32) -> i32 {
        let mut dp = vec![0; (n + 1) as usize];
        dp[0] = 1;
        dp[1] = 1;

        for i in 2..=n as usize {
            for j in 1..=i {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }

        dp[n as usize]
    }

    pub fn preorder_traversal(root: Tree) -> Vec<i32> {
        let mut res = vec![];
        Self::preorder_helper(root, &mut res);
        res
    }

    fn preorder_helper(root: Tree, res: &mut Vec<i32>) {
        match root {
            None => {}
            Some(root) => {
                let root = root.borrow();
                res.push(root.val);
                Self::preorder_helper(root.left.clone(), res);
                Self::preorder_helper(root.right.clone(), res);
            }
        }
    }

    pub fn postorder_traversal(root: Tree) -> Vec<i32> {
        let mut res = vec![];
        Self::postorder_helper(root, &mut res);
        res
    }

    fn postorder_helper(root: Tree, res: &mut Vec<i32>) {
        match root {
            None => {}
            Some(root) => {
                let root = root.borrow();
                Self::postorder_helper(root.left.clone(), res);
                Self::postorder_helper(root.right.clone(), res);
                res.push(root.val);
            }
        }
    }

    pub fn breadth_first_traversal(root: Tree) -> Vec<i32> {
        let mut res = vec![];
        let mut queue = VecDeque::new();
        queue.push_back(root.clone());
        while let Some(front) = queue.pop_front() {
            match front {
                None => {}
                Some(front) => {
                    let front = front.borrow();
                    res.push(front.val);
                    queue.push_back(front.left.clone());
                    queue.push_back(front.right.clone());
                }
            }
        }
        res
    }

    /*
    https://leetcode.com/problems/binary-tree-inorder-traversal/
    */
    pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut res = vec![];
        Self::inorder_helper(&root, &mut res);
        res
    }

    fn inorder_helper(root: &Option<Rc<RefCell<TreeNode>>>, res: &mut Vec<i32>) {
        match root.as_ref() {
            None => {}
            Some(node) => {
                let node = node.borrow();
                Self::inorder_helper(&node.left, res);
                res.push(node.val);
                Self::inorder_helper(&node.right, res);
            }
        }
    }

    pub fn level_order_traversal(root: Tree) -> Vec<Vec<i32>> {
        let mut res = vec![];
        if root.is_none() {
            return res;
        }
        let mut queue = VecDeque::new();
        queue.push_back(root.clone());
        let mut level_cnt = 1;
        while !queue.is_empty() {
            let mut level_res = vec![];
            let mut next_level_cnt = 0;
            for _ in 0..level_cnt {
                if let Some(node) = queue.pop_front().unwrap() {
                    let node = node.borrow();
                    level_res.push(node.val);
                    if node.left.is_some() {
                        queue.push_back(node.left.clone());
                        next_level_cnt += 1
                    }
                    if node.right.is_some() {
                        queue.push_back(node.right.clone());
                        next_level_cnt += 1
                    }
                }
            }
            res.push(level_res);
            level_cnt = next_level_cnt;
        }

        res
    }

    fn zigzag_level_traversal(root: Tree) -> Vec<Vec<i32>> {
        let mut res = vec![];
        if root.is_none() {
            return res;
        }
        let mut queue = VecDeque::new();
        queue.push_back(root.unwrap().clone());
        let mut zigzag = false;
        let mut level_cnt = 1;

        while level_cnt > 0 {
            let mut level_res = vec![];
            let mut next_level_cnt = 0;
            for _ in 0..level_cnt {
                if let Some(node) = queue.pop_front() {
                    let node = node.borrow();
                    level_res.push(node.val);
                    if let Some(left) = node.left.clone() {
                        queue.push_back(left);
                        next_level_cnt += 1;
                    }
                    if let Some(right) = node.right.clone() {
                        queue.push_back(right);
                        next_level_cnt += 1;
                    }
                }
            }
            if zigzag {
                level_res.reverse();
            }
            level_cnt = next_level_cnt;
            zigzag = !zigzag;
            res.push(level_res);
        }

        res
    }

    pub fn level_order_bottom(root: Tree) -> Vec<Vec<i32>> {
        let mut res = Self::level_order_traversal(root);

        res.reverse();
        res
    }

    pub fn build_tree(preorder_tree: Vec<i32>, inorder: Vec<i32>) -> Tree {
        if preorder_tree.len() == 0 {
            return None;
        }
        let mut preorder_tree = preorder_tree;
        let inorder = inorder;
        let root_val = preorder_tree.remove(0);
        let mut root = TreeNode::new(root_val);
        let root_idx = inorder.iter().position(|&x| x == root_val).unwrap();

        let left_inorder = inorder[0..root_idx].to_vec();
        let right_inorder = inorder[root_idx + 1..].to_vec();
        let left_preorder = preorder_tree[0..left_inorder.len()].to_vec();
        let right_preorder = preorder_tree[left_inorder.len()..].to_vec();
        root.left = Self::build_tree(left_preorder, left_inorder);
        root.right = Self::build_tree(right_preorder, right_inorder);
        Some(Rc::new(RefCell::new(root)))
    }

    pub fn build_tree_from_inorder_and_postorder(inorder: Vec<i32>, postorder: Vec<i32>) -> Tree {
        if inorder.is_empty() {
            return None;
        }
        let mut postorder = postorder;
        let last = postorder.pop().unwrap();
        let mut root = TreeNode::new(last);
        let root_idx = inorder.iter().position(|&x| x == root.val).unwrap();
        let left_inorder = inorder[0..root_idx].to_vec();
        let right_inorder = inorder[root_idx + 1..].to_vec();

        let left_postorder = postorder[0..left_inorder.len()].to_vec();
        let right_postorder = postorder[left_inorder.len()..].to_vec();
        root.left = Self::build_tree_from_inorder_and_postorder(left_inorder, left_postorder);
        root.right = Self::build_tree_from_inorder_and_postorder(right_inorder, right_postorder);

        Some(Rc::new(RefCell::new(root)))
    }

    /*
    https://leetcode.com/problems/symmetric-tree/
     */
    pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        Self::is_symmetric_helper(&root, &root)
    }

    fn is_symmetric_helper(
        left: &Option<Rc<RefCell<TreeNode>>>,
        right: &Option<Rc<RefCell<TreeNode>>>,
    ) -> bool {
        match (left, right) {
            (None, None) => true,
            (Some(left), Some(right)) => {
                let left = left.borrow();
                let right = right.borrow();
                left.val == right.val
                    && Self::is_symmetric_helper(&left.left, &right.right)
                    && Self::is_symmetric_helper(&left.right, &right.left)
            }
            _ => false,
        }
    }
    pub fn is_balanced(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        match root {
            None => return true,
            Some(root) => {
                let root = root.borrow();
                let left = Self::max_depth(root.left.clone());
                let right = Self::max_depth(root.right.clone());
                if (left - right).abs() > 1 {
                    return false;
                }
                Self::is_balanced(root.left.clone()) && Self::is_balanced(root.right.clone())
            }
        }
    }

    pub fn has_path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> bool {
        match root {
            None => false,
            Some(root) => {
                let root = root.borrow();
                if root.left.is_none() && root.right.is_none() {
                    return root.val == target_sum;
                }
                Self::has_path_sum(root.left.clone(), target_sum - root.val)
                    || Self::has_path_sum(root.right.clone(), target_sum - root.val)
            }
        }
    }

    pub fn has_path_sum_ii(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> Vec<Vec<i32>> {
        let mut res = vec![];
        let mut path = vec![];
        Self::path_sum_helper(root, target_sum, &mut path, &mut res);
        res
    }

    fn path_sum_helper(
        root: Option<Rc<RefCell<TreeNode>>>,
        target_sum: i32,
        path: &mut Vec<i32>,
        res: &mut Vec<Vec<i32>>,
    ) {
        match root {
            None => {}
            Some(root) => {
                let root = root.borrow();
                path.push(root.val);
                if root.left.is_none() && root.right.is_none() && root.val == target_sum {
                    res.push(path.clone());
                    path.pop();
                    return;
                }
                Self::path_sum_helper(root.left.clone(), target_sum - root.val, path, res);
                Self::path_sum_helper(root.right.clone(), target_sum - root.val, path, res);
                path.pop();
                return;
            }
        }
    }

    /*
    https://leetcode.com/problems/recover-binary-search-tree/
     */

    pub fn recover_tree(root: &mut Tree) {
        let mut first: Tree = None;
        let mut second: Tree = None;
        let mut prev: Tree = None;

        Self::recover_helper(root, &mut first, &mut second, &mut prev);

        std::mem::swap(
            &mut first.as_ref().unwrap().borrow_mut().val,
            &mut second.as_ref().unwrap().borrow_mut().val,
        );
    }

    fn recover_helper(root: &mut Tree, first: &mut Tree, second: &mut Tree, prev: &mut Tree) {
        match root {
            None => {}
            Some(root) => {
                Self::recover_helper(&mut root.borrow().left.clone(), first, second, prev);
                if prev.is_some() && prev.as_ref().unwrap().borrow().val > root.borrow().val {
                    if first.is_none() {
                        *first = prev.clone();
                    }
                    *second = Some(root.clone());
                }
                *prev = Some(root.clone());
                Self::recover_helper(&mut root.borrow().right.clone(), first, second, prev);
            }
        }
    }

    // calculate the sum of depth of all nodes
    // depth of a node is the number of edges from the node to the tree's root node
    pub fn sum_of_depth(root: Tree) -> usize {
        Self::sum_depth_helper(root, 0)
    }

    fn sum_depth_helper(root: Tree, depth: usize) -> usize {
        match root {
            None => 0,
            Some(root) => {
                let root = root.borrow();
                depth
                    + Self::sum_depth_helper(root.left.clone(), depth + 1)
                    + Self::sum_depth_helper(root.right.clone(), depth + 1)
            }
        }
    }

    fn node_distance(root: Tree, p: i32, q: i32) -> i32 {
        let mut p_path = vec![];
        let mut q_path = vec![];
        Self::node_distance_helper(root.clone(), p, &mut p_path);
        Self::node_distance_helper(root.clone(), q, &mut q_path);
        p_path.reverse();
        q_path.reverse();
        let mut i = 0;
        while i < p_path.len() && i < q_path.len() {
            if p_path[i] != q_path[i] {
                break;
            }
            i += 1;
        }

        (p_path.len() + q_path.len() - 2 * i) as i32
    }

    fn node_distance_helper(root: Tree, target: i32, path: &mut Vec<i32>) -> bool {
        match root {
            None => false,
            Some(root) => {
                let root = root.borrow();
                if root.val == target {
                    path.push(root.val);
                    return true;
                }
                if Self::node_distance_helper(root.left.clone(), target, path)
                    || Self::node_distance_helper(root.right.clone(), target, path)
                {
                    path.push(root.val);
                    return true;
                }
                false
            }
        }
    }

    pub fn sorted_array_to_bst(nums: Vec<i32>) -> Tree {
        Self::sorted_array_to_bst_helper(&nums, 0, nums.len())
    }

    fn sorted_array_to_bst_helper(nums: &Vec<i32>, start: usize, end: usize) -> Tree {
        if start >= end {
            return None;
        }
        let mid: usize = (start + end) / 2;
        let mut root = TreeNode::new(nums[mid]);
        root.left = Self::sorted_array_to_bst_helper(nums, start, mid);
        root.right = Self::sorted_array_to_bst_helper(nums, mid + 1, end);
        Some(Rc::new(RefCell::new(root)))
    }

    pub fn is_binary_search_tree(root: Tree) -> bool {
        match root {
            None => true,
            Some(root) => {
                let root = root.borrow();
                if root.left.is_none() && root.right.is_none() {
                    return true;
                }
                let left = root.left.clone();
                let right = root.right.clone();
                if left.is_some() && left.as_ref().unwrap().borrow().val >= root.val {
                    return false;
                }
                if right.is_some() && right.as_ref().unwrap().borrow().val <= root.val {
                    return false;
                }

                Self::is_binary_search_tree(root.left.clone())
                    && Self::is_binary_search_tree(root.right.clone())
            }
        }
    }

    pub fn is_balance(root: Tree) -> bool {
        match root {
            None => true,
            Some(root) => {
                let root = root.borrow();
                if root.left.is_none() && root.right.is_none() {
                    return true;
                }
                let left = Self::max_depth(root.left.clone());
                let right = Self::max_depth(root.right.clone());
                if (left - right).abs() > 1 {
                    return false;
                }
                Self::is_balance(root.left.clone()) && Self::is_balance(root.right.clone())
            }
        }
    }

    /*
    https://leetcode.com/problems/flatten-binary-tree-to-linked-list/

    flatten tree in preorder
    thus: root -> left -> right
     */
    pub fn flatten(root: &mut Tree) {
        if root.is_none() {
            return;
        }
        let mut stack = vec![];
        stack.push(root.clone());
        while let Some(node) = stack.pop() {
            let node = node.unwrap();
            if node.as_ref().borrow().right.is_some() {
                stack.push(node.as_ref().borrow_mut().right.clone());
            }
            if node.as_ref().borrow().left.is_some() {
                stack.push(node.as_ref().borrow_mut().left.clone());
            }
            if let Some(last) = stack.last() {
                node.as_ref().borrow_mut().right = last.clone();
                node.as_ref().borrow_mut().left = None;
            }
        }
    }

    pub fn sorted_list_to_bst(head: Option<Box<ListNode<i32>>>) -> Tree {
        let mut nums = vec![];
        let mut head = head;
        while let Some(node) = head {
            nums.push(node.val);
            head = node.next;
        }
        Self::sorted_array_to_bst(nums)
    }

    pub fn sum_numbers(root: Tree) -> i32 {
        Self::sum_helper(root, 0)
    }

    pub fn sum_helper(root: Tree, path: i32) -> i32 {
        match root {
            None => 0,
            Some(root) => {
                let root = root.borrow();
                let path = path * 10 + root.val;
                if root.left.is_none() && root.left.is_none() {
                    return path;
                }
                let sum = Self::sum_helper(root.left.clone(), path)
                    + Self::sum_helper(root.right.clone(), path);
                sum
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_depth() {
        let mut root = TreeNode::new(3);
        let mut left = TreeNode::new(9);
        let right = TreeNode::new(20);
        let left_left = TreeNode::new(15);
        let left_right = TreeNode::new(7);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::max_depth(Some(std::rc::Rc::new(std::cell::RefCell::new(root)))),
            3
        );
    }

    #[test]
    fn test_min_depth() {
        let mut root = TreeNode::new(3);
        let mut left = TreeNode::new(9);
        let right = TreeNode::new(20);
        let left_left = TreeNode::new(15);
        let left_right = TreeNode::new(7);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::min_depth(Some(std::rc::Rc::new(std::cell::RefCell::new(root)))),
            2
        );
    }

    #[test]
    fn test_is_valid_bst() {
        let mut root = TreeNode::new(2);
        let left = TreeNode::new(1);
        let right = TreeNode::new(3);

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::is_valid_bst(Some(std::rc::Rc::new(std::cell::RefCell::new(root)))),
            true
        );
    }

    #[test]
    fn test_same_tree() {
        let mut root = TreeNode::new(1);
        let left = TreeNode::new(2);
        let right = TreeNode::new(3);

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        let mut root2 = TreeNode::new(1);
        let left2 = TreeNode::new(2);
        let right2 = TreeNode::new(3);

        root2.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left2)));
        root2.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right2)));

        assert_eq!(
            TreeSolution::is_same_tree(
                Some(std::rc::Rc::new(std::cell::RefCell::new(root))),
                Some(std::rc::Rc::new(std::cell::RefCell::new(root2)))
            ),
            true
        );
    }

    #[test]
    fn test_num_trees() {
        assert_eq!(TreeSolution::num_trees(3), 5);
    }

    #[test]
    fn test_preorder_traversal() {
        let mut root = TreeNode::new(1);
        let mut right = TreeNode::new(2);
        let right_left = TreeNode::new(3);

        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::preorder_traversal(Some(std::rc::Rc::new(std::cell::RefCell::new(root)))),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn test_inorder_traversal() {
        let mut root = TreeNode::new(1);
        let mut right = TreeNode::new(2);
        let right_left = TreeNode::new(3);

        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::inorder_traversal(Some(std::rc::Rc::new(std::cell::RefCell::new(root)))),
            vec![1, 3, 2]
        );
    }

    #[test]
    fn test_postorder_travesal() {
        let mut root = TreeNode::new(1);
        let mut right = TreeNode::new(2);
        let right_left = TreeNode::new(3);

        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::postorder_traversal(Some(std::rc::Rc::new(std::cell::RefCell::new(
                root
            )))),
            vec![3, 2, 1]
        );
    }

    #[test]
    fn test_breadth_first_search() {
        let mut root = TreeNode::new(1);
        let mut left = TreeNode::new(2);
        let mut right = TreeNode::new(3);
        let left_left = TreeNode::new(4);
        let left_right = TreeNode::new(5);
        let right_left = TreeNode::new(6);
        let right_right = TreeNode::new(7);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));
        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::breadth_first_traversal(Some(std::rc::Rc::new(std::cell::RefCell::new(
                root
            )))),
            vec![1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn test_is_symmetric() {
        let mut root = TreeNode::new(1);
        let mut left = TreeNode::new(2);
        let mut right = TreeNode::new(2);
        let left_left = TreeNode::new(3);
        let left_right = TreeNode::new(4);
        let right_left = TreeNode::new(4);
        let right_right = TreeNode::new(3);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));
        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::is_symmetric(Some(std::rc::Rc::new(std::cell::RefCell::new(root)))),
            true
        );
    }

    #[test]
    fn test_is_balance() {
        let mut root = TreeNode::new(3);
        let mut left = TreeNode::new(9);
        let right = TreeNode::new(20);
        let left_left = TreeNode::new(15);
        let left_right = TreeNode::new(7);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::is_balanced(Some(std::rc::Rc::new(std::cell::RefCell::new(root)))),
            true
        );
    }

    #[test]
    fn test_has_path_sum() {
        let mut root = TreeNode::new(5);
        let mut left = TreeNode::new(4);
        let mut right = TreeNode::new(8);
        let mut left_left = TreeNode::new(11);
        let left_right = TreeNode::new(13);
        let right_left = TreeNode::new(4);
        let mut right_right: TreeNode = TreeNode::new(1);
        let left_left_left = TreeNode::new(7);
        let left_left_right = TreeNode::new(2);
        let right_right_right = TreeNode::new(5);

        left_left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left_left)));
        left_left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left_right)));
        right_right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right_right_right)));

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));
        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::has_path_sum(Some(std::rc::Rc::new(std::cell::RefCell::new(root))), 22),
            true
        );
    }

    #[test]
    fn test_has_path_sum_ii() {
        let mut root = TreeNode::new(5);
        let mut left = TreeNode::new(4);
        let mut right = TreeNode::new(8);
        let mut left_left = TreeNode::new(11);
        let left_right = TreeNode::new(13);
        let right_left = TreeNode::new(4);
        let mut right_right: TreeNode = TreeNode::new(1);
        let left_left_left = TreeNode::new(7);
        let left_left_right = TreeNode::new(2);
        let right_right_right = TreeNode::new(5);

        left_left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left_left)));
        left_left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left_right)));
        right_right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right_right_right)));

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));
        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::has_path_sum_ii(
                Some(std::rc::Rc::new(std::cell::RefCell::new(root))),
                22
            ),
            vec![vec![5, 4, 11, 2], vec![5, 4, 13]]
        );
    }

    #[test]
    fn test_recover_tree() {
        let root = std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(1)));
        let left = std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(3)));
        let right = std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(2)));

        root.as_ref().borrow_mut().left = Some(left.clone());
        left.as_ref().borrow_mut().right = Some(right);

        TreeSolution::recover_tree(&mut Some(root.clone()));

        assert_eq!(TreeSolution::inorder_traversal(Some(root)), vec![1, 2, 3]);

        let root = std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(3)));
        let left = std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(1)));
        let right = std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(4)));
        let right_left = std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(2)));

        root.as_ref().borrow_mut().left = Some(left.clone());
        root.as_ref().borrow_mut().right = Some(right.clone());
        right.as_ref().borrow_mut().left = Some(right_left);

        assert_eq!(
            TreeSolution::inorder_traversal(Some(root.clone())),
            vec![1, 3, 2, 4]
        );
        TreeSolution::recover_tree(&mut Some(root.clone()));
        assert_eq!(
            TreeSolution::inorder_traversal(Some(root)),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn test_sum_of_depth() {
        let mut root = TreeNode::new(3);
        let mut left = TreeNode::new(9);
        let right = TreeNode::new(20);
        let left_left = TreeNode::new(15);
        let left_right = TreeNode::new(7);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::sum_of_depth(Some(std::rc::Rc::new(std::cell::RefCell::new(root)))),
            6
        );
    }

    #[test]
    fn test_node_distence() {
        let mut root = TreeNode::new(3);
        let mut left = TreeNode::new(5);
        let mut right = TreeNode::new(1);
        let left_left = TreeNode::new(6);
        let mut left_right = TreeNode::new(2);
        let right_left = TreeNode::new(0);
        let right_right = TreeNode::new(8);
        let left_right_left = TreeNode::new(7);
        let left_right_right = TreeNode::new(4);

        left_right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right_left)));
        left_right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right_right)));
        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));
        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right_right)));
        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::node_distance(
                Some(std::rc::Rc::new(std::cell::RefCell::new(root))),
                5,
                1
            ),
            2
        );
    }

    #[test]
    fn test_sort_array_to_bst() {
        let nums = vec![-10, -3, 0, 5, 9];
        let mut root = TreeNode::new(0);
        let mut left = TreeNode::new(-3);
        let mut right = TreeNode::new(9);
        let left_left = TreeNode::new(-10);
        let right_left = TreeNode::new(5);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));
        assert_eq!(
            TreeSolution::sorted_array_to_bst(nums.clone()),
            Some(std::rc::Rc::new(std::cell::RefCell::new(root)))
        );
    }

    #[test]
    fn test_sorted_list_to_bst() {
        let lists = ListNode::from_vec(vec![-10, -3, 0, 5, 9]);

        let mut root = TreeNode::new(0);
        let mut left = TreeNode::new(-3);
        let mut right = TreeNode::new(9);
        let left_left = TreeNode::new(-10);
        let right_left = TreeNode::new(5);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::sorted_list_to_bst(lists),
            Some(std::rc::Rc::new(std::cell::RefCell::new(root)))
        );
    }

    #[test]
    fn test_level_traversal() {
        let mut root = TreeNode::new(3);
        let mut left = TreeNode::new(9);
        let right = TreeNode::new(20);
        let left_left = TreeNode::new(15);
        let left_right = TreeNode::new(7);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));
        assert_eq!(
            TreeSolution::level_order_traversal(Some(std::rc::Rc::new(std::cell::RefCell::new(
                root
            )))),
            vec![vec![3], vec![9, 20], vec![15, 7]]
        );

        assert_eq!(
            TreeSolution::level_order_traversal(None),
            Vec::<Vec<i32>>::new()
        )
    }

    #[test]
    fn test_zigzag_level_traversal() {
        let mut root = TreeNode::new(3);
        let mut left = TreeNode::new(9);
        let right = TreeNode::new(20);
        let left_left = TreeNode::new(15);
        let left_right = TreeNode::new(7);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));
        assert_eq!(
            TreeSolution::zigzag_level_traversal(Some(std::rc::Rc::new(std::cell::RefCell::new(
                root
            )))),
            vec![vec![3], vec![20, 9], vec![15, 7]]
        );

        assert_eq!(
            TreeSolution::zigzag_level_traversal(None),
            Vec::<Vec<i32>>::new()
        )
    }

    #[test]
    fn test_level_order_bottom() {
        let mut root = TreeNode::new(3);
        let mut left = TreeNode::new(9);
        let right = TreeNode::new(20);
        let left_left = TreeNode::new(15);
        let left_right = TreeNode::new(7);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));
        assert_eq!(
            TreeSolution::level_order_bottom(Some(std::rc::Rc::new(std::cell::RefCell::new(root)))),
            vec![vec![15, 7], vec![9, 20], vec![3]]
        );

        assert_eq!(
            TreeSolution::level_order_bottom(None),
            Vec::<Vec<i32>>::new()
        )
    }

    #[test]
    fn test_build_tree() {
        let preorder = vec![3, 9, 20, 15, 7];
        let inorder = vec![9, 3, 15, 20, 7];
        let mut root = TreeNode::new(3);
        let left = TreeNode::new(9);
        let mut right = TreeNode::new(20);
        let right_left = TreeNode::new(15);
        let right_right = TreeNode::new(7);

        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::build_tree(preorder, inorder),
            Some(std::rc::Rc::new(std::cell::RefCell::new(root)))
        );
    }

    #[test]
    fn test_build_tree_from_inorder_postorder() {
        let inorder = vec![9, 3, 15, 20, 7];
        let postorder = vec![9, 15, 7, 20, 3];
        let mut root = TreeNode::new(3);
        let left = TreeNode::new(9);
        let mut right = TreeNode::new(20);
        let right_left = TreeNode::new(15);
        let right_right = TreeNode::new(7);

        right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(right_left)));
        right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right_right)));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::build_tree_from_inorder_and_postorder(inorder, postorder),
            Some(std::rc::Rc::new(std::cell::RefCell::new(root)))
        );
    }

    #[test]
    fn test_flatten() {
        let mut root = TreeNode::new(1);
        let mut left = TreeNode::new(2);
        let mut right = TreeNode::new(5);
        let left_left = TreeNode::new(3);
        let left_right = TreeNode::new(4);
        let right_right = TreeNode::new(6);

        left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left_left)));
        left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(left_right)));
        right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right_right)));
        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));
        let root = Some(std::rc::Rc::new(std::cell::RefCell::new(root)));

        TreeSolution::flatten(&mut root.clone());

        assert_eq!(root.as_ref().unwrap().borrow().val, 1);
        assert_eq!(
            root.as_ref()
                .unwrap()
                .borrow()
                .right
                .as_ref()
                .unwrap()
                .borrow()
                .val,
            2
        );
        assert_eq!(root.as_ref().unwrap().borrow().left, None);
        assert_eq!(
            root.as_ref()
                .unwrap()
                .borrow()
                .right
                .as_ref()
                .unwrap()
                .borrow()
                .right
                .as_ref()
                .unwrap()
                .borrow()
                .val,
            3
        );
        assert_eq!(
            root.as_ref()
                .unwrap()
                .borrow()
                .right
                .as_ref()
                .unwrap()
                .borrow()
                .right
                .as_ref()
                .unwrap()
                .borrow()
                .right
                .as_ref()
                .unwrap()
                .borrow()
                .val,
            4
        );
    }

    #[test]
    fn test_sum_numbers() {
        let mut root = TreeNode::new(1);
        let left = TreeNode::new(2);
        let right = TreeNode::new(3);

        // left.left = Some(std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(4))));
        // left.right = Some(std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(5))));
        // right.left = Some(std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(6))));
        // right.right = Some(std::rc::Rc::new(std::cell::RefCell::new(TreeNode::new(7))));

        root.left = Some(std::rc::Rc::new(std::cell::RefCell::new(left)));
        root.right = Some(std::rc::Rc::new(std::cell::RefCell::new(right)));

        assert_eq!(
            TreeSolution::sum_numbers(Some(std::rc::Rc::new(std::cell::RefCell::new(root)))),
            12 + 13
        );
    }

    fn new_node(val: i32) -> Tree {
        Some(Rc::new(RefCell::new(TreeNode::new(val))))
    }
}
