use std::{cell::RefCell, rc::Rc};

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<std::rc::Rc<std::cell::RefCell<TreeNode>>>,
    pub right: Option<std::rc::Rc<std::cell::RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    #[allow(dead_code)]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

struct TreeSolution {}

impl TreeSolution {
    #[allow(dead_code)]
    pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if root.is_none() {
            return 0;
        }

        let left = Self::max_depth(root.as_ref().unwrap().borrow().left.clone());
        let right = Self::max_depth(root.as_ref().unwrap().borrow().right.clone());

        left.max(right) + 1
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
}
