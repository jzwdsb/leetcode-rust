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

    /*
    https://leetcode.com/problems/binary-tree-inorder-traversal/
    */
    #[allow(dead_code)]
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

    /*
    https://leetcode.com/problems/symmetric-tree/
     */
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
}
