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

    pub fn is_same_tree(p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
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
}
