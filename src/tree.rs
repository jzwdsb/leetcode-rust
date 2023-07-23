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

struct TreeSolution {}

impl TreeSolution {
    pub fn max_depth(root: Option<std::rc::Rc<std::cell::RefCell<TreeNode>>>) -> i32 {
        if root.is_none() {
            return 0;
        }

        let left = Self::max_depth(root.as_ref().unwrap().borrow().left.clone());
        let right = Self::max_depth(root.as_ref().unwrap().borrow().right.clone());

        left.max(right) + 1
    }
    pub fn min_depth(root: Option<std::rc::Rc<std::cell::RefCell<TreeNode>>>) -> i32 {
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
}
