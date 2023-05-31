pub struct StackSolution {}

impl StackSolution {
    pub fn simplify_path(path: String) -> String {
        let mut stack = std::collections::VecDeque::new();
        let words = path.split("/");

        for word in words {
            if word == "." || word == "" {
                continue;
            } else if word == ".." {
                stack.pop_back();
            } else {
                stack.push_back(word.to_string());
            }
        }
        if stack.is_empty() {
            return "/".to_string();
        }

        stack.iter().fold("".to_string(), |acc, x| acc + "/" + x)
    }
}

pub fn main() {

}

#[test]
fn test_simplify_path() {
    assert_eq!(StackSolution::simplify_path("/home/".to_string()), "/home");
    assert_eq!(StackSolution::simplify_path("/../".to_string()), "/");
    assert_eq!(StackSolution::simplify_path("/home//foo/".to_string()), "/home/foo");
    assert_eq!(StackSolution::simplify_path("/a/./b/../../c/".to_string()), "/c");
    assert_eq!(StackSolution::simplify_path("/a/../../b/../c//.//".to_string()), "/c");
    assert_eq!(StackSolution::simplify_path("/a//b////c/d//././/..".to_string()), "/a/b/c"); 
}