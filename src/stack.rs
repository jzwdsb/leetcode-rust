#![allow(dead_code)]

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

    pub fn longest_valid_parentheses(s: String) -> i32 {
        let mut stack = std::collections::VecDeque::new();
        let mut max = 0;
        stack.push_back(-1);

        for (i, c) in s.chars().enumerate() {
            if c == '(' {
                stack.push_back(i as i32);
            } else {
                stack.pop_back();
                if stack.is_empty() {
                    // update start index
                    stack.push_back(i as i32);
                } else {
                    max = max.max(i as i32 - stack.back().unwrap());
                }
            }
        }
        max
    }
}

pub fn main() {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_simplify_path() {
        assert_eq!(StackSolution::simplify_path("/home/".to_string()), "/home");
        assert_eq!(StackSolution::simplify_path("/../".to_string()), "/");
        assert_eq!(
            StackSolution::simplify_path("/home//foo/".to_string()),
            "/home/foo"
        );
        assert_eq!(
            StackSolution::simplify_path("/a/./b/../../c/".to_string()),
            "/c"
        );
        assert_eq!(
            StackSolution::simplify_path("/a/../../b/../c//.//".to_string()),
            "/c"
        );
        assert_eq!(
            StackSolution::simplify_path("/a//b////c/d//././/..".to_string()),
            "/a/b/c"
        );
    }

    #[test]
    fn test_longest_valid_parentheses() {
        assert_eq!(
            StackSolution::longest_valid_parentheses("(()".to_string()),
            2
        );
        assert_eq!(
            StackSolution::longest_valid_parentheses(")()())".to_string()),
            4
        );
        assert_eq!(
            StackSolution::longest_valid_parentheses("()(()".to_string()),
            2
        );
    }
}
