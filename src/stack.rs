#![allow(dead_code)]

use std::collections::VecDeque;

pub struct StackSolution {}

impl StackSolution {
    pub fn simplify_path(path: String) -> String {
        let mut stack = std::collections::VecDeque::new();
        let words = path.split('/');

        for word in words {
            if word == "." || word.is_empty() {
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

    // evaluate reverse polish notation
    // supported operators: +, -, *, /

    pub fn eval_rpn(tokens: Vec<String>) -> i32 {
        let mut stack = VecDeque::new();

        for token in tokens {
            match token.as_str() {
                "+" => {
                    let a = stack.pop_back().expect("invalid input");
                    let b = stack.pop_back().expect("invalid input");
                    stack.push_back(a + b);
                }
                "-" => {
                    let a = stack.pop_back().expect("invalid input");
                    let b = stack.pop_back().expect("invalid input");
                    stack.push_back(b - a);
                }
                "*" => {
                    let a = stack.pop_back().expect("invalid input");
                    let b = stack.pop_back().expect("invalid input");
                    stack.push_back(a * b);
                }
                "/" => {
                    let a = stack.pop_back().expect("invalid input");
                    let b = stack.pop_back().expect("invalid input");
                    stack.push_back(b / a);
                }
                _ => {
                    stack.push_back(token.parse::<i32>().expect("invalid input"));
                }
            }
        }

        stack.pop_back().unwrap()
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

    #[test]
    fn test_eval_rpn() {
        assert_eq!(
            StackSolution::eval_rpn(vec![
                "2".to_string(),
                "1".to_string(),
                "+".to_string(),
                "3".to_string(),
                "*".to_string()
            ]),
            9
        );
        assert_eq!(
            StackSolution::eval_rpn(vec![
                "4".to_string(),
                "13".to_string(),
                "5".to_string(),
                "/".to_string(),
                "+".to_string()
            ]),
            6
        );
        assert_eq!(
            StackSolution::eval_rpn(vec![
                "10".to_string(),
                "6".to_string(),
                "9".to_string(),
                "3".to_string(),
                "+".to_string(),
                "-11".to_string(),
                "*".to_string(),
                "/".to_string(),
                "*".to_string(),
                "17".to_string(),
                "+".to_string(),
                "5".to_string(),
                "+".to_string()
            ]),
            22
        );
    }
}
