#![allow(dead_code)]

pub struct RecursiveSolution {}

impl RecursiveSolution {
    /*
    link: https://leetcode.com/problems/regular-expression-matching/
    a simple regexp matcher
    '.' matches any single character
    '*' matches zero or more of the preceding element
    we can define the matching process as follows
    1. if next pattern char is not '*', then current input char must be matched else return false
    2. if next pattern char is '*', then current input char can be matched or not. check next input chat until not matched
    3. if current pattern char is '.', then current input char is matched, check next input and pattern char
    4. if current pattern char is not '.', then current input char must be matched, check next input and pattern char
    5. if input chars are all matched and pattern chars are all matched, then return true
    6. if input chars are all matched and pattern chars are not all matched, then return false
    7. if input chars are not all matched and pattern chars are all matched, then return false
     */
    pub fn is_match(s: String, p: String) -> bool {
        let s_char = s.chars().collect::<Vec<char>>();
        let p_char = p.chars().collect::<Vec<char>>();
        return Self::match_helper(&s_char, &p_char, 0, 0);
    }

    fn match_helper(s: &Vec<char>, p: &Vec<char>, s_index: usize, p_index: usize) -> bool {
        if s_index == s.len() && p_index == p.len() {
            return true;
        }
        if s_index == s.len() && p_index < p.len() {
            if p_index + 1 < p.len() && p[p_index + 1] == '*' {
                return Self::match_helper(s, p, s_index, p_index + 2);
            } else {
                return false;
            }
        }

        // still input chars after the end of pattern
        if s_index < s.len() && p_index == p.len() {
            return false;
        }

        // if next char is *
        if p_index + 1 < p.len() && p[p_index + 1] == '*' {
            // current chat is matched or pattern char is .
            if p[p_index] == '.' || s[s_index] == p[p_index] {
                // move to next char in input string or move to next pattern char
                return Self::match_helper(s, p, s_index + 1, p_index)
                    || Self::match_helper(s, p, s_index, p_index + 2);
            } else {
                // current char is not matched
                return Self::match_helper(s, p, s_index, p_index + 2);
            }
        }
        // current char is matched or pattern char is .
        if p[p_index] == '.' || p[p_index] == s[s_index] {
            // both move to next char
            return Self::match_helper(s, p, s_index + 1, p_index + 1);
        }
        // nothing is matched
        return false;
    }
}

pub fn main() {}

#[test]
fn test_is_match() {
    assert_eq!(
        RecursiveSolution::is_match("aa".to_string(), "a".to_string()),
        false
    );
    assert_eq!(
        RecursiveSolution::is_match("aa".to_string(), "a*".to_string()),
        true
    );
    assert_eq!(
        RecursiveSolution::is_match("ab".to_string(), ".*".to_string()),
        true
    );
    assert_eq!(
        RecursiveSolution::is_match("aab".to_string(), "c*a*b".to_string()),
        true
    );
    assert_eq!(
        RecursiveSolution::is_match("mississippi".to_string(), "mis*is*p*.".to_string()),
        false
    );
    assert_eq!(
        RecursiveSolution::is_match("ab".to_string(), ".*c".to_string()),
        false
    );
    assert_eq!(
        RecursiveSolution::is_match("aaa".to_string(), "a*a".to_string()),
        true
    );
    assert_eq!(
        RecursiveSolution::is_match("aaa".to_string(), "ab*a*c*a".to_string()),
        true
    );
}
