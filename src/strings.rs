pub struct StringSolution {}

// Solution from https://leetcode.com/problems/valid-number/solutions/3461898/finite-state-machine/
// use state machine to solve check is number problem
enum CheckState {
    FloatSign, // ACCEPTS: '+' | '-' | '.' | '0'..='9'
    FloatInit, // ACCEPTS: '.' | '0'..='9'
    FloatNum,  // ACCEPTS: '.' | 'e' | 'E' | '0'..='9'
    IntInit,   // ACCEPTS: '0'..='9'
    IntNum,    // ACCEPTS: 'e' | 'E' | '0'..='9'
    ExpSign,   // ACCEPTS: '+' | '-' | '0'..='9'
    ExpInit,   // ACCEPTS: '0'..='9'
    ExpNum,    // ACCEPTS: '0'..='9'
}

// use state machine to solve check is number problem
// by enumerate all possible state and transition
impl CheckState {
    pub fn accept(&self, c: char) -> Result<Self, ()> {
        match (c, self) {
            ('+' | '-', Self::FloatSign) => Ok(Self::FloatInit),
            ('+' | '-', Self::ExpSign) => Ok(Self::ExpInit),
            ('.', Self::FloatSign | Self::FloatInit) => Ok(Self::IntInit),
            ('.', Self::FloatNum) => Ok(Self::IntNum),
            ('e' | 'E', Self::FloatNum | Self::IntNum) => Ok(Self::ExpSign),
            ('0'..='9', Self::FloatSign | Self::FloatInit | Self::FloatNum) => Ok(Self::FloatNum),
            ('0'..='9', Self::IntInit | Self::IntNum) => Ok(Self::IntNum),
            ('0'..='9', Self::ExpSign | Self::ExpInit | Self::ExpNum) => Ok(Self::ExpNum),
            _ => Err(()),
        }
    }
    pub fn is_valid_end_state(&self) -> bool {
        matches!(self, Self::FloatNum | Self::IntNum | Self::ExpNum)
    }
}

impl StringSolution {
    /*
    leetcode link: https://leetcode.com/problems/count-ways-to-build-good-strings/

    the actual prupose of the questions is to find out how many ways we can sum one and zero to a range of [low, high]

    algorithms: dynamic programming
                define a dp array with length of high + 1
                dp[i] represents the number of ways to sum one and zero to i
                dp[0] = 1
                dp[i] = dp[i - zero] + dp[i - one]
                ans = sum(dp[low..=high])

     */
    pub fn count_good_string(low: i32, high: i32, zero: i32, one: i32) -> i32 {
        let modulo = 1_000_000_000 + 7;
        let (low, high) = (low as usize, high as usize);
        let (one, zero) = (one as usize, zero as usize);
        let mut ans = 0;
        let mut dp = vec![0; high + 1];

        dp[0] = 1;

        for i in 0..(high + 1) {
            let add_zero = i + zero;
            let add_one = i + one;
            if add_zero <= high {
                dp[add_zero] = (dp[add_zero] + dp[i]) % modulo;
            }
            if add_one <= high {
                dp[add_one] = (dp[add_one] + dp[i]) % modulo;
            }
            if i >= low {
                ans = (ans + dp[i]) % modulo;
            }
        }

        ans
    }
    /*
    link: https://leetcode.com/problems/count-and-say/
    fn(1) = "1"
    fn(n) = count nums of fn(n-1)
    count nums = count each char in fn(n-1) and print the times and value
     */

    pub fn count_and_say(n: i32) -> String {
        if n == 1 {
            return "1".to_string();
        }
        let mut ans = String::new();
        let prev = Self::count_and_say(n - 1);
        let mut count = 1;
        let mut i = 0;
        while i < prev.len() {
            if i == prev.len() - 1
                || prev.chars().nth(i).unwrap() != prev.chars().nth(i + 1).unwrap()
            {
                ans.push_str(count.to_string().as_str());
                ans.push_str(prev.as_str().chars().nth(i).unwrap().to_string().as_str());
                count = 1;
            } else {
                count += 1;
            }
            i += 1;
        }
        ans
    }
    /*
    link: https://leetcode.com/problems/multiply-strings/description/
    we need to simulate the process of multiplication
    the process of 123 x 456 can be described as
        123
        456
        ---
        738
       615
      492
        ---
      56088
    we can see that the result of each multiplication is stored in a matrix
    for position mat[i,j] = num1[i] * num2[j] + carry
                 carry = mat[i-1,j-1] / 10

    we define the input as Vec<char> nums1 and Vec<char> nums2
    the maximum length of the output is len(num1) + len(nums2)
    the we can define a vector to store the result of each multiplication
    we don't need to define a matrix with size of len(num1) * len(num2)
    we can just define a vector with size of len(num1) + len(num2)
    we iterate each char in num1 and num2 and calculate the result of each multiplication
    for i in num1 and j in num2
    vec[i,j] = num1[i] * num2[j] + carry + vec[i,j]

    we connect the element in the vector to a string and return it
     */

    pub fn multiply(num1: String, num2: String) -> String {
        let mut res = String::new();
        let num1 = num1.chars().rev().collect::<Vec<char>>();
        let num2 = num2.chars().rev().collect::<Vec<char>>();

        // num_res defines the result of each multiplication
        // num_res[i, j] = num1[i] * num2[j]
        let mut num_res = vec![0; num1.len() + num2.len()];
        for i in 0..num1.len() {
            let mut carry = 0;
            let n1 = num1[i] as i32 - '0' as i32;
            for j in 0..num2.len() {
                let n2 = num2[j] as i32 - '0' as i32;
                let mut sum = n1 * n2 + carry + num_res[i + j];
                carry = sum / 10;
                sum = sum % 10;
                num_res[i + j] = sum;
            }
            if carry > 0 {
                num_res[i + num2.len()] += carry;
            }
        }
        let mut i = num_res.len() - 1;

        while i > 0 && num_res[i] == 0 {
            i -= 1;
        }
        for j in (0..=i).rev() {
            res.push_str(num_res[j].to_string().as_str());
        }

        res
    }

    pub fn is_anagram(s: String, t: String) -> bool {
        let mut s = s.chars().collect::<Vec<char>>();
        let mut t = t.chars().collect::<Vec<char>>();
        s.sort();
        t.sort();
        s == t
    }

    /*
    link: https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/description/
    the default String does not implement the Pattern trait
    need to convert the String to &str
     */
    pub fn str_str(haystack: String, needle: String) -> i32 {
        haystack.find(&needle).map(|i| i as i32).unwrap_or(-1)
    }

    /*
    link: https://leetcode.com/problems/group-anagrams/description/
    we can sort each string and use the sorted string as the key of the hashmap
    push the string to the value of the hashmap
    return the values of the hashmap
     */

    pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
        let mut map: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for s in strs {
            let mut s_ = s.chars().collect::<Vec<char>>();
            s_.sort();
            let key = s_.iter().collect::<String>();
            if map.contains_key(&key) {
                map.get_mut(&key).unwrap().push(s);
            } else {
                map.insert(key, vec![s]);
            }
        }

        map.into_values().collect()
    }

    /*
    link: https://leetcode.com/problems/valid-number/
    brute search did work but it got time limit exceeded
    need to optimize the solution
     */

    pub fn is_number(s: String) -> bool {
        let s = s.trim();
        let mut state = CheckState::FloatSign;
        for c in s.chars() {
            state = match state.accept(c) {
                Ok(s) => s,
                Err(()) => return false,
            }
        }
        return state.is_valid_end_state();
    }

    pub fn is_number_brute(s: String) -> bool {
        // trim leading and trailing whitespace
        let s = s.trim().to_string();
        if s.len() == 0 {
            return false;
        }
        return Self::check_exp(&s) || Self::check_decimal(&s) || Self::check_integer(&s, false);
    }

    fn check_exp(s: &String) -> bool {
        if s.len() == 0 {
            return false;
        }
        if !s.contains('e') {
            return false;
        }

        let strs: Vec<&str> = s.split("e").collect();
        if strs.len() != 2 {
            return false;
        }
        if strs[0].len() == 0 || strs[1].len() == 0 {
            return false;
        }
        return (Self::check_decimal(&strs[0].to_string())
            || Self::check_integer(&strs[0].to_string(), false))
            && Self::check_integer(&strs[1].to_string(), true);
    }

    fn check_decimal(s: &String) -> bool {
        if s.len() == 0 {
            return false;
        }
        if !s.contains('.') {
            return false;
        }
        let strs: Vec<&str> = s.split(".").collect();
        if strs.len() == 1 {
            return Self::check_integer(&strs[0].to_string(), true);
        }
        if strs.len() == 2 {
            return Self::check_integer(&strs[0].to_string(), false)
                && Self::check_integer(&strs[1].to_string(), true);
        }
        return false;
    }

    fn check_integer(s: &String, is_sub: bool) -> bool {
        if s.len() == 0 {
            return false;
        }
        let mut i = 0;

        if s.chars().nth(i).unwrap() == '+' || s.chars().nth(i).unwrap() == '-' {
            if is_sub {
                return false;
            }
            i += 1;
        }
        if i == s.len() {
            return false;
        }
        while i < s.len() {
            if !s.chars().nth(i).unwrap().is_digit(10) {
                return false;
            }
            i += 1;
        }
        true
    }

    /*
    link: https://leetcode.com/problems/add-binary/
     */

    pub fn add_binary(a: String, b: String) -> String {
        let mut a = a.chars().collect::<Vec<char>>();
        let mut b = b.chars().collect::<Vec<char>>();
        let mut res = String::new();
        let mut carry = 0;
        while a.len() > 0 || b.len() > 0 {
            let mut sum = carry;
            if a.len() > 0 {
                sum += a.pop().unwrap().to_digit(10).unwrap();
            }
            if b.len() > 0 {
                sum += b.pop().unwrap().to_digit(10).unwrap();
            }
            carry = sum / 2;
            res.push_str(&(sum % 2).to_string());
        }
        if carry > 0 {
            res.push_str(&carry.to_string());
        }
        res.chars().rev().collect::<String>()
    }

    pub fn is_subsequence(s: String, t: String) -> bool {
        let mut i = 0;
        let mut j = 0;
        let s = s.chars().collect::<Vec<char>>();
        let t = t.chars().collect::<Vec<char>>();
        while i < s.len() && j < t.len() {
            if s[i] == t[j] {
                i += 1;
            }
            j += 1;
        }
        i == s.len()
    }

    /*
    links: https://leetcode.com/problems/buddy-strings/
    swap two characters in a string so the s == goal

     */

    pub fn buddy_strings(s: String, goal: String) -> bool {
        if s.len() != goal.len() {
            return false;
        }
        let mut s = s.chars().collect::<Vec<char>>();
        let goal = goal.chars().collect::<Vec<char>>();
        let mut diff = Vec::new();
        for i in 0..s.len() {
            if s[i] != goal[i] {
                diff.push(i);
            }
        }
        // if diff.len() == 0, then we need to check if there is a duplicate character
        // if there is a duplicate character, then we can swap it with itself
        // otherwise, we cannot swap any character
        if diff.len() == 0 {
            let mut map = std::collections::HashSet::new();
            for c in s {
                if map.contains(&c) {
                    return true;
                }
                map.insert(c);
            }
            return false;
        }
        if diff.len() != 2 {
            return false;
        }
        s.swap(diff[0], diff[1]);
        s == goal
    }

    pub fn is_palindrome(s: String) -> bool {
        let alpha = s
            .chars()
            .filter(|c| c.is_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect::<String>();
        alpha == alpha.chars().rev().collect::<String>()
    }

    pub fn reverse_vowels(s: String) -> String {
        let mut s = s.chars().collect::<Vec<char>>();
        let mut i = 0;
        let mut j = s.len() - 1;
        while i < j {
            if !Self::is_vowel(s[i]) {
                i += 1;
                continue;
            }
            if !Self::is_vowel(s[j]) {
                j -= 1;
                continue;
            }
            s.swap(i, j);
            i += 1;
            j -= 1;
        }
        s.into_iter().collect::<String>()
    }

    fn is_vowel(c: char) -> bool {
        matches!(
            c,
            'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O'
        )
    }

    pub fn is_isomorphic(s: String, t: String) -> bool {
        let mut map = std::collections::HashMap::new();
        let mut map2 = std::collections::HashMap::new();
        let s = s.chars().collect::<Vec<char>>();
        let t = t.chars().collect::<Vec<char>>();
        for i in 0..s.len() {
            match (map.get(&s[i]), map2.get(&t[i])) {
                (Some(sc), Some(tc)) => {
                    if sc != &t[i] || tc != &s[i] {
                        return false;
                    }
                },
                (Some(sc), None) => {
                    if sc != &t[i] {
                        return false;
                    }
                    map2.insert(t[i], s[i]);
                },
                (None, Some(tc)) => {
                    if tc != &s[i] {
                        return false;
                    }
                    map.insert(s[i], t[i]);
                },
                (None, None) => {
                    map.insert(s[i], t[i]);
                    map2.insert(t[i], s[i]);
                }
                
            }
        }
        true
    }
}

pub fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_good_string() {
        assert_eq!(StringSolution::count_good_string(3, 3, 1, 1), 8);
        assert_eq!(StringSolution::count_good_string(2, 3, 1, 2), 5);
        assert_eq!(StringSolution::count_good_string(10, 10, 2, 1), 89);
    }

    #[test]
    fn test_count_and_say() {
        assert_eq!(StringSolution::count_and_say(1), "1");
        assert_eq!(StringSolution::count_and_say(2), "11");
        assert_eq!(StringSolution::count_and_say(3), "21");
        assert_eq!(StringSolution::count_and_say(4), "1211");
        assert_eq!(StringSolution::count_and_say(5), "111221");
        assert_eq!(StringSolution::count_and_say(6), "312211");
    }

    #[test]
    fn test_multiply() {
        assert_eq!(
            StringSolution::multiply("2".to_string(), "3".to_string()),
            "6"
        );
        assert_eq!(
            StringSolution::multiply("123".to_string(), "456".to_string()),
            "56088"
        );
        assert_eq!(
            StringSolution::multiply("123456789".to_string(), "987654321".to_string()),
            "121932631112635269"
        );
    }

    #[test]
    fn test_str_str() {
        assert_eq!(
            StringSolution::str_str("hello".to_string(), "ll".to_string()),
            2
        );
        assert_eq!(
            StringSolution::str_str("aaaaa".to_string(), "bba".to_string()),
            -1
        );
        assert_eq!(StringSolution::str_str("".to_string(), "".to_string()), 0);
        assert_eq!(StringSolution::str_str("".to_string(), "a".to_string()), -1);
        assert_eq!(StringSolution::str_str("a".to_string(), "".to_string()), 0);
        assert_eq!(
            StringSolution::str_str("mississippi".to_string(), "issip".to_string()),
            4
        );
    }

    #[test]
    fn test_group_anagrams() {
        let mut result = StringSolution::group_anagrams(vec![
            "eat".to_string(),
            "tea".to_string(),
            "tan".to_string(),
            "ate".to_string(),
            "nat".to_string(),
            "bat".to_string(),
        ]);
        for r in result.iter_mut() {
            r.sort();
        }
        result.sort();
        let mut expect = vec![
            vec!["ate".to_string(), "eat".to_string(), "tea".to_string()],
            vec!["bat".to_string()],
            vec!["nat".to_string(), "tan".to_string()],
        ];
        for e in expect.iter_mut() {
            e.sort();
        }
        expect.sort();
        assert_eq!(result, expect);
    }

    #[test]
    fn test_is_number() {
        assert_eq!(StringSolution::is_number_brute("0".to_string()), true);
        assert_eq!(StringSolution::is_number_brute(" 0.1 ".to_string()), true);
        assert_eq!(StringSolution::is_number_brute("abc".to_string()), false);
        assert_eq!(StringSolution::is_number_brute("1 a".to_string()), false);
        assert_eq!(StringSolution::is_number_brute("2e10".to_string()), true);
        assert_eq!(
            StringSolution::is_number_brute(" -90e3   ".to_string()),
            true
        );
        assert_eq!(StringSolution::is_number_brute(" 1e".to_string()), false);
        assert_eq!(StringSolution::is_number_brute("1.+1".to_string()), false);

        assert_eq!(StringSolution::is_number("0".to_string()), true);
        assert_eq!(StringSolution::is_number(" 0.1 ".to_string()), true);
        assert_eq!(StringSolution::is_number("abc".to_string()), false);
        assert_eq!(StringSolution::is_number("1 a".to_string()), false);
        assert_eq!(StringSolution::is_number("2e10".to_string()), true);
        assert_eq!(StringSolution::is_number(" -90e3   ".to_string()), true);
        assert_eq!(StringSolution::is_number(" 1e".to_string()), false);
        assert_eq!(StringSolution::is_number("1.+1".to_string()), false);
    }

    #[test]
    fn test_add_binary() {
        assert_eq!(
            StringSolution::add_binary("11".to_string(), "1".to_string()),
            "100"
        );
        assert_eq!(
            StringSolution::add_binary("1010".to_string(), "1011".to_string()),
            "10101"
        );
    }

    #[test]
    fn test_is_subsequence() {
        assert_eq!(
            StringSolution::is_subsequence("abc".to_string(), "ahbgdc".to_string()),
            true
        );
        assert_eq!(
            StringSolution::is_subsequence("axc".to_string(), "ahbgdc".to_string()),
            false
        );
    }

    #[test]
    fn test_buddy_string() {
        assert_eq!(
            StringSolution::buddy_strings("ab".to_string(), "ba".to_string()),
            true
        );
        assert_eq!(
            StringSolution::buddy_strings("ab".to_string(), "ab".to_string()),
            false
        );
        assert_eq!(
            StringSolution::buddy_strings("aa".to_string(), "aa".to_string()),
            true
        );
        assert_eq!(
            StringSolution::buddy_strings("aaaaaaabc".to_string(), "aaaaaaacb".to_string()),
            true
        );
        assert_eq!(
            StringSolution::buddy_strings("".to_string(), "aa".to_string()),
            false
        );
    }

    #[test]
    fn test_is_palindrome() {
        assert_eq!(
            StringSolution::is_palindrome("A man, a plan, a canal: Panama".to_string()),
            true
        )
    }

    #[test]
    fn test_reverse_vowel() {
        assert_eq!(
            StringSolution::reverse_vowels("hello".to_string()),
            "holle".to_string()
        );
        assert_eq!(
            StringSolution::reverse_vowels("leetcode".to_string()),
            "leotcede".to_string()
        );
    }

    #[test]
    fn test_is_isomorphic() {
        assert_eq!(
            StringSolution::is_isomorphic("egg".to_string(), "add".to_string()),
            true
        );
        assert_eq!(
            StringSolution::is_isomorphic("foo".to_string(), "bar".to_string()),
            false
        );
        assert_eq!(
            StringSolution::is_isomorphic("paper".to_string(), "title".to_string()),
            true
        );
        assert_eq!(
            StringSolution::is_isomorphic("ab".to_string(), "aa".to_string()),
            false
        );
    }
}
