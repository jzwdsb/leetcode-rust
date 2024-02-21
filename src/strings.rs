#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

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
                sum %= 10;
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
        use std::collections::hash_map::Entry::Vacant;
        let mut map: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for s in strs {
            let mut s_ = s.chars().collect::<Vec<char>>();
            s_.sort();
            let key = s_.iter().collect::<String>();
            if let Vacant(e) = map.entry(key.clone()) {
                e.insert(vec![s]);
            } else {
                map.get_mut(&key).unwrap().push(s);
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
        state.is_valid_end_state()
    }

    pub fn is_number_brute(s: String) -> bool {
        // trim leading and trailing whitespace
        let s = s.trim().to_string();
        if s.is_empty() {
            return false;
        }
        Self::check_exp(&s) || Self::check_decimal(&s) || Self::check_integer(&s, false)
    }

    fn check_exp(s: &str) -> bool {
        if s.is_empty() {
            return false;
        }
        if !s.contains('e') {
            return false;
        }

        let strs: Vec<&str> = s.split('e').collect();
        if strs.len() != 2 {
            return false;
        }
        if strs[0].is_empty() || strs[1].is_empty() {
            return false;
        }
        (Self::check_decimal(strs[0]) || Self::check_integer(strs[0], false))
            && Self::check_integer(strs[1], true)
    }

    fn check_decimal(s: &str) -> bool {
        if s.is_empty() {
            return false;
        }
        if !s.contains('.') {
            return false;
        }
        let strs: Vec<&str> = s.split('.').collect();
        if strs.len() == 1 {
            return Self::check_integer(strs[0], true);
        }
        if strs.len() == 2 {
            return Self::check_integer(strs[0], false) && Self::check_integer(strs[1], true);
        }
        false
    }

    fn check_integer(s: &str, is_sub: bool) -> bool {
        if s.is_empty() {
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
            if !s.chars().nth(i).unwrap().is_ascii_digit() {
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
        while !a.is_empty() || !b.is_empty() {
            let mut sum = carry;
            if !a.is_empty() {
                sum += a.pop().unwrap().to_digit(10).unwrap();
            }
            if !b.is_empty() {
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
        if diff.is_empty() {
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
        matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O')
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
                }
                (Some(sc), None) => {
                    if sc != &t[i] {
                        return false;
                    }
                    map2.insert(t[i], s[i]);
                }
                (None, Some(tc)) => {
                    if tc != &s[i] {
                        return false;
                    }
                    map.insert(s[i], t[i]);
                }
                (None, None) => {
                    map.insert(s[i], t[i]);
                    map2.insert(t[i], s[i]);
                }
            }
        }
        true
    }

    // s is a string of digits
    // create a palindrome number from s
    // return the maximum palindrome number
    pub fn max_palindromic_number(s: String) -> String {
        let mut digit_cnts = [0; 10];
        for c in s.chars() {
            if let Some(digit) = c.to_digit(10) {
                digit_cnts[digit as usize] += 1;
            }
        }

        let (mut half, mut mid) = (String::new(), String::new());

        // build half string from digit that appears even times
        // this is the second half of the palindrome
        // build mid string from digit that appears odd times
        // mid string can only build once
        for i in (0..=9).rev() {
            let count = digit_cnts[i];
            if count % 2 == 0 {
                half.push_str(&i.to_string().repeat(count / 2));
            } else if mid.is_empty() {
                mid.push_str(&i.to_string().repeat(count));
            }
        }

        // remove the trailing zeros
        half = half.trim_end_matches('0').to_string();

        half.clone() + &mid + &half.chars().rev().collect::<String>()
    }

    pub fn compare_version(version1: String, version2: String) -> i32 {
        let mut v1 = version1.split('.').map(|s| s.parse().unwrap());
        let mut v2 = version2.split('.').map(|s| s.parse().unwrap());
        loop {
            match (v1.next(), v2.next()) {
                (Some(v1), v2) if v1 > v2.unwrap_or(0) => return 1,
                (v1, Some(v2)) if v2 > v1.unwrap_or(0) => return -1,
                (None, None) => return 0,
                _ => continue,
            }
        }
    }
    /*
    https://leetcode.com/problems/decode-ways/

    we can use dp to solve this problem

    define dp[n], where dp[i] is the number of ways to decode the string s[0..i]

    dp[i] = dp[i-1] + dp[i-2] if s[i-1] and s[i] can be decoded as a single character
    dp[i] = dp[i-1] if s[i-1] and s[i] cannot be decoded as a single character

    dp[0] = 1
    dp[1] = 1 if s[0] != '0'
    dp[1] = 0 if s[0] == '0'

     */
    pub fn num_decodings(s: String) -> i32 {
        let mut dp = vec![0; s.len() + 1];
        let s = s.chars().collect::<Vec<char>>();
        dp[0] = 1;
        dp[1] = if s[0] == '0' { 0 } else { 1 };
        for i in 2..=s.len() {
            let one = s[i - 1..i]
                .iter()
                .collect::<String>()
                .parse::<i32>()
                .unwrap();
            let two = s[i - 2..i]
                .iter()
                .collect::<String>()
                .parse::<i32>()
                .unwrap();
            if (1..=9).contains(&one) {
                dp[i] += dp[i - 1];
            }
            if (10..=26).contains(&two) {
                dp[i] += dp[i - 2];
            }
        }
        dp[s.len()]
    }

    /*
    https://leetcode.com/problems/longest-palindromic-substring/

    find the longest palindromic substring in a string
    define dp[i][j] as the substring s[i..=j] is a palindrome
    dp[i][j] = 1 if s[i..=j] is a palindrome
               0 else
    dp[i][j] = 1 if s[i] == s[j] && (j - i <= 2 || dp[i+1][j-1] == 1)
             = 0 else
     */

    pub fn longest_palindrome(s: String) -> String {
        let mut dp = vec![vec![0; s.len()]; s.len()];
        let s = s.chars().collect::<Vec<char>>();
        let mut max_len = 0;
        let (mut start, mut end) = (0, 0);
        for i in (0..s.len()).rev() {
            for j in i..s.len() {
                // dp[i][j] = 1 means s[i..=j] is a palindrome
                if s[i] == s[j] && (j - i <= 2 || dp[i + 1][j - 1] == 1) {
                    dp[i][j] = 1;
                    if j - i + 1 > max_len {
                        max_len = j - i + 1;
                        start = i;
                        end = j;
                    }
                }
            }
        }
        s[start..=end].iter().collect::<String>()
    }

    /*
    https://leetcode.com/problems/word-break/

    define dp
    dp[i] = true if s[0..i] can be segmented into a space-separated sequence of one or more dictionary words
            false otherwise
     */

    pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
        let mut dp = vec![false; s.len() + 1];
        let s = s.chars().collect::<Vec<char>>();
        dp[0] = true;
        for i in 1..=s.len() {
            for word in word_dict.iter() {
                if i >= word.len() && dp[i - word.len()] {
                    let match_word = s[i - word.len()..i] == word.chars().collect::<Vec<char>>();
                    if match_word {
                        dp[i] = true;
                        break;
                    }
                }
            }
        }
        dp[s.len()]
    }

    /*
    https://leetcode.com/problems/interleaving-string/

    define dp

    dp[i][j] = true if s1[0..i] and s2[0..j] can be interleaved to s3[0..i+j]
               false otherwise
     */

    pub fn is_interleave(s1: String, s2: String, s3: String) -> bool {
        let s1 = s1.chars().collect::<Vec<char>>();
        let s2 = s2.chars().collect::<Vec<char>>();
        let s3 = s3.chars().collect::<Vec<char>>();
        if s1.len() + s2.len() != s3.len() {
            return false;
        }
        let mut dp = vec![vec![false; s2.len() + 1]; s1.len() + 1];
        dp[0][0] = true;
        for i in 1..=s1.len() {
            dp[i][0] = dp[i - 1][0] && s1[i - 1] == s3[i - 1];
        }
        for j in 1..=s2.len() {
            dp[0][j] = dp[0][j - 1] && s2[j - 1] == s3[j - 1];
        }
        for i in 1..=s1.len() {
            for j in 1..=s2.len() {
                dp[i][j] = (dp[i - 1][j] && s1[i - 1] == s3[i + j - 1])
                    || (dp[i][j - 1] && s2[j - 1] == s3[i + j - 1]);
            }
        }
        dp[s1.len()][s2.len()]
    }

    pub fn word_pattern(pattern: String, s: String) -> bool {
        let mut map = std::collections::HashMap::<char, &str>::new();
        let mut map2 = std::collections::HashMap::<&str, char>::new();
        let pattern = pattern.chars().collect::<Vec<char>>();
        let s = s.split(' ').collect::<Vec<&str>>();
        if s.len() != pattern.len() {
            return false;
        }

        for i in 0..s.len() {
            match (map.get(&pattern[i]), map2.get(&s[i])) {
                (Some(&word), Some(&p)) => {
                    if word != s[i] || p != pattern[i] {
                        return false;
                    }
                }
                (Some(&c), None) => {
                    if c != s[i] {
                        return false;
                    }
                    map2.insert(s[i], pattern[i]);
                }
                (None, Some(&p)) => {
                    if p != pattern[i] {
                        return false;
                    }
                    map.insert(pattern[i], s[i]);
                }
                (None, None) => {
                    map.insert(pattern[i], s[i]);
                    map2.insert(s[i], pattern[i]);
                }
            }
        }

        true
    }

    /*
    https://leetcode.com/problems/partitioning-into-minimum-number-of-deci-binary-numbers/
     */

    pub fn min_partitions(n: String) -> i32 {
        let mut max = 0;
        for c in n.chars() {
            let digit = c.to_digit(10).unwrap();
            if digit > max {
                max = digit;
            }
        }
        max as i32
    }

    pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
        let mut dp = vec![vec![0; text2.len() + 1]; text1.len() + 1];
        let text1 = text1.chars().collect::<Vec<char>>();
        let text2 = text2.chars().collect::<Vec<char>>();
        for i in 1..=text1.len() {
            for j in 1..=text2.len() {
                if text1[i - 1] == text2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    continue;
                }
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
        dp[text1.len()][text2.len()]
    }

    pub fn longest_common_sub_string(text1: String, text2: String) -> i32 {
        let mut dp = vec![vec![0; text2.len() + 1]; text1.len() + 1];
        let text1 = text1.chars().collect::<Vec<char>>();
        let text2 = text2.chars().collect::<Vec<char>>();
        let mut max = 0;
        for i in 1..=text1.len() {
            for j in 1..=text2.len() {
                if text1[i - 1] == text2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    max = std::cmp::max(max, dp[i][j]);
                }
            }
        }
        max
    }

    /*
    https://leetcode.com/problems/palindrome-partitioning/
     */

    pub fn partition_palindrome(s: String) -> Vec<Vec<String>> {
        let mut res = Vec::new();
        let mut path = Vec::new();
        Self::dfs(&s, 0, &mut path, &mut res);
        res
    }

    fn dfs(s: &str, start: usize, path: &mut Vec<String>, res: &mut Vec<Vec<String>>) {
        if start == s.len() {
            res.push(path.clone());
            return;
        }
        for i in start..s.len() {
            if Self::is_palindrome(s[start..=i].to_string()) {
                path.push(s[start..=i].to_string());
                Self::dfs(s, i + 1, path, res);
                path.pop();
            }
        }
    }

    pub fn max_number_of_balloons(text: String) -> i32 {
        let mut map = std::collections::HashMap::new();
        for c in text.chars() {
            *map.entry(c).or_insert(0) += 1;
        }
        map.entry('l').and_modify(|c| *c /= 2);
        map.entry('o').and_modify(|c| *c /= 2);
        let mut ans = std::i32::MAX;
        for c in "balloon".chars() {
            let count = map.entry(c).or_insert(0);
            if *count < ans {
                ans = *count;
            }
        }
        ans
    }

    pub fn reverse_words(s: String) -> String {
        s.split_ascii_whitespace()
            .rev()
            .filter(|&word| !word.is_empty())
            .map(|word| word.to_string())
            .collect::<Vec<String>>()
            .join(" ")
    }

    pub fn get_hint(secret: String, guess: String) -> String {
        let mut secret = secret.chars().collect::<Vec<char>>();
        let mut guess = guess.chars().collect::<Vec<char>>();
        let mut bulls = 0;
        let mut cows = 0;
        let mut map = std::collections::HashMap::new();
        for i in 0..secret.len() {
            if secret[i] == guess[i] {
                bulls += 1;
                secret[i] = ' ';
                guess[i] = ' ';
            } else {
                *map.entry(secret[i]).or_default() += 1;
            }
        }
        for &c in &guess {
            if c != ' ' && map.contains_key(&c) {
                cows += 1;
                let count = map.entry(c).or_insert(0);
                *count -= 1;
                if *count == 0 {
                    map.remove(&c);
                }
            }
        }
        format!("{}A{}B", bulls, cows)
    }

    pub fn halves_are_alike(s: String) -> bool {
        let s = s.to_ascii_lowercase();
        let (mut count1, mut count2) = (0, 0);
        for (i, c) in s.chars().enumerate() {
            if i < s.len() / 2 {
                if matches!(c, 'a' | 'e' | 'i' | 'o' | 'u') {
                    count1 += 1;
                }
            } else if matches!(c, 'a' | 'e' | 'i' | 'o' | 'u') {
                count2 += 1;
            }
        }
        count1 == count2
    }

    pub fn max_length_between_equal_characters(s: String) -> i32 {
        let mut map = std::collections::HashMap::new();
        let s = s.chars().collect::<Vec<char>>();
        let mut max = -1;
        for (i, c) in s.iter().enumerate() {
            if let Some(&j) = map.get(c) {
                max = std::cmp::max(max, (i - j - 1) as i32);
            } else {
                map.insert(s[i], i);
            }
        }
        max
    }

    pub fn min_operations(s: String) -> i32 {
        let s = s.chars().collect::<Vec<char>>();
        let mut count1 = 0; // the number of operations to make s[i] == '0'
        let mut count2 = 0; // the number of operations to make s[i] == '1'
        for (i, &c) in s.iter().enumerate() {
            if i % 2 == 0 {
                if c == '1' {
                    count1 += 1;
                } else {
                    count2 += 1;
                }
            } else if c == '0' {
                count1 += 1;
            } else {
                count2 += 1;
            }
        }
        count1.min(count2)
    }

    pub fn path_crossing(path: String) -> bool {
        let mut visited = HashSet::new();
        let mut curr = (0, 0);
        visited.insert(curr);
        for c in path.chars() {
            match c {
                'N' => curr.1 += 1,
                'S' => curr.1 -= 1,
                'E' => curr.0 += 1,
                'W' => curr.0 -= 1,
                _ => unreachable!(),
            }
            if visited.contains(&curr) {
                return true;
            }
            visited.insert(curr);
        }
        false
    }

    pub fn max_score(s: String) -> i32 {
        let s = s.chars().collect::<Vec<char>>();
        let mut max = 0;
        let mut zeros = 0;
        let mut ones = s.iter().filter(|&&c| c == '1').count() as i32;
        for &c in s.iter().take(s.len() - 1) {
            if c == '0' {
                zeros += 1;
            } else {
                ones -= 1;
            }
            max = std::cmp::max(max, zeros + ones);
        }
        max
    }

    pub fn min_deletions(s: String) -> i32 {
        let mut freqs = vec![0; 26];
        for c in s.chars() {
            freqs[c as usize - 'a' as usize] += 1;
        }

        let mut count = 0;
        let mut set = HashSet::new();
        for freq in freqs {
            if freq == 0 {
                continue;
            }
            if set.contains(&freq) {
                let mut freq = freq;
                while freq > 0 && set.contains(&freq) {
                    freq -= 1;
                    count += 1;
                }
                if freq > 0 {
                    set.insert(freq);
                }
            } else {
                set.insert(freq);
            }
        }
        count
    }

    pub fn find_anagrams(s: String, p: String) -> Vec<i32> {
        if p.len() > s.len() {
            return vec![];
        }
        let mut patterns = vec![0; 26];
        let mut freqs = vec![0; 26];
        let s = s.chars().collect::<Vec<char>>();
        let p = p.chars().collect::<Vec<char>>();
        let mut res = Vec::new();
        for i in 0..p.len() {
            patterns[p[i] as usize - 'a' as usize] += 1;
            freqs[s[i] as usize - 'a' as usize] += 1;
        }
        if patterns == freqs {
            res.push(0);
        }
        for i in p.len()..s.len() {
            freqs[s[i] as usize - 'a' as usize] += 1;
            freqs[s[i - p.len()] as usize - 'a' as usize] -= 1;
            if patterns == freqs {
                res.push((i - p.len() + 1) as i32);
            }
        }
        res
    }

    pub fn count_segments(s: String) -> i32 {
        // one line
        // s.split_ascii_whitespace().count() as i32
        if s.is_empty() {
            return 0;
        }
        let mut space_cnt = 0;
        let mut i = 0;
        // skip heading spaces
        let s = s.chars().collect::<Vec<char>>();
        while i < s.len() && s[i] == ' ' {
            i += 1;
        }
        while i < s.len() {
            if s[i] == ' ' {
                space_cnt += 1;
                // skip subsequent spaces
                while i < s.len() && s[i] == ' ' {
                    i += 1;
                }
            }
            i += 1;
        }

        // remove trailing space
        if s.last().unwrap() == &' ' {
            space_cnt -= 1;
        }

        space_cnt + 1
    }

    // use dp to solve the problem
    pub fn longest_str_chain(mut words: Vec<String>) -> i32 {
        words.sort_unstable_by_key(String::len);
        let mut map = HashMap::new();

        let mut res = 0;
        for word in words {
            map.insert(word.clone(), 1);
            for i in 0..word.len() {
                let prev_word = format!("{}{}", &word[..i], &word[i + 1..]);
                if let Some(&count) = map.get(&prev_word) {
                    map.insert(word.clone(), map[&word].max(count + 1));
                }
            }
            res = res.max(map[&word]);
        }

        res
    }

    pub fn find_the_difference(s: String, t: String) -> char {
        let mut char_count = [0; 26];
        for c in s.chars() {
            char_count[c as usize - 'a' as usize] += 1;
        }
        for c in t.chars() {
            char_count[c as usize - 'a' as usize] -= 1;
        }
        for (i, &count) in char_count.iter().enumerate() {
            if count < 0 {
                return (i as u8 + b'a') as char;
            }
        }
        unreachable!("no extra char found")
    }

    // remove duplicate letters so that every letter appears exactly once
    // return the result is the smallest in lexicographically order
    pub fn remove_duplicate_letters(s: String) -> String {
        let mut stack = vec![];
        let mut seen = HashSet::new();
        let mut last_occ = HashMap::new();
        for (i, c) in s.chars().enumerate() {
            last_occ.insert(c, i);
        }
        for (i, c) in s.chars().enumerate() {
            if !seen.contains(&c) {
                while let Some(&top) = stack.last() {
                    if c < top && i < *last_occ.get(&top).unwrap() {
                        seen.remove(&top);
                        stack.pop();
                    } else {
                        break;
                    }
                }
                seen.insert(c);
                stack.push(c);
            }
        }

        stack.iter().collect()
    }
} // impl StringSolution

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

    #[test]
    fn test_max_palindromic_number() {
        assert_eq!(
            StringSolution::max_palindromic_number("28398".to_string()),
            "898".to_string()
        );
    }

    #[test]
    fn test_compare_version() {
        assert_eq!(
            StringSolution::compare_version("1.01".to_string(), "1.001".to_string()),
            0
        );
        assert_eq!(
            StringSolution::compare_version("1.0".to_string(), "1.0.0".to_string()),
            0
        );
        assert_eq!(
            StringSolution::compare_version("0.1".to_string(), "1.1".to_string()),
            -1
        );
        assert_eq!(
            StringSolution::compare_version("1.0.1".to_string(), "1".to_string()),
            1
        );
    }

    #[test]
    fn test_decode_number() {
        assert_eq!(StringSolution::num_decodings("12".to_string()), 2);
        assert_eq!(StringSolution::num_decodings("226".to_string()), 3);
        assert_eq!(StringSolution::num_decodings("0".to_string()), 0);
        assert_eq!(StringSolution::num_decodings("06".to_string()), 0);
    }

    #[test]
    fn test_longest_palindrome() {
        assert_eq!(
            StringSolution::longest_palindrome("babad".to_string()),
            "aba".to_string()
        );
        assert_eq!(
            StringSolution::longest_palindrome("cbbd".to_string()),
            "bb".to_string()
        );
        assert_eq!(
            StringSolution::longest_palindrome("a".to_string()),
            "a".to_string()
        );
        assert_eq!(
            StringSolution::longest_palindrome("ac".to_string()),
            "c".to_string()
        );
        assert_eq!(
            StringSolution::longest_palindrome("aaaa".to_string()),
            "aaaa".to_string()
        );
    }

    #[test]
    fn test_word_break() {
        assert_eq!(
            StringSolution::word_break(
                "leetcode".to_string(),
                vec!["leet".to_string(), "code".to_string(),]
            ),
            true
        );
        assert_eq!(
            StringSolution::word_break(
                "applepenapple".to_string(),
                vec!["apple".to_string(), "pen".to_string()]
            ),
            true
        );
        assert_eq!(
            StringSolution::word_break(
                "catsandog".to_string(),
                vec![
                    "cats".to_string(),
                    "dog".to_string(),
                    "sand".to_string(),
                    "and".to_string(),
                    "cat".to_string()
                ]
            ),
            false
        );
    }

    #[test]
    fn test_is_interleave() {
        assert_eq!(
            StringSolution::is_interleave(
                "aabcc".to_string(),
                "dbbca".to_string(),
                "aadbbcbcac".to_string()
            ),
            true
        );
        assert_eq!(
            StringSolution::is_interleave(
                "aabcc".to_string(),
                "dbbca".to_string(),
                "aadbbbaccc".to_string()
            ),
            false
        );
        assert_eq!(
            StringSolution::is_interleave("".to_string(), "".to_string(), "".to_string()),
            true
        );
        assert_eq!(
            StringSolution::is_interleave("a".to_string(), "".to_string(), "a".to_string()),
            true
        )
    }

    #[test]
    fn test_word_pattern() {
        assert_eq!(
            StringSolution::word_pattern("abba".to_string(), "dog cat cat dog".to_string()),
            true
        );
        assert_eq!(
            StringSolution::word_pattern("abba".to_string(), "dog cat cat fish".to_string()),
            false
        );
        assert_eq!(
            StringSolution::word_pattern("abbc".to_string(), "dog cat cat dog".to_string()),
            false
        );
        assert_eq!(
            StringSolution::word_pattern("avva".to_string(), "dog dog dog dog".to_string()),
            false
        );
    }

    #[test]
    fn test_min_partitions() {
        assert_eq!(StringSolution::min_partitions("32".to_string()), 3);
        assert_eq!(StringSolution::min_partitions("82734".to_string()), 8);
        assert_eq!(
            StringSolution::min_partitions("27346209830709182346".to_string()),
            9
        );
    }

    #[test]
    fn test_longest_common_sub_sequence() {
        assert_eq!(
            StringSolution::longest_common_subsequence("abcde".to_string(), "ace".to_string()),
            3
        );
        assert_eq!(
            StringSolution::longest_common_subsequence("abc".to_string(), "def".to_string()),
            0
        );
    }

    #[test]
    fn test_longest_common_sub_string() {
        assert_eq!(
            StringSolution::longest_common_sub_string("abcde".to_string(), "ace".to_string()),
            1
        );
        assert_eq!(
            StringSolution::longest_common_sub_string("abc".to_string(), "def".to_string()),
            0
        );
    }

    #[test]
    fn test_partition_palindrome() {
        let mut result = StringSolution::partition_palindrome("aab".to_string());
        for r in result.iter_mut() {
            r.sort();
        }
        result.sort();
        let mut expect = vec![
            vec!["a".to_string(), "a".to_string(), "b".to_string()],
            vec!["aa".to_string(), "b".to_string()],
        ];
        for e in expect.iter_mut() {
            e.sort();
        }
        expect.sort();
        assert_eq!(result, expect);
    }

    #[test]
    fn test_max_number_of_balloons() {
        assert_eq!(
            StringSolution::max_number_of_balloons("nlaebolko".to_string()),
            1
        );
        assert_eq!(
            StringSolution::max_number_of_balloons("loonbalxballpoon".to_string()),
            2
        );
        assert_eq!(
            StringSolution::max_number_of_balloons("leetcode".to_string()),
            0
        );
    }

    #[test]
    fn test_reverse_words() {
        assert_eq!(
            StringSolution::reverse_words("The sky is blue".to_string()),
            "blue is sky The".to_string()
        );
    }

    #[test]
    fn test_get_hint() {
        assert_eq!(
            StringSolution::get_hint("1807".to_string(), "7810".to_string()),
            "1A3B".to_string()
        );
        assert_eq!(
            StringSolution::get_hint("1123".to_string(), "0111".to_string()),
            "1A1B".to_string()
        );
        assert_eq!(
            StringSolution::get_hint("1122".to_string(), "1222".to_string()),
            "3A0B".to_string()
        );
    }

    #[test]
    fn test_halves_are_alike() {
        assert_eq!(StringSolution::halves_are_alike("book".to_string()), true);
        assert_eq!(
            StringSolution::halves_are_alike("textbook".to_string()),
            false
        );
        assert_eq!(
            StringSolution::halves_are_alike("MerryChristmas".to_string()),
            false
        );
        assert_eq!(
            StringSolution::halves_are_alike("AbCdEfGh".to_string()),
            true
        );
    }

    #[test]
    fn test_max_length_between_equal_characters() {
        assert_eq!(
            StringSolution::max_length_between_equal_characters("aa".to_string()),
            0
        );
        assert_eq!(
            StringSolution::max_length_between_equal_characters("abca".to_string()),
            2
        );
        assert_eq!(
            StringSolution::max_length_between_equal_characters("cbzxy".to_string()),
            -1
        );
        assert_eq!(
            StringSolution::max_length_between_equal_characters("cabbac".to_string()),
            4
        );
    }

    #[test]
    fn test_min_operations() {
        assert_eq!(StringSolution::min_operations("0100".to_string()), 1);
        assert_eq!(StringSolution::min_operations("10".to_string()), 0);
        assert_eq!(StringSolution::min_operations("1111".to_string()), 2);
    }

    #[test]
    fn test_path_crossing() {
        assert_eq!(StringSolution::path_crossing("NES".to_string()), false);
        assert_eq!(StringSolution::path_crossing("NESWW".to_string()), true);
    }

    #[test]
    fn test_max_score() {
        assert_eq!(StringSolution::max_score("011101".to_string()), 5);
        assert_eq!(StringSolution::max_score("00111".to_string()), 5);
        assert_eq!(StringSolution::max_score("1111".to_string()), 3);
    }

    #[test]
    fn test_min_deletions() {
        assert_eq!(StringSolution::min_deletions("aab".to_string()), 0);
        assert_eq!(StringSolution::min_deletions("aaabbbcc".to_string()), 2);
        assert_eq!(StringSolution::min_deletions("ceabaacb".to_string()), 2);
    }

    #[test]
    fn test_find_anagrams() {
        assert_eq!(
            StringSolution::find_anagrams("cbaebabacd".to_string(), "abc".to_string()),
            vec![0, 6]
        );
        assert_eq!(
            StringSolution::find_anagrams("abab".to_string(), "ab".to_string()),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn test_count_segments() {
        assert_eq!(
            StringSolution::count_segments("Hello, my name is John".to_string()),
            5
        );
        assert_eq!(StringSolution::count_segments("Hello".to_string()), 1);
        assert_eq!(StringSolution::count_segments("".to_string()), 0);
        assert_eq!(StringSolution::count_segments("    ".to_string()), 0);
    }

    #[test]
    fn test_longest_str_chain() {
        assert_eq!(
            StringSolution::longest_str_chain(vec![
                "a".to_string(),
                "b".to_string(),
                "ba".to_string(),
                "bca".to_string(),
                "bda".to_string(),
                "bdca".to_string()
            ]),
            4
        );
        assert_eq!(
            StringSolution::longest_str_chain(vec![
                "xbc".to_string(),
                "pcxbcf".to_string(),
                "xb".to_string(),
                "cxbc".to_string(),
                "pcxbc".to_string()
            ]),
            5
        );

        assert_eq!(
            StringSolution::longest_str_chain(vec![
                "bdca".to_string(),
                "bda".to_string(),
                "ca".to_string(),
                "dca".to_string(),
                "a".to_string()
            ]),
            4
        )
    }

    #[test]
    fn test_remove_duplicates_letters() {
        assert_eq!(
            StringSolution::remove_duplicate_letters("bcabc".to_string()),
            "abc".to_string()
        );
        assert_eq!(
            StringSolution::remove_duplicate_letters("cbacdcbc".to_string()),
            "acdb"
        )
    }
}
