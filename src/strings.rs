pub struct StringSolution {}

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
        let prev = Self::count_and_say(n-1);
        let mut count = 1;
        let mut i = 0;
        while i < prev.len() {
            if i == prev.len()-1 || prev.chars().nth(i).unwrap() != prev.chars().nth(i+1).unwrap() {
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
        let mut res  = String::new();
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
                let mut sum = n1 * n2 + carry + num_res[i+j];
                carry = sum / 10;
                sum = sum % 10;
                num_res[i+j] = sum;
            }
            if carry > 0 {
                num_res[i+num2.len()] += carry;
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
        let mut map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
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

}


pub fn main() {
    
}

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
    assert_eq!(StringSolution::multiply("2".to_string(), "3".to_string()), "6");
    assert_eq!(StringSolution::multiply("123".to_string(), "456".to_string()), "56088");
    assert_eq!(StringSolution::multiply("123456789".to_string(), "987654321".to_string()), "121932631112635269");
}

#[test]
fn test_str_str() {
    assert_eq!(StringSolution::str_str("hello".to_string(), "ll".to_string()), 2);
    assert_eq!(StringSolution::str_str("aaaaa".to_string(), "bba".to_string()), -1);
    assert_eq!(StringSolution::str_str("".to_string(), "".to_string()), 0);
    assert_eq!(StringSolution::str_str("".to_string(), "a".to_string()), -1);
    assert_eq!(StringSolution::str_str("a".to_string(), "".to_string()), 0);
    assert_eq!(StringSolution::str_str("mississippi".to_string(), "issip".to_string()), 4);
}

#[test]
fn test_group_anagrams() {
    let mut result = StringSolution::group_anagrams(vec!["eat".to_string(), "tea".to_string(), "tan".to_string(), "ate".to_string(), "nat".to_string(), "bat".to_string()]);
    for r in result.iter_mut() {
        r.sort();
    }
    result.sort();
    let mut expect = vec![vec!["ate".to_string(), "eat".to_string(), "tea".to_string()], vec!["bat".to_string()], vec!["nat".to_string(), "tan".to_string()]];
    for e in expect.iter_mut() {
        e.sort();
    }
    expect.sort();
    assert_eq!(result, expect);
}
