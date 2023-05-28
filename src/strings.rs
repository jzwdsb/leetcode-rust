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
        let prev = StringSolution::count_and_say(n-1);
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
    123 * 456 = 123 * 6 + 123 * 50 + 123 * 400
    123 * 6 = 3 * 6 + 20 * 6 + 100 * 6
    123 * 50 = 3 * 50 + 20 * 50 + 100 * 50
    123 * 400 = 3 * 400 + 20 * 400 + 100 * 400

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