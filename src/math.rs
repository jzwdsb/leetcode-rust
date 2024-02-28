#![allow(dead_code)]

pub struct MathSolution {}

impl MathSolution {
    pub fn reverse(x: i32) -> i32 {
        let mut x = x;
        let mut ans: i32 = 0;

        while x != 0 {
            let pop = x % 10;
            x /= 10;
            if ans > i32::MAX / 10 || (ans == i32::MAX / 10 && pop > 7) {
                return 0;
            }
            if ans < i32::MIN / 10 || (ans == i32::MIN / 10 && pop < -8) {
                return 0;
            }
            ans = ans * 10 + pop;
        }
        ans
    }

    /*
    link: https://leetcode.com/problems/divide-two-integers/
    we can use binary search to search the [0, dividend] to find the answer
    time complexity O(log(dividend))
     */
    pub fn divide(dividend: i32, divisor: i32) -> i32 {
        if dividend == divisor {
            return 1;
        }
        let is_neg = dividend.is_negative() ^ divisor.is_negative();
        let (dividend, divisor) = ((dividend as i64).abs(), (divisor as i64).abs());
        let (mut left, mut right) = (0i64, dividend);
        while left < right {
            let mid = left + ((right + 1 - left) >> 1);
            if dividend >= Self::simulate_multiple(divisor, mid) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        let ans: i64 = match is_neg {
            true => -(left),
            false => left,
        };
        if ans > (i32::MAX as i64) {
            return i32::MAX;
        }
        if ans < (i32::MIN as i64) {
            return i32::MIN;
        }
        ans as i32
    }

    fn simulate_multiple(x: i64, y: i64) -> i64 {
        let mut ans = 0;
        let mut add = x;
        let mut y = y;
        while y > 0 {
            if y & 1 == 1 {
                ans += add;
            }
            y >>= 1;
            add += add;
        }

        ans
    }
    /*
    link: https://leetcode.com/problems/powx-n/
     */

    pub fn pow(x: f64, n: i32) -> f64 {
        if x == 0.0 {
            return 0.0;
        }
        if n == 0 {
            return 1.0;
        }
        if x.abs() == 1.0 {
            return if n & 1 == 1 && x.is_sign_negative() {
                -1.0
            } else {
                1.0
            };
        }
        if n == i32::MIN {
            return 0.0;
        }

        let mut res = 1.0;
        let is_neg = n.is_negative();
        let mut n = n.unsigned_abs();
        let mut x = x;
        while n > 0 {
            if n & 1 == 1 {
                res *= x;
            }
            x *= x;
            n >>= 1;
        }
        if is_neg {
            res = 1.0 / res;
        }
        res
    }

    pub fn my_sqrt(x: i32) -> i32 {
        let mut left = 0_i64;
        let mut right = x as i64;
        while left < right {
            let mid = left + ((right - left + 1) >> 1);
            if mid * mid <= x as i64 {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        left as i32
    }

    pub fn distance_traveled(main_tank: i32, additional_tank: i32) -> i32 {
        (main_tank + ((main_tank - 1) / 4).min(additional_tank)) * 10
    }

    pub fn is_happy(n: i32) -> bool {
        let mut n = n;
        let mut set = std::collections::HashSet::new();
        while n != 1 {
            let mut sum = 0;
            while n > 0 {
                let digit = n % 10;
                sum += digit * digit;
                n /= 10;
            }
            if set.contains(&sum) {
                return false;
            }
            set.insert(sum);
            n = sum;
        }
        true
    }

    pub fn gnerate_pascal_triangle(num_rows: usize) -> Vec<Vec<i32>> {
        let mut res: Vec<Vec<i32>> = vec![];
        for i in 0..num_rows {
            let mut curr_row = vec![1; i + 1];
            if let Some(prev_row) = res.get(i.saturating_sub(1)) {
                for j in 0..=i {
                    if j == i || j == 0 {
                        curr_row[j] = 1;
                    } else {
                        curr_row[j] = prev_row[j - 1] + prev_row[j];
                    }
                }
            }

            res.push(curr_row);
        }

        res
    }

    pub fn get_pascal_triangle_row(row_index: i32) -> Vec<i32> {
        let row_index = row_index as usize;
        let mut res = vec![1; row_index + 1];
        for i in 0..=row_index {
            for j in (1..i).rev() {
                res[j] += res[j - 1];
            }
        }
        res
    }

    pub fn find_content_child(greedy: Vec<i32>, cookies: Vec<i32>) -> usize {
        let mut greedy = greedy;
        let mut cookies = cookies;
        greedy.sort_unstable();
        cookies.sort_unstable();

        let mut i = 0;
        let mut j = 0;
        let mut res = 0;

        while i < greedy.len() && j < cookies.len() {
            if greedy[i] <= cookies[j] {
                res += 1;
                i += 1;
                j += 1;
            } else {
                j += 1;
            }
        }

        res
    }

    /*
    https://leetcode.com/problems/factorial-trailing-zeroes/description/

    find out how many trailing zeroes in n!

    the number of trailing zeroes is determined by the number of 5 in n!
    5 * 2 = 10, so we only need to count the number of 5 in n!

     */

    pub fn trailing_zeroes(n: i32) -> i32 {
        let mut n = n;
        let mut res = 0;
        while n > 0 {
            res += n / 5;
            n /= 5;
        }
        res
    }

    /*
    https://leetcode.com/problems/gray-code/description/

    The gray code is a binary numeral system where two successive values differ in only one bit.
     */

    pub fn gray_code(n: i32) -> Vec<i32> {
        let mut res = vec![0]; // start with 0
        for i in 0..n {
            let mut curr = res.clone();
            curr.reverse();
            let add = 1 << i;
            for e in &mut curr {
                *e += add;
            }
            res.append(&mut curr);
        }
        res
    }

    pub fn fraction_to_decimal(numerator: i32, denominator: i32) -> String {
        let mut res = String::new();
        let mut numerator = numerator as i64;
        let mut denominator = denominator as i64;
        if (numerator.is_negative() ^ denominator.is_negative()) && numerator != 0 {
            res.push('-');
        }
        numerator = numerator.abs();
        denominator = denominator.abs();
        res.push_str(&(numerator / denominator).to_string());
        let mut remainder = numerator % denominator;
        if remainder == 0 {
            return res;
        }
        res.push('.');
        let mut map = std::collections::HashMap::new();
        while remainder != 0 {
            if let Some(&index) = map.get(&remainder) {
                // find the repeating part
                res.insert(index, '(');
                res.push(')');
                break;
            }
            map.insert(remainder, res.len());
            remainder *= 10;
            res.push_str(&(remainder / denominator).to_string());
            remainder %= denominator;
        }
        res
    }

    pub fn get_permutation(n: i32, k: i32) -> String {
        let mut nums: Vec<i32> = (1..=n).collect();
        let mut res = String::new();
        let mut k = k - 1;
        let mut factorial = (1..n).product::<i32>();
        for i in (1..=n).rev() {
            let index = (k / factorial) as usize;
            res.push_str(&nums[index].to_string());
            nums.remove(index);
            if i > 1 {
                k %= factorial;
                factorial /= i - 1;
            }
        }
        res
    }

    pub fn can_messure_water(jug1: i32, jug2: i32, target_capacity: i32) -> bool {
        if jug1 + jug2 < target_capacity {
            return false;
        }
        if jug1 == target_capacity || jug2 == target_capacity || jug1 + jug2 == target_capacity {
            return true;
        }
        if jug1 == 0 || jug2 == 0 {
            return target_capacity == 0;
        }
        target_capacity % Self::gcd(jug1, jug2) == 0
    }

    fn gcd(a: i32, b: i32) -> i32 {
        if b == 0 {
            return a;
        }
        Self::gcd(b, a % b)
    }

    /*
    break n into the sum of positive integers, find the maximum product of those integers

    we can get a conculsion that the maximum product is the exponent.
    product = n^x. so there are two factors, n and x.
    the x gives more contribution to the product based on math knowledge.
    so we need to break n into small numbers as much as possible.
    2 and 3 are the best numbers to break n into.
    we can do some test to find which one is better
    n = 9, 3 * 3 = 9, 2 * 2 * 2 * 1 = 8
    n = 10, 3 * 3 * 3 * 1 = 27, 2 * 2 * 2 * 2 * 1 = 16
    n = 11, 3 * 3 * 3 * 2 = 54, 2 * 2 * 2 * 2 * 2 * 1 = 32

    so we can get the conclusion that we need to break n into 3 as much as possible.
    when we break the num into 2, we actually break it into 4, because 2 + 2 = 2 * 2 = 4,
    which decrease the x in the product.

    but if reminder is 4, we can break it into 2 * 2, so we can get the maximum product 4.
    3 * 1 < 2 * 2, so we need to break 4 into 2 and 2
     */

    pub fn integer_break(n: i32) -> i32 {
        if n == 2 {
            return 1;
        }
        if n == 3 {
            return 2;
        }
        let mut res = 1;
        let mut n = n;
        while n > 4 {
            res *= 3;
            n -= 3;
        }
        res * n
    }

    /*
    if a number is power of two, then there should be only one bit is 1 in the binary representation
    8 = 1000
    n-1 will flip all the bits after the rightmost 1
    8-1 = 7 = 0111
    so n & (n-1) should be 0
    */
    pub fn is_power_of_two(n: i32) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }

    /*
    https://leetcode.com/problems/bitwise-and-of-numbers-range/
    if right > left, then last bit must be zero, because the last bit changes every increment
    then we can shift right and left to the right by 1, and do the same thing again
    and then shift the result to the left by 1
    if right == left, the bitwise itself is still itself, we can return it directly

     */

    pub fn range_bitwise_and(left: i32, right: i32) -> i32 {
        if right > left {
            Self::range_bitwise_and(left.wrapping_shr(1), right.wrapping_shr(1)).wrapping_shl(1)
        } else {
            right
        }
    }

    /*
    https://leetcode.com/problems/arranging-coins/
    use n number of coins to form a staircase, find the number of complete rows
     */

    pub fn arrange_coins(n: i32) -> i32 {
        // binary search get TLE error
        // let mut left = 0;
        // let mut right = n;
        // while left < right {
        //     let mid = (left + right + 1) / 2;
        //     if mid * (mid + 1) / 2 <= n {
        //         left = mid;
        //     } else {
        //         right = mid - 1;
        //     }
        // }
        // left
        (((8.0 * n as f64 + 1.0).sqrt() - 1.0) / 2.0) as i32
    }
} // impl MathSolution

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_reverse() {
        assert_eq!(MathSolution::reverse(123), 321);
        assert_eq!(MathSolution::reverse(-123), -321);
        assert_eq!(MathSolution::reverse(120), 21);
        assert_eq!(MathSolution::reverse(0), 0);
        assert_eq!(MathSolution::reverse(900000), 9);
    }

    #[test]
    fn test_divide() {
        assert_eq!(MathSolution::divide(10, 3), 3);
        assert_eq!(MathSolution::divide(7, -3), -2);
        assert_eq!(MathSolution::divide(0, 1), 0);
        assert_eq!(MathSolution::divide(1, 1), 1);
        assert_eq!(MathSolution::divide(-1, 1), -1);
        assert_eq!(MathSolution::divide(-2147483648, -1), 2147483647);
        assert_eq!(MathSolution::divide(-1010369383, -2147483648), 0);
    }

    #[test]
    fn test_pow() {
        assert!(MathSolution::pow(2.0, 10) - 1024.0 < 0.00001);
        assert!(MathSolution::pow(2.1, 3) - 9.261 < 0.00001);
        assert!(MathSolution::pow(2.0, -2) - 0.25 < 0.00001);
        assert!(MathSolution::pow(0.00001, 2147483647) - 0.0 < 0.00001);
        assert!(MathSolution::pow(1.0, 2147483647) - 1.0 < 0.00001);

        assert!(MathSolution::pow(2.0, -2147483648) - 0.0 < 0.00001);
    }

    #[test]
    fn test_my_sqrt() {
        assert_eq!(MathSolution::my_sqrt(0), 0);
        assert_eq!(MathSolution::my_sqrt(1), 1);
        assert_eq!(MathSolution::my_sqrt(4), 2);
        assert_eq!(MathSolution::my_sqrt(8), 2);
        assert_eq!(MathSolution::my_sqrt(9), 3);
        assert_eq!(MathSolution::my_sqrt(2147395599), 46339);
    }

    #[test]
    fn test_distance_traveled() {
        assert_eq!(MathSolution::distance_traveled(5, 10), 60);
        assert_eq!(MathSolution::distance_traveled(1, 2), 10);
        assert_eq!(MathSolution::distance_traveled(9, 2), 110);
    }

    #[test]
    fn test_is_happy() {
        assert!(MathSolution::is_happy(19));
        assert!(!MathSolution::is_happy(2));
    }

    #[test]
    fn test_pascal_triangle() {
        assert_eq!(
            MathSolution::gnerate_pascal_triangle(5),
            vec![
                vec![1],
                vec![1, 1],
                vec![1, 2, 1],
                vec![1, 3, 3, 1],
                vec![1, 4, 6, 4, 1]
            ]
        );
    }

    #[test]
    fn test_get_row_in_pascal_triangle() {
        assert_eq!(MathSolution::get_pascal_triangle_row(3), vec![1, 3, 3, 1]);
        assert_eq!(MathSolution::get_pascal_triangle_row(0), vec![1]);
        assert_eq!(MathSolution::get_pascal_triangle_row(1), vec![1, 1]);
    }

    #[test]
    fn test_find_content_child() {
        assert_eq!(
            MathSolution::find_content_child(vec![1, 2, 3], vec![1, 1]),
            1
        );
        assert_eq!(
            MathSolution::find_content_child(vec![1, 2], vec![1, 2, 3]),
            2
        );

        assert_eq!(
            MathSolution::find_content_child(vec![10, 9, 8, 7], vec![5, 6, 7, 8]),
            2
        )
    }

    #[test]
    fn test_trailing_zeros() {
        assert_eq!(MathSolution::trailing_zeroes(3), 0);
        assert_eq!(MathSolution::trailing_zeroes(5), 1);
        assert_eq!(MathSolution::trailing_zeroes(0), 0);
        assert_eq!(MathSolution::trailing_zeroes(10), 2);
        assert_eq!(MathSolution::trailing_zeroes(30), 7);
    }

    #[test]
    fn test_gray_code() {
        assert_eq!(MathSolution::gray_code(2), vec![0, 1, 3, 2]);
        assert_eq!(MathSolution::gray_code(0), vec![0]);
    }

    #[test]
    fn test_fraction_to_decimal() {
        assert_eq!(MathSolution::fraction_to_decimal(1, 2), "0.5");
        assert_eq!(MathSolution::fraction_to_decimal(2, 1), "2");
        assert_eq!(MathSolution::fraction_to_decimal(2, 3), "0.(6)");
        assert_eq!(MathSolution::fraction_to_decimal(4, 333), "0.(012)");
        assert_eq!(MathSolution::fraction_to_decimal(1, 5), "0.2");
        assert_eq!(MathSolution::fraction_to_decimal(1, 6), "0.1(6)");
        assert_eq!(MathSolution::fraction_to_decimal(1, 7), "0.(142857)");
        assert_eq!(MathSolution::fraction_to_decimal(1, 90), "0.0(1)");
        assert_eq!(MathSolution::fraction_to_decimal(1, 99), "0.(01)");
        assert_eq!(MathSolution::fraction_to_decimal(0, -5), "0");
        assert_eq!(MathSolution::fraction_to_decimal(-22, -11), "2");
    }

    #[test]
    fn test_get_permutation() {
        assert_eq!(MathSolution::get_permutation(3, 3), "213");
        assert_eq!(MathSolution::get_permutation(4, 9), "2314");
        assert_eq!(MathSolution::get_permutation(3, 1), "123");
        assert_eq!(MathSolution::get_permutation(3, 2), "132");
        assert_eq!(MathSolution::get_permutation(3, 4), "231");
        assert_eq!(MathSolution::get_permutation(3, 5), "312");
        assert_eq!(MathSolution::get_permutation(3, 6), "321");
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(MathSolution::is_power_of_two(1));
        assert!(MathSolution::is_power_of_two(16));
        assert!(!MathSolution::is_power_of_two(218));
    }

    #[test]
    fn test_range_bitwise_and() {
        assert_eq!(MathSolution::range_bitwise_and(5, 7), 4);
        assert_eq!(MathSolution::range_bitwise_and(0, 1), 0);
        assert_eq!(MathSolution::range_bitwise_and(1, 2147483647), 0);
    }

    #[test]
    fn test_arrange_coins() {
        assert_eq!(MathSolution::arrange_coins(5), 2);
        assert_eq!(MathSolution::arrange_coins(8), 3);
    }
}
