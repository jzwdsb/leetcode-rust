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
            true => -(left as i64),
            false => left as i64,
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
            y = y >> 1;
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
        let mut n = n.abs() as u32;
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
        let mut left = 0 as i64;
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
}

pub fn main() {}

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
