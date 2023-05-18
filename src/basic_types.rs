
pub struct BasicTypes {}

impl BasicTypes {
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
    pub fn divide(dividend: i32, divsor: i32) -> i32 {
        if dividend == divsor {
            return 1
        }
        let is_neg = dividend.is_negative() ^ divsor.is_negative();
        let (mut left, mut right) = (0i64, dividend as i64);
        while left < right {
            let mid = left + ((right+1-left) >> 1);
            if dividend as i64 >= BasicTypes::simulate_multiple(divsor.abs() as i64, mid) {
                left = mid;
            } else {
                right = mid-1;
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
}

#[test]
fn test_reverse() {
    assert_eq!(BasicTypes::reverse(123), 321);
    assert_eq!(BasicTypes::reverse(-123), -321);
    assert_eq!(BasicTypes::reverse(120), 21);
    assert_eq!(BasicTypes::reverse(0), 0);
    assert_eq!(BasicTypes::reverse(900000), 9);
}

#[test]
fn test_divide() {
    assert_eq!(BasicTypes::divide(10, 3), 3);
    assert_eq!(BasicTypes::divide(7, -3), -2);
    assert_eq!(BasicTypes::divide(0, 1), 0);
    assert_eq!(BasicTypes::divide(1, 1), 1);
}