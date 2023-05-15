
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
}

#[test]
fn test_reverse() {
    assert_eq!(BasicTypes::reverse(123), 321);
    assert_eq!(BasicTypes::reverse(-123), -321);
    assert_eq!(BasicTypes::reverse(120), 21);
    assert_eq!(BasicTypes::reverse(0), 0);
    assert_eq!(BasicTypes::reverse(900000), 9);
}
