pub struct SlidingWindow{}

impl SlidingWindow {
    pub fn max_vowels(s: String, k:i32) -> i32 {
        // this could do the work, but time limit exceeded
        // s.as_bytes().windows(k as usize).map(|x| {
        //     x.iter().filter(|&c| {
        //         VOEWLS.contains(&(*c as char))
        //     }).count()
        // }).max().unwrap_or(0) as i32
        let s = s.as_bytes();
        let k = k as usize;
        let count: usize = s.iter().take(k).filter(|&x| Self::is_vowel(*x)).count();
        s.iter()
         .skip(k)
         .zip(s.iter())
         .fold(
             (count, count),
             |a, x| {
                 let (max, count) = a;
                 let (cur, left) = x;
                 match (Self::is_vowel(*left), Self::is_vowel(*cur)) {
                     (false, true) => (max.max(count + 1), count + 1),
                     (true, false) => (max, count - 1),
                     _ => (max, count),
                 }
             }
         ).0 as i32
    }

    fn is_vowel(c: u8) -> bool {
        match c {
            b'a' | b'e' | b'i' | b'o' | b'u' => true,
            _ => false,
        }
    }
}

pub fn main() {

}

#[test]
fn test_max_vowels() {
    assert_eq!(SlidingWindow::max_vowels("abciiidef".to_string(), 3), 3);
    assert_eq!(SlidingWindow::max_vowels("aeiou".to_string(), 2), 2);
    assert_eq!(SlidingWindow::max_vowels("leetcode".to_string(), 3), 2);
    assert_eq!(SlidingWindow::max_vowels("rhythms".to_string(), 4), 0);
    assert_eq!(SlidingWindow::max_vowels("tryhard".to_string(), 4), 1);
}

