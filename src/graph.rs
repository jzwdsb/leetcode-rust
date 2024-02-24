#![allow(dead_code)]

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::rc::Rc;

struct Node {
    val: i32,
    nodes: Vec<Rc<RefCell<Node>>>,
}

pub struct GraphSolution {}

impl GraphSolution {
    /*
    link: https://leetcode.com/problems/course-schedule/
    check whether it is possible to finish all courses.
    travel all the nodes of the graph using DFS and check if there is a cycle or unvisited node.
     */

    pub fn can_finish(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        let mut graph = vec![vec![]; num_courses as usize];
        let mut visited = vec![0; num_courses as usize];

        for edge in prerequisites {
            graph[edge[0] as usize].push(edge[1] as usize);
        }

        for i in 0..num_courses {
            if !Self::dfs(&graph, &mut visited, i as usize) {
                return false;
            }
        }
        true
    }

    fn dfs(graph: &Vec<Vec<usize>>, visited: &mut Vec<i32>, i: usize) -> bool {
        if visited[i] == 1 {
            return false;
        }
        if visited[i] == -1 {
            return true;
        }
        visited[i] = 1;
        for &j in &graph[i] {
            if !Self::dfs(graph, visited, j) {
                return false;
            }
        }
        visited[i] = -1;
        true
    }

    /*
    link: https://leetcode.com/problems/maximal-network-rank/
    find the maximal network rank of the given cities.
    network rank of two different cities is defined as the total number of directly connected roads to either city.
    the maximal network rank is the maximum network rank of all pairs of different cities.

    1. create a graph using adjacency list
    2. find the network rank of each pair of cities
    3. return the maximum network rank
     */

    pub fn maximal_network_rank(n: i32, roads: Vec<Vec<i32>>) -> i32 {
        // adjacency list
        let mut graph = vec![vec![]; n as usize];
        let mut max = 0;
        for road in roads {
            graph[road[0] as usize].push(road[1] as usize);
            graph[road[1] as usize].push(road[0] as usize);
        }

        // find the network rank of each pair of cities
        // update the maximum network rank
        for i in 0..n {
            for j in i + 1..n {
                // find the network rank of i and j
                let mut count = graph[i as usize].len() + graph[j as usize].len();
                if graph[i as usize].contains(&(j as usize)) {
                    // if i and j are directly connected, the the road is counted twice
                    // so we need to subtract 1
                    count -= 1;
                }
                max = max.max(count);
            }
        }
        max as i32
    }

    /*
    link: https://leetcode.com/problems/path-with-minimum-effort
    plain dfs with visited array will cause TLE

    we can use dijkstra with heap to solve this problem
    use heap to store the cost and position

     */

    pub fn minimum_effort_path(heights: Vec<Vec<i32>>) -> i32 {
        let direction = [(0, 1), (0, -1i32), (1, 0), (-1i32, 0)];
        let mut visited = vec![vec![false; heights[0].len()]; heights.len()];
        // binary heap will sort the elements in the heap by the first element of the tuple
        // so we need to reverse the cost to make the heap sort the cost in ascending order
        let mut heap = BinaryHeap::<(Reverse<i32>, usize, usize)>::new();
        heap.push((Reverse(0), 0, 0));
        loop {
            let (Reverse(cost), x, y) = heap.pop().unwrap();
            if x == heights.len() - 1 && y == heights[0].len() - 1 {
                // reach the destination, return the cost
                break cost;
            }
            if visited[x][y] {
                continue;
            }
            visited[x][y] = true;
            for (dx, dy) in direction {
                let nx = x.wrapping_add_signed(dx as isize);
                let ny = y.wrapping_add_signed(dy as isize);
                if nx < heights.len() && ny < heights[0].len() && !visited[nx][ny] {
                    // the maximum cost of the path from (x, y) to (nx, ny) is the maximum of the current cost
                    // and the difference between the heights of (nx, ny) and (x, y)
                    let cost2 = cost.max((heights[nx][ny] - heights[x][y]).abs());
                    heap.push((Reverse(cost2), nx, ny));
                }
            }
        }
    }
    /*
    https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

    implemetation of dijkstra algorithm
     */
    pub fn dijkstra() -> i32 {
        todo!("dijkstra")
    }

    // a[i]: the start of road i
    // b[i]: the end of road i
    // n: citys
    pub fn max_network_rank(a: Vec<i32>, b: Vec<i32>, _n: i32) -> i32 {
        let mut max = 0;
        let mut count = HashMap::<i32, i32>::new();
        for i in 0..a.len() {
            count.entry(a[i]).and_modify(|v| *v += 1).or_insert(1);
            count.entry(b[i]).and_modify(|v| *v += 1).or_insert(1);
        }
        for i in 0..a.len() {
            max = max.max(count.get(&a[i]).unwrap() + count.get(&b[i]).unwrap() - 1)
        }

        max
    }

    /*
    link: https://leetcode.com/problems/detonate-the-maximum-bombs/
    intput is a vector of tuple (x, y, r), x,y is the coordinate of the bomb
    r is the radius of the bomb,
    return the maximum number of bombs that can be detonated with only detonating one bomb

    build the adjacency matrix of the graph first
    then travel all the nodes of the graph using DFS & backtracking to
    find the maximum depth of the graph

    almost time limit exceeded
    need to optimize the algorithm
     */
    pub fn maximum_detonation(bombs: Vec<(i32, i32, i32)>) -> i32 {
        let mut ans = 0;
        let mut graph = vec![vec![0; bombs.len()]; bombs.len()];

        for i in 0..bombs.len() {
            for j in i + 1..bombs.len() {
                if i == j {
                    continue;
                }
                let distance = Self::distance(&bombs[i], &bombs[j]);
                if distance <= bombs[i].2 as f64 {
                    graph[i][j] = 1;
                }
                if distance <= bombs[j].2 as f64 {
                    graph[j][i] = 1;
                }
            }
        }

        for i in 0..bombs.len() {
            let mut path = vec![];
            ans = ans.max(Self::walk_bomb_graph(&graph, &mut path, i));
        }

        ans as i32
    }

    fn distance(a: &(i32, i32, i32), b: &(i32, i32, i32)) -> f64 {
        let (x1, y1, _) = a;
        let (x2, y2, _) = b;
        (((*x1 as i64 - *x2 as i64).pow(2) + (*y1 as i64 - *y2 as i64).pow(2)) as f64).sqrt()
    }

    fn walk_bomb_graph(graph: &Vec<Vec<usize>>, path: &mut Vec<usize>, pos: usize) -> usize {
        if path.contains(&pos) {
            return path.len();
        }

        path.push(pos);

        let mut depth = path.len();
        for next in 0..graph[pos].len() {
            if graph[pos][next] == 0 {
                continue;
            }
            depth = depth.max(Self::walk_bomb_graph(graph, path, next));
        }

        depth
    }

    pub fn word_search(board: Vec<Vec<char>>, word: String) -> bool {
        let mut visited = vec![vec![false; board[0].len()]; board.len()];
        let mut ans = false;
        for i in 0..board.len() {
            for j in 0..board[0].len() {
                if board[i][j] == word.chars().next_back().unwrap() {
                    ans = ans
                        || Self::dfs_word_search(
                            &board,
                            &mut visited,
                            i as i32,
                            j as i32,
                            &word,
                            0,
                        );
                    if ans {
                        return ans;
                    }
                }
            }
        }
        ans
    }

    fn dfs_word_search(
        board: &Vec<Vec<char>>,
        visited: &mut Vec<Vec<bool>>,
        i: i32,
        j: i32,
        word: &str,
        pos: usize,
    ) -> bool {
        if pos == word.len() {
            return true;
        }
        if i >= board.len() as i32
            || j >= board[0].len() as i32
            || i < 0
            || j < 0
            || visited[i as usize][j as usize]
        {
            return false;
        }
        if board[i as usize][j as usize] != word.chars().nth(pos).unwrap() {
            return false;
        }
        visited[i as usize][j as usize] = true;
        let ans = Self::dfs_word_search(board, visited, i + 1, j, word, pos + 1)
            || Self::dfs_word_search(board, visited, i, j + 1, word, pos + 1)
            || Self::dfs_word_search(board, visited, i - 1, j, word, pos + 1)
            || Self::dfs_word_search(board, visited, i, j - 1, word, pos + 1);
        visited[i as usize][j as usize] = false;
        ans
    }

    pub fn shortest_path_with_max_coin(layers: Vec<Vec<&str>>) -> i32 {
        let mut visited = vec![vec![false; layers[0].len()]; layers.len()];

        let mut deque = VecDeque::new();

        let mut start = (0i32, 0i32);
        let mut goal = (0i32, 0i32);
        for i in 0..layers.len() {
            for j in 0..layers[0].len() {
                if layers[i][j] == "S" {
                    start = (i as i32, j as i32);
                }
                if layers[i][j] == "G" {
                    goal = (i as i32, j as i32);
                }
            }
        }
        deque.push_back((start.0, start.1, 0));
        let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let left_up = (start.0.min(goal.0), start.1.min(goal.1));
        let right_down = (start.0.max(goal.0), start.1.max(goal.1));
        let mut max_coins = -1;
        while let Some((x, y, coin)) = deque.pop_front() {
            if x == goal.0 && y == goal.1 {
                max_coins = max_coins.max(coin);
                continue;
            }
            visited[x as usize][y as usize] = true;
            for (dx, dy) in directions {
                let nx = x.wrapping_add(dx);
                let ny = y.wrapping_add(dy);
                if nx < left_up.0
                    || nx > right_down.0
                    || ny < left_up.1
                    || ny > right_down.1
                    || visited[nx as usize][ny as usize]
                {
                    continue;
                }
                let mut next_coin = coin;
                if layers[nx as usize][ny as usize] != "G" {
                    next_coin += layers[nx as usize][ny as usize].parse::<i32>().unwrap();
                }
                deque.push_back((nx, ny, next_coin));
            }
        }

        max_coins
    }

    pub fn word_ladder(start: String, end: String, word_list: Vec<String>) -> i32 {
        let mut visited = vec![false; word_list.len()];
        let mut deque = VecDeque::new();
        deque.push_back((start, 1));
        while let Some((word, step)) = deque.pop_front() {
            if word == end {
                return step;
            }
            for i in 0..word_list.len() {
                if visited[i] {
                    continue;
                }
                if Self::is_one_char_diff(&word, &word_list[i]) {
                    deque.push_back((word_list[i].clone(), step + 1));
                    visited[i] = true;
                }
            }
        }
        0
    }

    fn is_one_char_diff(a: &str, b: &str) -> bool {
        let mut diff = 0;
        for i in 0..a.len() {
            if a.chars().nth(i).unwrap() != b.chars().nth(i).unwrap() {
                diff += 1;
                if diff > 1 {
                    return false;
                }
            }
        }
        diff == 1
    }

    fn breadth_first_traversal(node: Rc<RefCell<Node>>) -> Vec<i32> {
        let mut queue = VecDeque::new();
        let mut visited = [false; 100];
        let mut ans = vec![];
        queue.push_back(node);
        while let Some(node) = queue.pop_front() {
            let node = node.borrow();
            if visited[node.val as usize] {
                continue;
            }
            visited[node.val as usize] = true;
            ans.push(node.val);
            for child in node.nodes.iter() {
                queue.push_back(child.clone());
            }
        }
        ans
    }

    /*
    https://leetcode.com/problems/cheapest-flights-within-k-stops/

    find the most cheapest price from src to dst with at most k stops.

    https://en.wikipedia.org/wiki/Bellman-Ford_algorithm

     */

    pub fn find_cheapest_price(
        n: i32,                        // number of cities
        flights: Vec<(i32, i32, i32)>, // flights[i] = (src, dst, price)
        src: i32,                      // source city
        dst: i32,                      // destination city
        k: i32,                        // maximum number of stops
    ) -> i32 {
        let mut matrix = vec![vec![0; n as usize]; n as usize];
        for flight in flights {
            matrix[flight.0 as usize][flight.1 as usize] = flight.2;
        }
        let mut cost = vec![i32::MAX; n as usize];
        cost[src as usize] = 0;
        for _ in 0..=k {
            let mut cost_tmp = cost.clone();
            for i in 0..n as usize {
                if cost[i] == i32::MAX {
                    continue;
                }
                for (j, &price) in matrix[i].iter().enumerate() {
                    if price == 0 {
                        continue;
                    }
                    cost_tmp[j] = cost_tmp[j].min(cost[i] + price);
                }
            }
            cost = cost_tmp;
        }

        if cost[dst as usize] == i32::MAX {
            -1
        } else {
            cost[dst as usize]
        }
    }
    /*
    https://leetcode.com/problems/find-all-people-with-secret/
     */

    pub fn find_all_people(
        n: i32,
        mut meetings: Vec<(i32, i32, i32)>,
        first_person: i32,
    ) -> Vec<i32> {
        meetings.sort_unstable_by_key(|x| x.2);
        let mut parent = (0..n as usize).collect::<Vec<usize>>();
        parent[first_person as usize] = 0; // disjoint set
        let mut i = 0;
        while i < meetings.len() {
            let mut j = i;
            // merge people who are in the same group
            while j < meetings.len() && meetings[j].2 == meetings[i].2 {
                let mut a = Self::find_root(&mut parent, meetings[j].0 as usize);
                let mut b = Self::find_root(&mut parent, meetings[j].1 as usize);

                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                parent[b] = a;
                j += 1;
            }
            j = i;
            // reset each person who is not in the same group as the first person
            while j < meetings.len() && meetings[j].2 == meetings[i].2 {
                if Self::find_root(&mut parent, meetings[j].0 as usize) != 0 {
                    parent[meetings[j].0 as usize] = meetings[j].0 as usize;
                }
                if Self::find_root(&mut parent, meetings[j].1 as usize) != 0 {
                    parent[meetings[j].1 as usize] = meetings[j].1 as usize;
                }
                j += 1;
            }
            i = j;
        }
        let mut ans = Vec::new();
        // find all the people who are in the same group as the 0
        for i in 0..parent.len() {
            // if the root of the disjoint set is 0
            // then the person is in the same group as the first person
            if Self::find_root(&mut parent, i) == 0 {
                ans.push(i as i32);
            }
        }

        ans
    }

    // find the root of the disjoint set for given i
    fn find_root(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            parent[i] = Self::find_root(parent, parent[i]);
        }
        parent[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_finish() {
        assert_eq!(GraphSolution::can_finish(2, vec![vec![1, 0]]), true);
        assert_eq!(
            GraphSolution::can_finish(2, vec![vec![1, 0], vec![0, 1]]),
            false
        );
    }

    #[test]
    fn test_maximal_network_rank() {
        assert_eq!(
            GraphSolution::maximal_network_rank(
                4,
                vec![vec![0, 1], vec![0, 3], vec![1, 2], vec![1, 3]]
            ),
            4
        );
        assert_eq!(
            GraphSolution::maximal_network_rank(
                5,
                vec![
                    vec![0, 1],
                    vec![0, 3],
                    vec![1, 2],
                    vec![1, 3],
                    vec![2, 3],
                    vec![2, 4]
                ],
            ),
            5
        );
    }

    #[test]
    fn test_minimum_effort_path() {
        assert_eq!(
            GraphSolution::minimum_effort_path(vec![vec![1, 2, 2], vec![3, 8, 2], vec![5, 3, 5]]),
            2
        );
        assert_eq!(
            GraphSolution::minimum_effort_path(vec![vec![1, 2, 3], vec![3, 8, 4], vec![5, 3, 5]]),
            1
        );
        assert_eq!(
            GraphSolution::minimum_effort_path(vec![
                vec![1, 2, 1, 1, 1],
                vec![1, 2, 1, 2, 1],
                vec![1, 2, 1, 2, 1],
                vec![1, 2, 1, 2, 1],
                vec![1, 1, 1, 2, 1]
            ]),
            0
        );
        assert_eq!(
            GraphSolution::minimum_effort_path(vec![vec![1, 10, 6, 7, 9, 10, 4, 9]]),
            9
        );
    }

    #[test]
    fn test_max_network_rank() {
        // TODO: add test case for max_network_rank
    }

    #[test]
    fn test_maximum_detonation() {
        assert_eq!(
            GraphSolution::maximum_detonation(vec![(2, 1, 3), (6, 1, 4)]),
            2
        );
        assert_eq!(
            GraphSolution::maximum_detonation(vec![(1, 1, 5), (10, 10, 5)]),
            1
        );
        assert_eq!(
            GraphSolution::maximum_detonation(vec![
                (1, 2, 3),
                (2, 3, 1),
                (3, 4, 2),
                (4, 5, 3),
                (5, 6, 4)
            ]),
            5
        );
        assert_eq!(
            GraphSolution::maximum_detonation(vec![(2, 1, 3), (6, 1, 4)]),
            2
        );
        assert_eq!(
            GraphSolution::maximum_detonation(vec![
                (54, 95, 4),
                (99, 46, 3),
                (29, 21, 3),
                (96, 72, 8),
                (49, 43, 3),
                (11, 20, 3),
                (2, 57, 1),
                (69, 51, 7),
                (97, 1, 10),
                (85, 45, 2),
                (38, 47, 1),
                (83, 75, 3),
                (65, 59, 3),
                (33, 4, 1),
                (32, 10, 2),
                (20, 97, 8),
                (35, 37, 3)
            ]),
            1
        );
        assert_eq!(
            GraphSolution::maximum_detonation(vec![
                (85024, 58997, 3532),
                (65196, 42043, 9739),
                (85872, 75029, 3117),
                (73014, 91183, 7092),
                (29098, 40864, 7624),
                (11469, 13607, 4315),
                (98722, 69681, 9656),
                (75140, 42250, 421),
                (92580, 44040, 4779),
                (58474, 78273, 1047),
                (27683, 4203, 6186),
                (10714, 24238, 6243),
                (60138, 81791, 3496),
                (16227, 92418, 5622),
                (60496, 64917, 2463),
                (59241, 62074, 885),
                (11961, 163, 5815),
                (37757, 43214, 3402),
                (21094, 98519, 1678),
                (49368, 22385, 1431),
                (6343, 53798, 159),
                (80129, 9282, 5139),
                (69565, 32036, 6827),
                (59372, 64978, 6575),
                (44948, 71199, 7095),
                (46390, 91701, 1667),
                (37144, 98691, 8128),
                (13558, 81505, 4653),
                (41234, 48161, 9304),
                (14852, 3206, 5369)
            ]),
            3
        );
    }

    #[test]
    fn test_word_search() {
        assert_eq!(
            GraphSolution::shortest_path_with_max_coin(vec![
                vec!["S", "3", "2", "4"],
                vec!["1", "2", "G", "1"],
                vec!["2", "4", "2", "5"],
                vec!["7", "1", "4", "4"],
            ]),
            5
        );
        assert_eq!(
            GraphSolution::shortest_path_with_max_coin(vec![
                vec!["6", "3", "2", "G"],
                vec!["3", "2", "3", "1"],
                vec!["2", "3", "8", "5"],
                vec!["S", "1", "4", "3"],
            ]),
            19
        );
    }

    #[test]
    fn test_word_ladder() {
        assert_eq!(
            GraphSolution::word_ladder(
                "hit".to_string(),
                "cog".to_string(),
                vec![
                    "hot".to_string(),
                    "dot".to_string(),
                    "dog".to_string(),
                    "lot".to_string(),
                    "log".to_string(),
                    "cog".to_string()
                ]
            ),
            5
        );
        assert_eq!(
            GraphSolution::word_ladder(
                "hit".to_string(),
                "cog".to_string(),
                vec![
                    "hot".to_string(),
                    "dot".to_string(),
                    "dog".to_string(),
                    "lot".to_string(),
                    "log".to_string()
                ]
            ),
            0
        );
    }

    #[test]
    fn test_find_cheapest_price() {
        assert_eq!(
            GraphSolution::find_cheapest_price(
                3,
                vec![(0, 1, 100), (1, 2, 100), (0, 2, 500)],
                0,
                2,
                1
            ),
            200
        );
        assert_eq!(
            GraphSolution::find_cheapest_price(
                3,
                vec![(0, 1, 100), (1, 2, 100), (0, 2, 500)],
                0,
                2,
                0
            ),
            500
        );
        assert_eq!(
            GraphSolution::find_cheapest_price(
                4,
                vec![
                    (0, 1, 100),
                    (1, 2, 100),
                    (2, 0, 100),
                    (1, 3, 600),
                    (2, 3, 200)
                ],
                0,
                3,
                1
            ),
            700
        );
        assert_eq!(
            GraphSolution::find_cheapest_price(
                10,
                vec![
                    (3, 4, 4),
                    (2, 5, 6),
                    (4, 7, 10),
                    (9, 6, 5),
                    (7, 4, 4),
                    (6, 2, 10),
                    (6, 8, 6),
                    (7, 9, 4),
                    (1, 5, 4),
                    (1, 0, 4),
                    (9, 7, 3),
                    (7, 0, 5),
                    (6, 5, 8),
                    (1, 7, 6),
                    (4, 0, 9),
                    (5, 9, 1),
                    (8, 7, 3),
                    (1, 2, 6),
                    (4, 1, 5),
                    (5, 2, 4),
                    (1, 9, 1),
                    (7, 8, 10),
                    (0, 4, 2),
                    (7, 2, 8)
                ],
                6,
                0,
                7
            ),
            14
        );
    }

    #[test]
    fn test_find_all_people() {
        assert_eq!(
            GraphSolution::find_all_people(6, vec![(1, 2, 5), (2, 3, 8), (1, 5, 10)], 1),
            vec![0, 1, 2, 3, 5]
        );
        assert_eq!(
            GraphSolution::find_all_people(4, vec![(3, 1, 3), (1, 2, 2), (0, 3, 3)], 3),
            vec![0, 1, 3]
        );
        assert_eq!(
            GraphSolution::find_all_people(6, vec![(0, 2, 1), (1, 3, 1), (4, 5, 1)], 1),
            vec![0, 1, 2, 3]
        )
    }
}
