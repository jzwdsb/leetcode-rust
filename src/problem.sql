/*
table name: Employee
+-------------+------+
| Column Name | Type |
+-------------+------+
| id          | int  |
| salary      | int  |
+-------------+------+
select the second highest salary from the Employee table.
*/

-- Solution:
select (
        select distinct
            salary
        from Employee
        order by salary desc
        limit 1
        offset
            1
    ) as SecondHighestSalary;

/*
same table, select the Nth highest salary from the Employee table.
*/

CREATE FUNCTION getNthHighestSalary(N INT) RETURNS 
INT 
BEGIN 
	SET N = N -1;
	RETURN (
	    SELECT DISTINCT
	        salary
	    FROM Employee
	    ORDER BY salary DESC
	    LIMIT 1
	    OFFSET
	        N
	);
END

/*
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| score       | decimal |
+-------------+---------+

Write a solution to find the rank of the scores. The ranking should be calculated according to the following rules:

The scores should be ranked from the highest to the lowest.
If there is a tie between two scores, both should have the same ranking.
After a tie, the next ranking number should be the next consecutive integer value. In other words, there should be no holes between ranks.
Return the result table ordered by score in descending order.
*/
-- https://www.mysqltutorial.org/mysql-window-functions/mysql-dense_rank-function/
-- dense_rank() function is used to assign a rank to each row within a partition of a result set.
-- The rank of a row is one plus the number of distinct ranks that come before the row in question.
-- Solution:
select score, 
	   dense_rank() over(order by score desc) as rank
from Scores
order by score desc;


/*
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| email       | varchar |
+-------------+---------+
id is the primary key (column with unique values) for this table.
Each row of this table contains an email. The emails will not contain uppercase letters.

Write a solution to report all the duplicate emails. Note that it's guaranteed that the email field is not NULL.
*/

SELECT email 
FROM Person
GROUP BY email
HAVING COUNT(email) > 1;